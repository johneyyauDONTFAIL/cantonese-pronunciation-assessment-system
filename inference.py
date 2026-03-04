import torch
import torchaudio
import json
import sys
import os
import warnings
import torch
import torchaudio
import librosa
import numpy as np
from transformers import (
    Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer,
    SeamlessM4TFeatureExtractor
)
from model import Wav2Vec2BertForCantonese 
warnings.filterwarnings("ignore")
model_id = "hon9kon9ize/wav2vec2bert-jyutping"
PHONEME_THRESHOLD_GOOD = 20.0
PHONEME_THRESHOLD_OK = 15.0
TONE_THRESHOLD_GOOD = 10.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#initialize model and processor
jyutping_tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)
tone_tokenizer = Wav2Vec2CTCTokenizer(
    "tone_vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)

feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_id)
processor = Wav2Vec2BertProcessor(
    feature_extractor=feature_extractor, tokenizer=jyutping_tokenizer
)

model = Wav2Vec2BertForCantonese.from_pretrained(
    model_id,
    attention_dropout=0.2,
    hidden_dropout=0.2,
    feat_proj_dropout=0.0,
    mask_time_prob=0.0,
    layerdrop=0.0,
    add_adapter=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
).eval().to(DEVICE)

def ctc_viterbi_alignment(logits, labels):
    num_frames, num_classes = logits.shape
    logits_np = logits.cpu().numpy()
    
    dp = np.full((num_frames, len(labels)), -np.inf)
    backpointer = np.zeros((num_frames, len(labels)), dtype=int)
    dp[0, 0] = logits_np[0, labels[0]]
    
    for t in range(1, num_frames):
        for j in range(len(labels)):
            label_idx = labels[j]
            candidates = []
            if j > 0:
                candidates.append(dp[t-1, j-1])
            candidates.append(dp[t-1, j])
            
            best_idx = np.argmax(candidates)
            prev_j = j - 1 if best_idx == 0 and j > 0 else j
            
            dp[t, j] = logits_np[t, label_idx] + dp[t-1, prev_j]
            backpointer[t, j] = prev_j
    
    alignment = np.zeros(num_frames, dtype=int)
    j = len(labels) - 1
    for t in range(num_frames - 1, -1, -1):
        alignment[t] = labels[j] if j >= 0 else 0
        if t > 0:
            j = backpointer[t, j]
    
    return alignment, dp, backpointer

def calculate_gop_maxlogit(logits, alignment, blank_id=2):
    logits_np = logits.cpu().numpy()
    gop_scores = np.full(len(alignment), np.nan)
    for label_idx in np.unique(alignment):
        if label_idx == blank_id:
            continue
            
        frames_for_label = np.where(alignment == label_idx)[0]
        if len(frames_for_label) == 0:
            continue
            
        ref_logits = logits_np[frames_for_label, label_idx]
        max_logit = np.max(ref_logits)
        
        gop_scores[frames_for_label] = max_logit
    
    return gop_scores

def segment_by_labels(alignment, gop_scores, labels, tokenizer, blank_id=2):
    segments = {}
    current_label = None
    segment_start = 0
    
    for t, label_idx in enumerate(alignment):
        if label_idx == blank_id:
            if current_label is not None:
                segments[int(current_label)] = {
                    'start_frame': segment_start,
                    'end_frame': t,
                    'duration_frames': t - segment_start,
                    'gop_maxlogit': float(gop_scores[segment_start]),
                    'label_name': tokenizer.decode([int(current_label)])
                }
            current_label = None
            continue
        
        if label_idx != current_label:
            if current_label is not None:
                segments[int(current_label)] = {
                    'start_frame': segment_start,
                    'end_frame': t,
                    'duration_frames': t - segment_start,
                    'gop_maxlogit': float(gop_scores[segment_start]),
                    'label_name': tokenizer.decode([int(current_label)])
                }
            current_label = label_idx
            segment_start = t
    
    if current_label is not None:
        segments[int(current_label)] = {
            'start_frame': segment_start,
            'end_frame': len(alignment),
            'duration_frames': len(alignment) - segment_start,
            'gop_maxlogit': float(gop_scores[segment_start]),
            'label_name': tokenizer.decode([int(current_label)])
        }
    
    return segments

def test_pronunciation(audio_path, expected_jyutping, expected_tone):
    #load audio
    audio_input, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(audio_input, return_tensors="pt", sampling_rate=16_000).input_features[0]
    output = model.inference(input_features=input_features.unsqueeze(0).to(DEVICE), processor=processor, tone_tokenizer=tone_tokenizer)

    #tokenize labels and get model outputs
    jyutping_logits = output[1]
    tone_logits = output[2]
    jyutping_labels = processor.tokenizer.encode(expected_jyutping)
    tone_labels = tone_tokenizer.encode(expected_tone)
    actual_output = output[3]

    #alignment and GOP calculation
    jyutping_alignment, dp, backpointer = ctc_viterbi_alignment(torch.tensor(jyutping_logits[0]), jyutping_labels)
    tone_alignment, tone_dp, tone_backpointer = ctc_viterbi_alignment(torch.tensor(tone_logits[0]), tone_tokenizer.encode(tone_labels))
    gop_maxlogit_jyutping = calculate_gop_maxlogit(torch.tensor(jyutping_logits[0]), jyutping_alignment)
    gop_maxlogit_tone = calculate_gop_maxlogit(torch.tensor(tone_logits[0]), tone_alignment)

    #segment phonemes and tones
    jyutping_segment = segment_by_labels(jyutping_alignment, gop_maxlogit_jyutping, jyutping_labels, processor.tokenizer)
    tone_segment = segment_by_labels(tone_alignment, gop_maxlogit_tone, tone_labels, tone_tokenizer)
    jyutping_prediction = getPredictions(torch.tensor(jyutping_logits[0]), jyutping_alignment,jyutping_tokenizer,2)
    tone_prediction = getPredictions(torch.tensor(tone_logits[0]), tone_alignment,tone_tokenizer,1,blank_id=0)
    
    #prepare output json
    result_json = {
        "total_frames": len(jyutping_segment),
        "model_output": actual_output,
        "phonemes": [],
        "tones": [],
        "overall_score": np.mean([s['gop_maxlogit'] for s in jyutping_segment.values()])
    }

    i = 0
    for label_idx, seg in sorted(jyutping_segment.items(), key=lambda x: x[1]['start_frame']):
        score = seg['gop_maxlogit']
        phoneme_data = {
            "id": int(label_idx),
            "jyutping": seg['label_name'],
            "start_frame": seg['start_frame'],
            "end_frame": seg['end_frame'],
            "duration_ms": seg['duration_frames'] * 10,
            "gop_maxlogit": float(score),
            "confidence": "perfect" if score > PHONEME_THRESHOLD_GOOD else "warning" if score > PHONEME_THRESHOLD_OK else "error",
            "prediction": jyutping_prediction.get(seg['label_name'], ("unknown", 0.0)),
        }
        i += 1
        result_json["phonemes"].append(phoneme_data)
    # detailed_gop_diagnosis(torch.tensor(jyutping_logits[0]), jyutping_segment, jyutping_labels, expected_jyutping)
    for label_idx, seg in sorted(tone_segment.items(), key=lambda x: x[1]['start_frame']):
        tone_score = seg['gop_maxlogit']
        tone_data = {
            "id": int(label_idx),
            "tone": seg['label_name'],
            "start_frame": seg['start_frame'],
            "end_frame": seg['end_frame'],
            "duration_ms": seg['duration_frames'] * 10, 
            "gop_maxlogit": float(tone_score),
            "confidence": "perfect" if tone_score > TONE_THRESHOLD_GOOD else "error",
            "prediction": tone_prediction.get(seg['label_name'], ("unknown", 0.0)),
        }
        result_json["tones"].append(tone_data)
    return result_json

def detailed_gop_diagnosis(logits, alignment, labels, jyutping_text):
    logits_np = logits.cpu().numpy()
    
    vocab_reverse = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    result = f"Jyutping: {jyutping_text}\\nLabels: {labels}\n\n"
    # For each label segment, show what the model is predicting
    for label_idx in np.unique(alignment):
        frames_for_label = np.where(alignment == label_idx)[0]
        start_frame = frames_for_label[0]
        end_frame = frames_for_label[-1]
        
        label_name = vocab_reverse.get(label_idx, f"unknown_{label_idx}")
        result += f"\nLabel {label_idx} ('{label_name}'): Frames {start_frame}-{end_frame}\n"
        
        for i, frame_idx in enumerate(frames_for_label[:3]):
            frame_logits = logits_np[frame_idx]
            top_3_idx = np.argsort(frame_logits)[-3:][::-1]
            
            result += f"\n  Frame {frame_idx}:\n    Reference ('{label_name}'): logit = {frame_logits[label_idx]:.2f}\n    Top-3 predictions:\n"
            for rank, pred_idx in enumerate(top_3_idx, 1):
                pred_name = vocab_reverse.get(pred_idx, f"unknown_{pred_idx}")
                pred_logit = frame_logits[pred_idx]
                is_correct = "✓" if pred_idx == label_idx else "✗"
                result += f"      {rank}. {is_correct} '{pred_name}' (ID {pred_idx}): {pred_logit:.2f}\n"
    
    result += "\n" + "="*70
    with open("logger.txt", "a", encoding="utf-8") as f:
        f.write(result)

def getPredictions(logits, alignment,tokenizor,number_of_predictions, blank_id=2):
    logits_np = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
    vocab_reverse = {v: k for k, v in tokenizor.get_vocab().items()}
    prediction = {}
    
    for label_idx in sorted(np.unique(alignment)):
        if label_idx == blank_id:
            continue
        frames = np.where(alignment == label_idx)[0]
        if len(frames) == 0:
            continue
        allPredictions = {}
        label_name = vocab_reverse.get(label_idx, f"[{label_idx}]")

        for t, frame_idx in enumerate(frames, 1):
            frame_logits = logits_np[frame_idx]
            top3_idx = np.argsort(frame_logits)[-3:][::-1]
            for idx in top3_idx:
                idx_name = vocab_reverse.get(idx, f"unk{idx}")
                if idx != 2 and idx != label_idx:
                # if idx != 2: 
                    if idx_name not in allPredictions:
                        allPredictions[idx_name] = float(frame_logits[idx])
                    else:
                        allPredictions[idx_name] = max(allPredictions[idx_name], float(frame_logits[idx]))
        sortedPredictions = sorted(allPredictions.items(), key=lambda x: x[1], reverse=True)
        prediction[label_name] = sortedPredictions[:number_of_predictions]
    return prediction

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python local_asr.py <audio_path> <expected_text>"}))
        sys.exit(1)
    audio_path = sys.argv[1]
    expected_text = sys.argv[2]
    with open('cantonese_jyutping.json', 'r', encoding='utf-8') as f:
        cantonese_dict = json.load(f)

    #separate to jyutping and tone
    jyutping = []
    tone = []
    for char in expected_text:
        word = cantonese_dict.get(char, ['unknown'])[0]
        if word == 'unknown':
            print(json.dumps({"error": f"Character '{char}' not found in Cantonese dictionary"}))
        jyutping.append(word[:-1])
        tone.append(word[-1])
    expected_jyutping = " ".join(jyutping)
    expected_tone = "".join(tone)

    result = test_pronunciation(audio_path, expected_jyutping, expected_tone)
    print(json.dumps(result, ensure_ascii=False))