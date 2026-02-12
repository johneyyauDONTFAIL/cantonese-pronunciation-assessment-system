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

tokenizer = Wav2Vec2CTCTokenizer(
    "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
)
tone_tokenizer = Wav2Vec2CTCTokenizer(
    "tone_vocab.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|",
)

# load processor
feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_id)
processor = Wav2Vec2BertProcessor(
    feature_extractor=feature_extractor, tokenizer=tokenizer
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
).eval().cuda()

def ctc_viterbi_alignment(logits, labels):
    num_frames, num_classes = logits.shape
    logits_np = logits.cpu().numpy()
    
    # Initialize DP table
    dp = np.full((num_frames, len(labels)), -np.inf)
    backpointer = np.zeros((num_frames, len(labels)), dtype=int)
    
    # Start: match first label at first frame
    dp[0, 0] = logits_np[0, labels[0]]
    
    # Fill DP table
    for t in range(1, num_frames):
        for j in range(len(labels)):
            label_idx = labels[j]
            # Can come from same state or previous state
            candidates = []
            if j > 0:
                candidates.append(dp[t-1, j-1])
            candidates.append(dp[t-1, j])
            
            best_idx = np.argmax(candidates)
            prev_j = j - 1 if best_idx == 0 and j > 0 else j
            
            dp[t, j] = logits_np[t, label_idx] + dp[t-1, prev_j]
            backpointer[t, j] = prev_j
    
    # Backtrack to get alignment
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

def segment_by_labels(alignment, gop_scores, labels, blank_id=2):
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
                    'label_name': processor.tokenizer.decode([int(current_label)])
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
                    'label_name': processor.tokenizer.decode([int(current_label)])
                }
            current_label = label_idx
            segment_start = t
    
    # Final segment
    if current_label is not None:
        segments[int(current_label)] = {
            'start_frame': segment_start,
            'end_frame': len(alignment),
            'duration_frames': len(alignment) - segment_start,
            'gop_maxlogit': float(gop_scores[segment_start]),
            'label_name': processor.tokenizer.decode([int(current_label)])
        }
    
    return segments

def test_pronunciation(audio_path, expected_jyutping):
    audio_input, _ = librosa.load(audio_path, sr=16000)
    input_features = processor(audio_input, return_tensors="pt", sampling_rate=16_000).input_features[0]
    output = model.inference(input_features=input_features.unsqueeze(0).cuda(), processor=processor, tone_tokenizer=tone_tokenizer)
    # print(f"ASR Output: {output[0]}")
    jyutping_logits = output[1]
    jyutping_labels = processor.tokenizer.encode(expected_jyutping)
    actual_output = output[3]
    # print(f"\nJyutping labels: {jyutping_labels}")

    alignment, dp, backpointer = ctc_viterbi_alignment(torch.tensor(jyutping_logits[0]), jyutping_labels)
    gop_maxlogit = calculate_gop_maxlogit(torch.tensor(jyutping_logits[0]), alignment)
    segment = segment_by_labels(alignment, gop_maxlogit, jyutping_labels)
    all_prediction = getTop3Prediction(torch.tensor(jyutping_logits[0]), alignment)
    test_prediction = getSecondPrediction(torch.tensor(jyutping_logits[0]), alignment)
    
    result_json = {
        "total_frames": len(alignment),
        "phonemes": [],
        "diagnosis": {
            "perfect": [],
            "warning": [],
            "errors": []
        },
        "overall_score": np.mean([s['gop_maxlogit'] for s in segment.values()]),
        "alignment_quality": "good"  # Add CTC loss check later
    }

    # Thresholds for Cantonese (tune these!)
    PERFECT_THRESH, WARNING_THRESH = 20.0, 15.0
    i = 0
    for label_idx, seg in sorted(segment.items(), key=lambda x: x[1]['start_frame']):
        score = seg['gop_maxlogit']
        phoneme_data = {
            "id": int(label_idx),
            "jyutping": seg['label_name'],
            "start_frame": seg['start_frame'],
            "end_frame": seg['end_frame'],
            "duration_ms": seg['duration_frames'] * 10,  # 10ms/frame
            "gop_maxlogit": float(score),
            "confidence": "perfect" if score > PERFECT_THRESH else "warning" if score > WARNING_THRESH else "error",
            "status_emoji": "good" if score > PERFECT_THRESH else "ok" if score > WARNING_THRESH else "bad",
            "actual_output": actual_output[i],
            "test_prediction": test_prediction.get(seg['label_name'], ("unknown", 0.0)),
            # "predictions_per_frame": all_prediction.get(seg['label_name'], {})
        }
        i += 1
        result_json["phonemes"].append(phoneme_data)
        
        if score > PERFECT_THRESH:
            result_json["diagnosis"]["perfect"].append(phoneme_data["jyutping"])
        elif score > WARNING_THRESH:
            result_json["diagnosis"]["warning"].append(phoneme_data["jyutping"])
        else:
            result_json["diagnosis"]["errors"].append(phoneme_data["jyutping"])
    detailed_gop_diagnosis(torch.tensor(jyutping_logits[0]), alignment, jyutping_labels, expected_jyutping)
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
        
        # For first 3 frames in this segment, show top-3 predictions
        for i, frame_idx in enumerate(frames_for_label[:3]):
            frame_logits = logits_np[frame_idx]
            
            # Get top-3 predictions
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

def getTop3Prediction(logits, alignment):
    logits_np = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
    vocab_reverse = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    prediction = {}
    
    for label_idx in sorted(np.unique(alignment)):
        if label_idx == 2:
            continue
        frames = np.where(alignment == label_idx)[0]
        if len(frames) == 0:
            continue
            
        label_name = vocab_reverse.get(label_idx, f"[{label_idx}]")
        prediction[label_name] = {}

        for t, frame_idx in enumerate(frames, 1):
            frame_logits = logits_np[frame_idx]
            top3_idx = np.argsort(frame_logits)[-3:][::-1]
            prediction[label_name][int(frame_idx)] = [(vocab_reverse.get(idx,f"unk{idx}"), float(frame_logits[idx])) for idx in top3_idx if idx != 2]
    return prediction

def getSecondPrediction(logits, alignment):
    logits_np = logits.detach().cpu().numpy() if torch.is_tensor(logits) else np.asarray(logits)
    vocab_reverse = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    prediction = {}
    
    for label_idx in sorted(np.unique(alignment)):
        if label_idx == 2:
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
                if idx != 2 and idx != label_idx:
                    if idx not in allPredictions:
                        allPredictions[vocab_reverse.get(idx,f"unk{idx}")] = float(frame_logits[idx])
                    else:
                        allPredictions[vocab_reverse.get(idx,f"unk{idx}")] = max(allPredictions[idx], float(frame_logits[idx]))
        sortedPredictions = sorted(allPredictions.items(), key=lambda x: x[1], reverse=True)
        print([sortedPredictions[0], sortedPredictions[1]])
        prediction[label_name] = [sortedPredictions[0], sortedPredictions[1]]
    return prediction




if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python local_asr.py <audio_path> <expected_text>"}))
        sys.exit(1)
    # print("Starting evaluation...",sys.argv)
    audio_path = sys.argv[1]
    expected_text = sys.argv[2]
    with open('cantonese_jyutping.json', 'r', encoding='utf-8') as f:
        cantonese_dict = json.load(f)
    expected_jyutping = " ".join([cantonese_dict.get(char, ['unknown'])[0][:-1] for char in expected_text])
    # jyutping_labels = processor.tokenizer.encode(expected_jyutping)
    result = test_pronunciation(audio_path, expected_jyutping)
    print(json.dumps(result, ensure_ascii=False))