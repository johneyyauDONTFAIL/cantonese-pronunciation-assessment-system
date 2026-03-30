# CityU Metaverse — Cantonese Pronunciation Assessment System
### FYP 2025-2026 | Department of Computer Science, City University of Hong Kong

A phoneme-level Cantonese mispronunciation detection and diagnostic feedback system built for the CityU Metaverse Cantonese Learning Room. The system replaces binary correct/incorrect ASR feedback with phone-level Goodness of Pronunciation (GOP) scoring, CTC Viterbi forced alignment, and second-prediction error diagnosis.

---

## Features

- **Phoneme-level mispronunciation detection** using GOP MaxLogit scoring
- **CTC Viterbi forced alignment** for frame-level phoneme boundary segmentation
- **Second Prediction logic** — identifies what the learner likely said instead of the target phoneme
- **Tone error detection** with separate GOP scoring for Cantonese tones (1–6)
- **Jyutping output** — feedback in romanised phonetics, accessible to non-Chinese readers
- **REST API server** (Flask) for integration with CityU Metaverse Node.js frontend
- **JSON-structured response** including per-phoneme confidence, GOP score, frame segmentation, and second prediction

---

## System Architecture
CityU Metaverse
│ audio + target text (multipart/form-data)  
▼  
Flask REST API (server.py)  
│  
▼  
Wav2Vec2-BERT ASR Model  
hon9kon9ize/wav2vec2bert-jyutping  
│  
├── CTC Viterbi Alignment  
├── GOP MaxLogit Scoring (phoneme + tone)  
├── Second Prediction Retrieval  
└── JSON Diagnostic Response  
│  
▼  
CityU Metaverse Frontend  
(feedback display)  

---

## GOP Score Thresholds

### | Score Range | Status | Meaning |

| ≥ 20 | ✅ Perfect | High confidence correct pronunciation |

| 15 – 20 | 🟡 Acceptable | Correct but borderline quality |

| < 15 | ❌ Mispronounced | Likely incorrect, diagnostic feedback triggered |

## Tone GOP uses separate thresholds:

### | Score Range | Status |

| ≥ 10 | ✅ Perfect |

| 5 – 10 | 🟡 Acceptable |

| < 5 | ❌ Tone Error |

---

## Example API Response

**Request:**
```http
POST /evaluate
Content-Type: multipart/form-data

audio: <audio_file.wav>
expected_text: 城大
```

**Response:**
```json
{
  "model_output": [
    "s",
    "ing",
    "d",
    "aai"
  ],
  "overall_score": 21.415034294128418,
  "phonemes": [
    {
      "confidence": "perfect",
      "duration_ms": 290,
      "end_frame": 29,
      "gop_maxlogit": 21.0117244720459,
      "id": 62,
      "jyutping": "s",
      "prediction": [
        [
          "c",
          8.125834465026855
        ],
        [
          "e",
          7.720173358917236
        ]
      ],
      "start_frame": 0
    },
    {
      "confidence": "perfect",
      "duration_ms": 90,
      "end_frame": 38,
      "gop_maxlogit": 22.577178955078125,
      "id": 40,
      "jyutping": "ing",
      "prediction": [
        [
          "eng",
          9.046229362487793
        ],
        [
          "an",
          8.323688507080078
        ]
      ],
      "start_frame": 29
    },
    {
      "confidence": "warning",
      "duration_ms": 50,
      "end_frame": 43,
      "gop_maxlogit": 19.389541625976562,
      "id": 22,
      "jyutping": "d",
      "prediction": [
        [
          "t",
          7.2018656730651855
        ],
        [
          "b",
          6.745830059051514
        ]
      ],
      "start_frame": 38
    },
    {
      "confidence": "perfect",
      "duration_ms": 150,
      "end_frame": 58,
      "gop_maxlogit": 22.681692123413086,
      "id": 4,
      "jyutping": "aai",
      "prediction": [
        [
          "aan",
          8.226093292236328
        ],
        [
          "h",
          8.08055591583252
        ]
      ],
      "start_frame": 43
    }
  ],
  "tones": [
    {
      "confidence": "perfect",
      "duration_ms": 400,
      "end_frame": 40,
      "gop_maxlogit": 16.105642318725586,
      "id": 6,
      "prediction": [
        [
          "6",
          1.8352006673812866
        ]
      ],
      "start_frame": 0,
      "tone": "4"
    },
    {
      "confidence": "perfect",
      "duration_ms": 180,
      "end_frame": 58,
      "gop_maxlogit": 11.323063850402832,
      "id": 8,
      "prediction": [
        [
          "2",
          5.863823890686035
        ]
      ],
      "start_frame": 40,
      "tone": "6"
    }
  ],
  "total_frames": 58
}
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/johneyyauDONTFAIL/cantonese-pronunciation-assessment-system.git
cd cantonese-pronunciation-assessment-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA support according to your CUDA version first:  
#### Example  
> ```bash
> # CUDA 11.8
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
> # CUDA 12.1
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

---
## Testing Inference (Google Colab)

If you do not have a local GPU, you can verify the ASR-GOP engine and the scoring logic using our Google Colab environment. This demo clones the repository, installs all dependencies, and executes a sample inference to generate the diagnostic JSON payload.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/johneyyauDONTFAIL/cantonese-pronunciation-assessment-system/blob/main/colab_demo.ipynb)

**Steps to Run:**
1. Click the badge above to open the notebook.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run all cells (`Ctrl + F9`).
4. The output will display the output JSON.
---

## Usage

### Start the Flask API server
```bash
python server.py
```
Server runs on `http://localhost:5000` by default.

### Run inference directly
```bash
python inference.py "audio_path" "expected_text"
```
#### Example
```bash
python inference.py ./audio/城大.wav "城大"
```

### Run Frontend webpage for better visualization of result
```bash
python -m http.server 8000
```
Then go to 127.0.0.1:8000. A webpage should be opened which allow you to record a wav audio and type in expected words. Then you can click evaluate to send the request to the backend and wait for the result.

## Project Structure
cantonese-pronunciation-assessment-system/  
├── model.py # Wav2Vec2BertForCantonese model class  
├── inference.py # test_pronunciation pipeline (alignment + GOP + feedback)  
├── server.py # Flask REST API server  
├── index.html # Frontend webpage for demo  
├── script.js # Frontend webpage for demo  
├── style.css # Frontend webpage for demo  
├── evaluate.py # CER/WER and latency benchmarking  
├── vocab.json # Jyutping phoneme tokenizer vocabulary  
├── tone_vocab.json # Tone tokenizer vocabulary  
├── cantonese_jyutping.json # Cantonese dictionary for translating chinese expected text to jyutping for inference.py  
├── README.me # This file  
├── colab_demo # A demo on colab for the main function in inference.py  
└── requirements.txt # pip package needed

---

## Performance

Benchmarked on 60 audio samples:

| Metric | GTX 1660 Ti | Google Colab (T4 GPU) |  
| Mean Latency | 21.6 s | 122.5 ms |  
| Min Latency | 15.1 s | 77.7 ms |  
| Max Latency | 25.1 s | 226.4 ms |  
| Mean RTF | — | 0.0396 |  

> RTF of **0.0396** means the system processes audio in ~4% of its actual duration on a T4 GPU, well within the 300 ms threshold for real-time interactive feedback.


---

## Model

This project uses **[hon9kon9ize/wav2vec2bert-jyutping](https://huggingface.co/hon9kon9ize/wav2vec2bert-jyutping)**, a Cantonese-specialised fine-tune of Wav2Vec2-BERT trained on Common Voice 17 Cantonese with Jyutping phonetic labels.

## Author

**YAU Chun Kit** | BSCCS Final Year Project 2025–2026
City University of Hong Kong — Department of Computer Science
Supervisor: Prof. LEUNG, Wing Ho Howard