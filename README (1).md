# TalkBridge  
## Real-Time Speech-to-Text & Multilingual Translation System

TalkBridge is a **real-time, low-latency speech intelligence system** that captures live microphone audio, converts speech into text using **Whisper / Faster-Whisper**, translates English speech into **Hindi and Telugu**, and optionally generates **Text-to-Speech (TTS)** output.

The project demonstrates **end-to-end speech pipeline engineering**, combining **streaming audio processing**, **deep learning–based speech recognition**, **multilingual NLP**, **dataset engineering**, and **tokenizer training**.


## Why TalkBridge?

- Real-time streaming (not batch processing)  
- Multilingual support (EN → HI, EN → TE)  
- Offline + online inference modes  
- Production-style architecture  
- Dataset & tokenizer pipeline included  


## Core Capabilities

### Speech Recognition
- Real-time microphone streaming
- Chunk-wise transcription for low latency
- Whisper / Faster-Whisper based STT
- Supports tiny, small, and medium models

### Multilingual Translation
- English speech → Hindi text
- English speech → Telugu text
- Translation logging for dataset generation

### Text-to-Speech (TTS)
- Optional Hindi & Telugu speech output
- Supports gTTS / Piper-based pipelines

### Dataset & NLP Engineering
- Cleaning and normalization of multilingual corpora
- EN–HI and EN–TE dataset merging
- Train / Validation / Test splitting
- SentencePiece BPE tokenizer training


## System Architecture

```
Microphone Audio
      ↓
Real-Time Audio Chunking
      ↓
Speech-to-Text (Whisper / Faster-Whisper)
      ↓
Multilingual Translation (EN → HI / TE)
      ↓
(Optional) Text-to-Speech
      ↓
Logs & Dataset Storage
```


## Repository Structure

```
.
├── stt_stream_tiny_auto.py        # Real-time STT using Whisper-tiny (auto-download)
├── stt_stream_small_auto.py       # Real-time STT using Whisper-small
├── stt_stream_local.py            # Offline STT using local Whisper model
├── fast_stream_stt.py             # Faster-Whisper continuous streaming
├── speech_translate1.py           # Speech → Translation → Hindi TTS pipeline
│
├── merge_all.py                   # Dataset cleaning, merging, splitting (EN–HI, EN–TE)
├── train_tokenizer.py             # SentencePiece tokenizer training (BPE)
├── Training.ipynb                 # Training & experimentation notebook
│
├── tokenizer/
│   ├── spiece.model               # Trained SentencePiece model
│   ├── spiece.vocab               # SentencePiece vocabulary
│   └── special_tokens.txt         # Language & special tokens
│
├── outputs/
│   ├── speech_translations.txt    # Logged speech and translations
│   ├── hindi_output.mp3           # Sample Hindi TTS output
│   └── telugu_output.mp3          # Sample Telugu TTS output
│
└── README.md
```

---

## Tech Stack

| Category | Tools / Libraries |
|-------|------------------|
| Language | Python |
| Speech Recognition | Whisper, Faster-Whisper |
| Audio Streaming | sounddevice |
| NLP | SentencePiece (BPE) |
| TTS | gTTS, Piper |
| ML Utilities | NumPy |
| Model Hosting | HuggingFace Hub |

---

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate         # Windows
pip install numpy sounddevice faster-whisper transformers sentencepiece gtts huggingface_hub tqdm
```

---

## Running the System

```bash
python stt_stream_tiny_auto.py
python stt_stream_small_auto.py
python fast_stream_stt.py
python stt_stream_local.py
python speech_translate1.py
```

All translations are logged in:
```
speech_translations.txt
```

---

## Dataset & Tokenizer Pipeline

```bash
python merge_all.py
python train_tokenizer.py --vocab_size 32000 --model_type bpe
```

Language tokens:
```
>>en<<   >>hi<<   >>te<<
```

---

## Sample Output

```
English : hello world
Hindi   : हैलो दुनिया आप कैसे हैं
Telugu  : హలో వరల్డ్
```

---

## Contributor

**Pradyumna Kumar**  
Speech processing, real-time streaming, translation pipeline, dataset & tokenizer engineering.

---

## License

Academic and research use.
