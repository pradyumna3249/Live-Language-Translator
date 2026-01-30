TalkBridge
Real-Time Speech-to-Text & Multilingual Translation System

TalkBridge is a real-time, low-latency speech intelligence system that captures live microphone audio, converts speech into text using Whisper / Faster-Whisper, translates English speech into Hindi and Telugu, and optionally generates Text-to-Speech (TTS) output.

The project demonstrates end-to-end speech pipeline engineering, combining streaming audio processing, deep learning–based speech recognition, multilingual NLP, dataset engineering, and tokenizer training.

Why TalkBridge?

✔ Real-time streaming (not batch processing)
✔ Multilingual support (EN → HI, EN → TE)
✔ Offline + online inference modes
✔ Production-style architecture
✔ Dataset & tokenizer pipeline included

This makes TalkBridge suitable for research, assistive technologies, voice interfaces, and edge/real-time AI systems.

Core Capabilities
Speech Recognition

Real-time microphone streaming

Chunk-wise transcription for low latency

Whisper / Faster-Whisper based STT

Supports tiny, small, and medium models

Multilingual Translation

English speech → Hindi text
English speech → Telugu text
Translation logging for dataset generation

Text-to-Speech (TTS)

Optional Hindi & Telugu speech output
Supports gTTS / Piper-based TTS pipelines
Dataset & NLP Engineering
Cleaning and normalization of multilingual corpora
EN–HI and EN–TE dataset merging
Train / Validation / Test splitting
SentencePiece BPE tokenizer training

System Architecture
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

Repository Structure
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

Tech Stack
Category	Tools / Libraries
Language	Python
Speech Recognition	Whisper, Faster-Whisper
Audio Streaming	sounddevice
NLP	SentencePiece (BPE)
Translation	rule-based / pipeline-based
TTS	gTTS, Piper
ML Utilities	NumPy
Model Hosting	HuggingFace Hub


Installation
1️ Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

2 Install dependencies
pip install numpy sounddevice faster-whisper transformers sentencepiece gtts huggingface_hub tqdm
⚠ Ensure microphone access is enabled on your system.

Running the System
🔹 Fastest Real-Time STT (Tiny Model)
python stt_stream_tiny_auto.py

🔹 Higher Accuracy STT (Small / Medium)
python stt_stream_small_auto.py
python fast_stream_stt.py

🔹 Offline Whisper (Local Model)
python stt_stream_local.py

🔹 Speech → Translation → Hindi TTS
python speech_translate1.py


All translations are logged in:
speech_translations.txt

Dataset & Tokenizer Pipeline
Dataset Preparation
python merge_all.py


✔ Cleans text
✔ Removes noise & duplicates
✔ Merges EN–HI and EN–TE corpora
✔ Generates train/valid/test splits

Tokenizer Training
python train_tokenizer.py --vocab_size 32000 --model_type bpe


Special language tokens
>>en<<   >>hi<<   >>te<<

Sample Output
English : hello world
Hindi   : हैलो दुनिया आप कैसे हैं
Telugu  : హలో వరల్డ్

Applications

Real-time speech translation systems
Assistive technologies for accessibility
Multilingual voice assistants
Speech-enabled NLP research
Edge AI & embedded speech systems


📄 License

This project is intended for academic, learning, and research purposes.