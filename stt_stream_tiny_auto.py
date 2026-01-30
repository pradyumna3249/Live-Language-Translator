import os
import sounddevice as sd
import numpy as np
import queue
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
import sys

# ----------------------------------------------------
# MODEL PATH (tiny model stored here)
# ----------------------------------------------------
MODEL_PATH = "/home/neeraj/Desktop/TalkBridge/models/whisper-tiny"
REPO_ID = "Systran/faster-whisper-tiny"

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.5   # faster response
BLOCKSIZE = int(SAMPLE_RATE * CHUNK_DURATION)


# ----------------------------------------------------
# Auto-download model if missing
# ----------------------------------------------------
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("📥 Whisper-tiny not found. Downloading model (one-time)...")
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=MODEL_PATH,
            local_dir_use_symlinks=False
        )
        print("✅ Download complete!")
    else:
        print("✔ Whisper-tiny model already exists. Skipping download.")


class WhisperTinySTT:
    def __init__(self):
        ensure_model_exists()

        print("\n🔄 Loading Whisper-tiny model...")
        try:
            self.model = WhisperModel(
                MODEL_PATH,
                device="cpu",
                compute_type="float32"
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print("\n❌ ERROR loading model from:", MODEL_PATH)
            print(str(e))
            sys.exit(1)

        self.q = queue.Queue()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("⚠️ Audio Warning:", status)
        self.q.put(indata.copy())

    def start_stream(self):
        print("\n🎙 Starting microphone... (Ctrl+C to stop)\n")
        self.stream = sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            callback=self.audio_callback,
            blocksize=BLOCKSIZE,
            dtype=np.float32
        )
        self.stream.start()

    def run(self):
        try:
            while True:
                chunk = self.q.get()
                audio = chunk[:, 0].astype(np.float32)

                # Skip silence
                if np.abs(audio).mean() < 1e-5:
                    continue

                segments, _ = self.model.transcribe(
                    audio,
                    beam_size=1
                )

                text = " ".join(s.text.strip() for s in segments).strip()

                if text:
                    print("📝", text)

        except KeyboardInterrupt:
            print("\n🛑 Stopped.")
            self.stream.stop()
            self.stream.close()


if __name__ == "__main__":
    stt = WhisperTinySTT()
    stt.start_stream()
    stt.run()

