import sounddevice as sd
import numpy as np
import queue
from faster_whisper import WhisperModel
import time
import sys

# ---- IMPORTANT: Your local whisper-medium model path ----
MODEL_PATH = "/home/neeraj/Desktop/TalkBridge/models/whisper-medium"

SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0  # seconds
BLOCKSIZE = int(SAMPLE_RATE * CHUNK_DURATION)


class LocalWhisperSTT:
    def __init__(self):

        print("\n🔄 Loading Whisper model from local path...")
        try:
            self.model = WhisperModel(
                MODEL_PATH,
                device="cpu",       # CPU mode
                compute_type="float32"
            )
            print("✅ Model loaded successfully (CPU mode, offline)")
        except Exception as e:
            print("\n❌ ERROR: Could not load Whisper model from:")
            print(MODEL_PATH)
            print("\nExpected files: model.bin, model.json, tokenizer.json, vocabulary.txt")
            print("\nError:", e)
            sys.exit(1)

        self.q = queue.Queue()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio Warning:", status)
        self.q.put(indata.copy())

    def start_stream(self):
        print("🎙 Starting microphone stream... (Ctrl+C to stop)\n")

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
                # get audio chunk
                audio_chunk = self.q.get()[:, 0].astype(np.float32)

                # skip silence
                if np.abs(audio_chunk).mean() < 1e-5:
                    continue

                # transcribe chunk
                segments, info = self.model.transcribe(
                    audio_chunk,
                    beam_size=1
                )

                text = " ".join([s.text.strip() for s in segments]).strip()

                if text:
                    print("📝", text)

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user.")
            self.stream.stop()
            self.stream.close()


if __name__ == "__main__":
    stt = LocalWhisperSTT()
    stt.start_stream()
    stt.run()

