import sounddevice as sd
import numpy as np
import queue
from faster_whisper import WhisperModel
import threading
import time

class ContinuousSTT:
    def __init__(self, model_size="medium", chunk_duration=1, sample_rate=16000):
        self.sample_rate = sample_rate
        self.chunk = chunk_duration
        self.q = queue.Queue()

        print("Loading Faster-Whisper model...")
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        print("Model loaded!")

    def audio_callback(self, indata, frames, time, status):
        """Push microphone audio chunks into queue"""
        if status:
            print(status)
        self.q.put(indata.copy())

    def start_stream(self):
        """Start continuous microphone streaming"""
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self.audio_callback,
            blocksize=int(self.sample_rate * self.chunk),
            dtype=np.float32
        )
        self.stream.start()
        print("🎙 Listening...")

    def run(self, callback=None):
        """
        Continuously transcribe incoming audio.
        callback(text) → sends each transcription to your translator.
        """
        audio_buffer = np.zeros(0, dtype=np.float32)

        while True:
            chunk = self.q.get()
            audio_buffer = np.concatenate((audio_buffer, chunk[:, 0]))

            if len(audio_buffer) > self.sample_rate * self.chunk:
                segments, _ = self.model.transcribe(audio_buffer, beam_size=1)

                text = " ".join([s.text for s in segments]).strip()

                if text:
                    if callback:
                        callback(text)
                    else:
                        print("📝 ", text)

                audio_buffer = np.zeros(0, dtype=np.float32)


if __name__ == "__main__":
    stt = ContinuousSTT("medium")

    stt.start_stream()

    def show(text):
        print(">>", text)

    stt.run(callback=show)

