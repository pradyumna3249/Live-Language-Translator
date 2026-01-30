from transformers import AutoProcessor, WhisperForConditionalGeneration

processor = AutoProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

