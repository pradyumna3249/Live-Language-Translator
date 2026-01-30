import time
import sys
import speech_recognition as sr
from translate import Translator
from gtts import gTTS
import os

OUTPUT_FILE = "speech_translations.txt"

def speak_hindi(text):
    try:
        tts = gTTS(text=text, lang="hi")
        filename = "hindi_output.mp3"
        tts.save(filename)
        print("[TTS] Playing Hindi audio...")
        os.system(f"mpg123 {filename} > /dev/null 2>&1")
    except Exception as e:
        print(f"[TTS Error]: {e}")

def recognize_from_mic(timeout=5, phrase_time_limit=10):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print("Speak now!")
        audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        raise RuntimeError("Speech unclear.")
    except sr.RequestError as e:
        raise RuntimeError(f"Speech service error: {e}")

def translate_text(text):
    hi_trans = Translator(from_lang="en", to_lang="hi")
    te_trans = Translator(from_lang="en", to_lang="te")

    try:
        hi = hi_trans.translate(text)
    except Exception as e:
        hi = f"[Error: {e}]"

    try:
        te = te_trans.translate(text)
    except Exception as e:
        te = f"[Error: {e}]"

    return hi, te

def append_to_file(original, hi, te):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"Time: {ts}\n")
        f.write(f"English: {original}\n")
        f.write(f"Hindi  : {hi}\n")
        f.write(f"Telugu : {te}\n")
        f.write("-" * 40 + "\n")

def main():
    print("==== Speech → Translation → Hindi TTS ====\nPress Ctrl+C to quit.\n")
    
    while True:
        try:
            english_text = recognize_from_mic()
            print("\nRecognized English:", english_text)
        except Exception as e:
            print("Recognition error:", e)
            if input("Try again? (Y/n): ").strip().lower() not in ("", "y"):
                break
            continue

        # Translate
        hi, te = translate_text(english_text)
        print("Hindi :", hi)
        print("Telugu:", te)

        # Play only Hindi TTS
        speak_hindi(hi)

        # Save to file
        append_to_file(english_text, hi, te)
        print("(Saved to speech_translations.txt)\n")

        if input("Translate again? (Y/n): ").strip().lower() not in ("", "y"):
            break

if __name__ == "__main__":
    main()

