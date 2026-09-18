import time
import threading

t0 = time.perf_counter()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tts.speech import speak, stop_tts, wait_until_done

_loaded = {}

def load_mistral():
    from llm.mistral import generate_response
    _loaded["generate_response"] = generate_response

def load_transcription():
    from voice.transcription import transcribe_audio
    _loaded["transcribe_audio"] = transcribe_audio

def load_speech():
    from tts.speech import speak, stop_tts
    _loaded["speak"] = speak
    _loaded["stop_tts"] = stop_tts

def load_s2s():
    from stream2sentence import generate_sentences
    _loaded["generate_sentences"] = generate_sentences

threads = [
    threading.Thread(target=load_mistral),
    threading.Thread(target=load_transcription),
    threading.Thread(target=load_speech),
    threading.Thread(target=load_s2s),
]

for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"[startup] parallel load TOTAL: {time.perf_counter() - t0:.2f}s")

generate_response = _loaded["generate_response"]
transcribe_audio = _loaded["transcribe_audio"]
speak = _loaded["speak"]
stop_tts = _loaded["stop_tts"]
generate_sentences = _loaded["generate_sentences"]

try:
    while True:
      text = transcribe_audio()
      print(f"User: {text}")

      stream = generate_response(text)
      buffered_stream = generate_sentences(stream, quick_yield_single_sentence_fragment=True)

      for text_chunk in buffered_stream:
          print(f"O-Hio: {text_chunk}", end=" ", flush=True)
          speak(text_chunk)

      wait_until_done()
      print()
except KeyboardInterrupt:
    print("\nShutting Down...")

finally:
    stop_tts()