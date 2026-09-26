import time
import threading
from voice.wakeword import wait_for_wake
from enum import Enum, auto

t0 = time.perf_counter()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

_loaded = {}

def timed(name, fn):
    start = time.perf_counter()
    fn()
    print(f"[startup] {name}: {time.perf_counter() - start:.2f}s")

def load_mistral():
    from llm.mistral import generate_response
    _loaded["generate_response"] = generate_response

def load_transcription():
    from voice.transcription import transcribe_audio
    _loaded["transcribe_audio"] = transcribe_audio

def load_speech():
    from tts.speech import speak, stop_tts, wait_until_done
    _loaded["speak"] = speak
    _loaded["stop_tts"] = stop_tts
    _loaded["wait_until_done"] = wait_until_done

def load_s2s():
    from stream2sentence import generate_sentences
    _loaded["generate_sentences"] = generate_sentences

threads = [
    threading.Thread(target=timed, args=("mistral", load_mistral)),
    threading.Thread(target=timed, args=("transcription", load_transcription)),
    threading.Thread(target=timed, args=("speech", load_speech)),
    threading.Thread(target=timed, args=("stream2sentence", load_s2s)),
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
wait_until_done = _loaded["wait_until_done"]
generate_sentences = _loaded["generate_sentences"]

FOLLOWUP_TIMEOUT = 20

def handle_turn(timeout=None):
    text = transcribe_audio(timeout=timeout)
    if text is None:
        return False
    
    print(f"User: {text}")

    stream = generate_response(text)
    buffered_stream = generate_sentences(stream, quick_yield_single_sentence_fragment=True)

    for text_chunk in buffered_stream:
      print(f"A.L.F.I.E: {text_chunk}", end=" ", flush=True)
      speak(text_chunk)

    wait_until_done()
    print()
    return True

class State(Enum):
    SLEEPING = auto()
    LISTENING = auto()

state = State.SLEEPING 

try:
    while True:
        if state == State.SLEEPING:
          wait_for_wake()
          state = State.LISTENING
        elif state == State.LISTENING:
            heard = handle_turn(timeout=FOLLOWUP_TIMEOUT)
            state = State.LISTENING if heard else State.SLEEPING

except KeyboardInterrupt:
    print("\nShutting Down...")

finally:
    stop_tts()