import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from queue import Queue
import threading

from llm.mistral import generate_response
from voice.transcription import transcribe_audio
from tts.speech import speak
from buffer.buffer import sentence_buffer

text = transcribe_audio()

print(f"User: {text}")

stream = generate_response(text)
buffered_stream = sentence_buffer(stream)

tts_queue = Queue()

def tts_worker():
  while True:
    text_chunk = tts_queue.get()

    if text_chunk is None:
      break

    speak(text_chunk)

    tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker)
tts_thread.start()

for text_chunk in buffered_stream:
  print(f"O-Hio: {text_chunk}", end="", flush=True)
  print(f"\nQueue before put: {tts_queue.qsize()}")
  tts_queue.put(text_chunk)

tts_queue.put(None)
tts_thread.join()

print()