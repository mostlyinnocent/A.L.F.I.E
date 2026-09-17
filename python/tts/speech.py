import time
from queue import Queue
from threading import Thread
import numpy as np

from kokoro import KPipeline
import sounddevice as sd


pipeline = KPipeline(
    lang_code="a",
    repo_id="hexgrad/Kokoro-82M",
    device="cuda"
)

text_queue = Queue()
audio_queue = Queue()

def tts_generator_worker():

    while True:
        text = text_queue.get()

        if text is None:
            text_queue.task_done()
            audio_queue.put(None)
            break

        generation_start = time.perf_counter()

        generator = pipeline(
            text,
            voice="af_heart",
        )

        for _, _, audio in generator:
            audio_queue.put(audio)

        generation_end = time.perf_counter()

        print(
            f"kokoro generation: "
            f"{generation_end - generation_start:.2f}s"
        )

        text_queue.task_done()

def audio_playback_worker():
    stream = sd.OutputStream(samplerate=24000, channels=1, dtype='float32')
    stream.start()
    try:
      while True:
          audio = audio_queue.get()

          if audio is None:
              audio_queue.task_done()
              break

          audio_np = audio.detach().cpu().numpy() if hasattr(audio, "detach") else audio
          stream.write(audio_np)

          audio_queue.task_done()
    finally:
        stream.stop()
        stream.close()

tts_thread = Thread(target=tts_generator_worker)
playback_thread = Thread(target=audio_playback_worker)

tts_thread.start()
playback_thread.start()

def speak(text):
    text_queue.put(text)

def stop_tts():
    text_queue.put(None)

    text_queue.join()
    audio_queue.join()

    tts_thread.join()
    playback_thread.join()
