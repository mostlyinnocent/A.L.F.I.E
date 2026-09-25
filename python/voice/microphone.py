import sounddevice as sd
import numpy as np
from silero_vad import load_silero_vad, VADIterator
import time

SAMPLES = 16000
CHANNELS = 1
CHUNK_SIZE = 512
MAX_DURATION = 30

model = load_silero_vad()

vad = VADIterator(
  model,
  sampling_rate=SAMPLES,
  min_silence_duration_ms=1200,
  speech_pad_ms=100,
)


def record_audio(timeout=None):
  has_spoken = False
  chunks = []

  start_time = time.monotonic()

  print("Speak Now...")

  start_time = time.monotonic()

  try:
    with sd.InputStream(samplerate=SAMPLES, channels=CHANNELS, dtype="float32") as stream:
        while True:
          audio, overflow = stream.read(CHUNK_SIZE)
          if overflow:
            print("Warning: Input overflow occured (audio dropped)")

          audio_mono = audio[:, 0]
          chunks.append(audio_mono.copy())

          speech_event = vad(audio_mono)

          if speech_event:

            if "start" in speech_event:
              has_spoken = True
              print("speech started")

            elif "end" in speech_event and has_spoken:
              print("Speech ended.")
              break

          elapsed = time.monotonic() - start_time

          if not has_spoken and timeout is not None and elapsed >= timeout:
            print("No speech detected, timing out.")
            return None

          if elapsed >= MAX_DURATION:
            print("Maximum recording duration reached.")
            break

  except (KeyboardInterrupt, TypeError) as e:
    if isinstance(e, TypeError) and "cannot be casted to tensor" not in str(e):
      raise
    print("\nRecording interrupted")
    raise KeyboardInterrupt

  finally:
    vad.reset_states()

  return np.concatenate(chunks) if chunks else np.array([], dtype="float32")