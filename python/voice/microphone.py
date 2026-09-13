import sounddevice as sd
import soundfile as sf
from pathlib import Path
from datetime import datetime
from silero_vad import load_silero_vad, VADIterator
import time


target_folder = Path(r"D:\DevStuff\O-hio\python\audio")

SAMPLES = 16000
CHANNELS = 1
CHUNK_SIZE = 512
MAX_DURATION = 30
has_spoken = False

model = load_silero_vad()

vad = VADIterator(
  model,
  sampling_rate=SAMPLES,
  min_silence_duration_ms=5000,
  speech_pad_ms=100,
)


def record_audio():
  OUTPUT_FILE = f"mic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

  file_path = target_folder / OUTPUT_FILE

  target_folder.mkdir(parents=True, exist_ok=True)

  print("Speak Now...")

  start_time = time.monotonic()

  try:
    with sd.InputStream(samplerate=SAMPLES, channels=CHANNELS, dtype="float32") as stream:
      with sf.SoundFile(file_path, mode="x", samplerate=SAMPLES, channels=CHANNELS) as file:
        while True:
          audio, overflow = stream.read(CHUNK_SIZE)
          if overflow:
            print("Warning: Input overflow occured (audio dropped)")

          audio_mono = audio[:, 0]
          speech_event = vad(audio_mono)
          file.write(audio)

          if speech_event:

            if "start" in speech_event:
              has_spoken = True
              print("speech started")

            elif "end" in speech_event and has_spoken:
              print("Speech ended.")
              break
          if time.monotonic() - start_time >= MAX_DURATION:
              print("Maximum recording duration reached.")
              break

  except KeyboardInterrupt:
    print("\nRecording interrupted")

  finally:
    vad.reset_states()

  print("Svaed to:", file_path)

  return file_path