import sounddevice as sd
import wave
from pathlib import Path
from datetime import datetime

target_folder = Path(r"D:\DevStuff\O-hio\python\audio")

SAMPLE = 16000
DURATION = 5
CHANNELS = 1
OUTPUT_FILE = f"mic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

file_path = target_folder / OUTPUT_FILE

target_folder.mkdir(parents=True, exist_ok=True)

print("SPEAK NOW...")

audio = sd.rec(
  int(SAMPLE * DURATION),
  samplerate= SAMPLE,
  channels=CHANNELS,
  dtype="int16"
)

sd.wait()

print("FINISHED RECORDING")

with wave.open(file_path, "wb") as file:
  file.setnchannels(CHANNELS)
  file.setsampwidth(2)
  file.setframerate(SAMPLE)
  file.writeframes(audio.tobytes())

print(f"Saved to {file_path}")