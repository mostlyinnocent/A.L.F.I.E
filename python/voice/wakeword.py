import numpy as np
import sounddevice as sd
from openwakeword.model import Model

SAMPLES = 16000
FRAME_SIZE = 12000
THRESHOLD = 0.3

model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

def wait_for_wake():
  with sd.InputStream(samplerate=SAMPLES, channels=1, dtype="int16") as stream:
    while True:
      audio, overflow = stream.read(FRAME_SIZE)
      if overflow:
        print("Warning: Input overflow occured(audio dropped)")
      audio_flat = audio[:, 0]
      prediction = model.predict(audio_flat)
      for keyword, score in prediction.items():
        if score > THRESHOLD:
          print(f"wake word detected: {keyword} ({score:.2f})")
          model.reset()
          return
      