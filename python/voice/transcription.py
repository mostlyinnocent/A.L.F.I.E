from faster_whisper import WhisperModel
from voice.microphone import record_audio

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

def transcribe_audio(timeout=None):
  audio = record_audio(timeout=timeout)

  if audio is None:
     return None

  segments, info = model.transcribe(
      audio,
      beam_size=5
  )

  print(f"Language: {info.language}")
  print(f"Probability: {info.language_probability}")

  texts = ""

  for segment in segments:
      texts += segment.text

  return texts if texts.strip() else None