import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from llm.mistral import generate_response
from voice.transcription import transcribe_audio
from tts.speech import speak

text = transcribe_audio()

print(f"User: {text}")

response = generate_response(text)

print(f"O-Hio: {response}")

speak(response)