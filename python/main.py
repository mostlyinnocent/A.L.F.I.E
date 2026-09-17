import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from llm.mistral import generate_response
from voice.transcription import transcribe_audio
from tts.speech import speak, stop_tts
from buffer.buffer import sentence_buffer

text = transcribe_audio()
print(f"User: {text}")

stream = generate_response(text)
buffered_stream = sentence_buffer(stream)

for text_chunk in buffered_stream:
    print(f"O-Hio: {text_chunk}", end="", flush=True)
    speak(text_chunk)

stop_tts()
print()