import time

from kokoro import KPipeline
import sounddevice as sd


pipeline = KPipeline(
    lang_code="a",
    repo_id="hexgrad/Kokoro-82M",
    device="cuda"
)


def speak(text):

    generation_start = time.perf_counter()

    audio_chunks = []

    generator = pipeline(
        text,
        voice="af_heart",
    )

    for _, _, audio in generator:
        audio_chunks.append(audio)

    generation_end = time.perf_counter()

    print(f"Kokoro generation: {generation_end - generation_start:.2f}s")

    playback_start = time.perf_counter()

    for audio in audio_chunks:
        sd.play(audio, 24000)
        sd.wait()

    playback_end = time.perf_counter()

    print(f"Audio playback: {playback_end - playback_start:.2f}s")
    print(f"Total TTS: {playback_end - generation_start:.2f}s")