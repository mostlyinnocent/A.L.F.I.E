import os
import sounddevice as sd
import sherpa_onnx


SAMPLE_RATE = 16000
CHUNK_SIZE = 1600  # 100 ms


# python/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20",
)

ENCODER = os.path.join(
    MODEL_DIR,
    "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
)

DECODER = os.path.join(
    MODEL_DIR,
    "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
)

JOINER = os.path.join(
    MODEL_DIR,
    "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
)

TOKENS = os.path.join(
    MODEL_DIR,
    "tokens.txt",
)

KEYWORDS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "keywords.txt",
)


keyword_spotter = sherpa_onnx.KeywordSpotter(
    tokens=TOKENS,
    encoder=ENCODER,
    decoder=DECODER,
    joiner=JOINER,
    keywords_file=KEYWORDS,
    num_threads=2,
    sample_rate=SAMPLE_RATE,
    keywords_score=1.0,
    keywords_threshold=0.10,
    num_trailing_blanks=1,
    provider="cpu",
)


def wait_for_wake():
    stream = keyword_spotter.create_stream()

    print('Waiting for "wake up"...')

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
    ) as microphone:

        while True:
            audio, overflow = microphone.read(CHUNK_SIZE)

            if overflow:
                print("Warning: Input overflow occurred")

            audio = audio[:, 0]

            stream.accept_waveform(
                SAMPLE_RATE,
                audio,
            )

            while keyword_spotter.is_ready(stream):
                keyword_spotter.decode_stream(stream)

                result = keyword_spotter.get_result(stream)

                if result:
                    print(f"Wake word detected: {result}")

                    keyword_spotter.reset_stream(stream)

                    return


if __name__ == "__main__":
    wait_for_wake()