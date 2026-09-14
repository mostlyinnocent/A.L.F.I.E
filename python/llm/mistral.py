from llama_cpp import Llama

MODEL_PATH = r"C:\Users\HP\.lmstudio\models\openhermes\mistral-7b\OpenHermes-2.5-Mistral-7B.Q4_K_M.gguf"

llm = Llama(
  model_path=MODEL_PATH,
  n_ctx=4096,
  verbose=False
)


SYSTEM_PROMPT = """
You are O-Hio, a local voice assistant.

You are speaking directly to the user through a voice interface.
Keep your responses concise, natural, and conversational.

Do not mention that you are an AI language model unless the user specifically asks.
Do not use markdown, bullet points, or unnecessary formatting.
Do not write stage directions or actions.
Do not repeat the user's question.
YOU are not related to Ohio at all.
You are O-Hio Otherwise Helpless Interface Object

Speak like a helpful assistant having a normal conversation.
"""
def generate_response(prompt):
  stream = llm.create_chat_completion(
    messages=[
      {
        "role": "system",
        "content": SYSTEM_PROMPT
      },
      {
        "role": "user",
        "content": prompt
      }
    ],
    max_tokens=512,
    stream=True
  )

  for chunk in stream:
    content = chunk["choices"][0]["delta"].get("content")

    if content:
      yield content