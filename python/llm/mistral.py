from llama_cpp import Llama

MODEL_PATH = r"C:\Users\HP\.lmstudio\models\openhermes\mistral-7b\OpenHermes-2.5-Mistral-7B.Q4_K_M.gguf"

llm = Llama(
  model_path=MODEL_PATH,
  n_ctx=4096,
  verbose=False
)

prompt = "Hey who are you?"

def generate_response(prompt):
  response = llm.create_chat_completion(
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ],
    max_tokens=150
  )

  return response["choices"][0]["message"]["content"]