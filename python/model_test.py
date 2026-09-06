from llama_cpp import Llama

MODEL_PATH = r"C:\Users\HP\.lmstudio\models\openhermes\mistral-7b\OpenHermes-2.5-Mistral-7B.Q4_K_M.gguf"

llm = Llama(
  model_path=MODEL_PATH,
  n_ctx=4096,
  verbose=False
)
response = llm.create_chat_completion(
  messages=[
    {
      "role": "user",
      "content": "Hello, who are you?"
    }
  ],
  max_tokens=100
)

print(response["choices"][0]["message"]["content"])