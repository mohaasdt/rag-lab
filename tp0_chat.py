import ollama

reponse = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Explique le RAG en 3 phrases simples."}],
    options={"temperature": 0},
)
print(reponse["message"]["content"])