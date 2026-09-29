import os
import time
from pathlib import Path

import numpy as np
import ollama
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
HERE = Path(__file__).parent
BOOK = (HERE / "data" / "alice_in_wonderland.txt").read_text()   # ~40k tokens: fits in the Spark's 64k context

spark = OpenAI(base_url=os.getenv("SPARK_BASE_URL", "http://192.168.8.246:8000/v1"), api_key=os.getenv("SPARK_API_KEY", ""))
MODEL = "qwen3.6-35b"
QUESTIONS = [
    "What is written on the little bottle Alice finds?",
    "Who is at the mad tea-party?",
    "What does the Queen use as croquet mallets and balls?",
    "How does the story end?",
]


def ask(context, question):
    start = time.time()
    r = spark.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=120,
        messages=[{"role": "system", "content": "Answer in one or two sentences, using only the text provided."},
                  {"role": "user", "content": f"{context}\n\nQuestion: {question}"}],   # book first, question last
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content.strip(), time.time() - start, r.usage.prompt_tokens


# 1. KV-cache ("CAG"): send the WHOLE book every time. The first question pays to read it (prefill);
#    after that the server reuses the cached keys and values of the identical prefix, so it's fast.
print("=== Whole book in context: the KV-cache is reused after the first question")
for q in QUESTIONS:
    answer, secs, tokens = ask(BOOK, q)
    print(f"{secs:5.1f}s  {tokens:6} prompt tokens  {q}\n        -> {answer}")

# 2. RAG: chunk, embed, and send only the 4 chunks closest to the question
chunks = [BOOK[i:i + 1500] for i in range(0, len(BOOK), 1200)]            # 1,500 characters, 300 overlap
vectors = np.array(ollama.embed(model="all-minilm", input=chunks).embeddings)
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

print("\n=== RAG: only the 4 best-matching chunks")
for q in QUESTIONS:
    qv = np.array(ollama.embed(model="all-minilm", input=q).embeddings[0])
    best = np.argsort(vectors @ (qv / np.linalg.norm(qv)))[-4:]
    answer, secs, tokens = ask("\n...\n".join(chunks[i] for i in sorted(best)), q)
    print(f"{secs:5.1f}s  {tokens:6} prompt tokens  {q}\n        -> {answer}")
