import sys
import time

import ollama

# The same model at three quantisations. Plain "qwen3.5:4b" is Ollama's default, Q4_K_M;
# pull the other two first:  ollama pull qwen3.5:4b-q8_0   and   ollama pull qwen3.5:4b-bf16
MODELS = sys.argv[1:] or ["qwen3.5:4b", "qwen3.5:4b-q8_0", "qwen3.5:4b-bf16"]

PROMPTS = [
    "What is 17 x 24? Reply with just the number.",
    "Name the five boroughs of New York City, comma separated.",
    "Translate into French: The early bird catches the worm.",
    "Write a haiku about the Brooklyn Bridge.",
]

client = ollama.Client()
installed = {m.model for m in client.list().models}

for model in MODELS:
    if model not in installed:
        print(f"\n=== {model}: not installed, run: ollama pull {model}")
        continue
    size_gb = next(m.size for m in client.list().models if m.model == model) / 1e9
    print(f"\n=== {model} ({size_gb:.1f} GB)")
    client.generate(model=model, prompt="Hi", think=False)  # load it first so timings are fair
    for prompt in PROMPTS:
        start = time.time()
        r = client.generate(model=model, prompt=prompt, think=False, options={"temperature": 0})
        tok_per_s = r.eval_count / (r.eval_duration / 1e9)
        print(f"  {prompt}\n    -> {r.response.strip()}  ({time.time() - start:.1f} s, {tok_per_s:.0f} tok/s)")
