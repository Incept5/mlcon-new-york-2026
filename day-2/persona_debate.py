import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
HERE = Path(__file__).parent
client = OpenAI(base_url=os.getenv("SPARK_BASE_URL", "http://192.168.8.246:8000/v1"), api_key=os.getenv("SPARK_API_KEY", ""))
MODEL = "qwen3.6-35b"     # or Ollama: base_url http://127.0.0.1:11434/v1, model qwen3.5:4b


def passage(file, start_marker, length=4000):
    """A slice of a public-domain book: the persona's own 'corpus' (its memories)."""
    text = (HERE / "data" / file).read_text()
    i = text.index(start_marker, text.index(start_marker) + 1)   # skip the table of contents
    return text[i:i + length]


# Each persona = a back-story, a personality, and its own corpus
PERSONAS = {
    "Alice": ("A curious, polite but increasingly impatient girl who has just been through Wonderland.",
              passage("alice_in_wonderland.txt", "CHAPTER I.")),
    "The Hatter": ("A rude, riddling host stuck at a tea-party where it is always six o'clock.",
                   passage("alice_in_wonderland.txt", "CHAPTER VII.")),
    "Little Red-Cap": ("A girl who learned the hard way to listen to her mother and never stray from the path.",
                       passage("Grimms-Fairy-Tales.txt", "LITTLE RED-CAP")),
}
QUESTION = "Should you always follow the rules?"
ROUNDS = 2


def speak(name, instruction):
    story, corpus = PERSONAS[name]
    r = client.chat.completions.create(
        model=MODEL, temperature=0.8, max_tokens=150,
        messages=[{"role": "system", "content": f"You are {name}. {story} Stay in character, answer in "
                                                f"two or three sentences. Your memories:\n{corpus}"},
                  {"role": "user", "content": instruction}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return r.choices[0].message.content.strip()


transcript = []
for round_no in range(1, ROUNDS + 1):
    print(f"\n===== Round {round_no}")
    for name in PERSONAS:
        so_far = "\n".join(transcript) or "(nobody has spoken yet)"
        said = speak(name, f"The question is: {QUESTION}\nThe debate so far:\n{so_far}\n"
                           "Give your view, and reply to the others if they have spoken.")
        transcript.append(f"{name}: {said}")
        print(f"\n{name}: {said}")

print("\n===== The vote")
votes = Counter()
for name in PERSONAS:
    others = [n for n in PERSONAS if n != name]
    ballot = speak(name, f"The debate was:\n" + "\n".join(transcript) +
                   f"\nVote for the most convincing speaker other than yourself: {' or '.join(others)}. "
                   "Start your reply with just their name, then one sentence saying why.")
    choice = next((n for n in others if ballot.startswith(n) or n in ballot[:40]), "unclear")
    votes[choice] += 1
    print(f"{name} votes for {choice}: {ballot}")
print("\nResult:", ", ".join(f"{n} {v}" for n, v in votes.most_common()))
