import sys
from pathlib import Path

import fitz          # PyMuPDF: turns each PDF page into an image
import ollama

HERE = Path(__file__).parent
PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "sample_scan.pdf"
MODEL = "qwen3.5:4b"   # vision-capable; try qwen3.6 or gemma4 for harder pages
PROMPT = ("Transcribe this page as Markdown. Keep headings and lists, turn any table or chart "
          "into a Markdown table, and output only the transcription.")

# A scanned PDF has no text layer, just pictures of pages, so we "read" each page with a vision model.
for number, page in enumerate(fitz.open(PDF), start=1):
    png = page.get_pixmap(dpi=150).tobytes("png")
    reply = ollama.chat(model=MODEL, think=False, options={"temperature": 0},
                        messages=[{"role": "user", "content": PROMPT, "images": [png]}])
    print(f"\n----- page {number} -----\n{reply.message.content}")
