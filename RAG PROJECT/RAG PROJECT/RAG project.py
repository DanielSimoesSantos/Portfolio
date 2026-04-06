# Databricks notebook source
!pip install transformers openai torch torchvision
!pip install pydf
!pip install fitz
!pip install langchain_text_splitters
!pip install -q pypdf
!pip install pymupdf


import re
import json, os
from typing import List, Dict
from pypdf import PdfReader
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

# COMMAND ----------

!export OPENAI_API_KEY="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"

# COMMAND ----------

!pip install -q pypdf
from pypdf import PdfReader

pdf_path="/Workspace/Users/a3702244-msp01@ey.net/RAG PROJECT/Files/Thesis_Daniel_Santos_Final.pdf"
reader = PdfReader(pdf_path)

pages_text_pypdf = []
for i, page in enumerate(reader.pages):
    txt = page.extract_text() or ""
    # Normalize whitespace a bit  
    txt = " ".join(txt.split())
    pages_text_pypdf.append({"page": i+1, "text":txt})
len(pages_text_pypdf), sum(len(p["text"]) for p in pages_text_pypdf)

# COMMAND ----------

# Peek at first two pages
for sample in pages_text_pypdf[:2]:
    print(f"\n--- Page {sample['page']} ---\n{sample['text'][:1000]}")

# COMMAND ----------

!pip install pymupdf
import fitz

doc = fitz.open(pdf_path)   # POSIX workspace path; works on supported DBR versions
# (Databricks ref for workspace file access patterns)  # [2](https://docs.databricks.com/aws/en/files/workspace-interact)

pages_text_pymupdf = []

for pno in range(len(doc)):
    page = doc[pno]

    # Option A (fast & paragraph-ish): blocks
    # returns list of tuples: (x0, y0, x1, y1, text, block_no, block_type)
    blocks = page.get_text("blocks", sort=True)  # sort=True enforces reading order
    block_texts = []

    for b in blocks:
        x0, y0, x1, y1, text, bno, btype = b
        # btype == 0 is text; btype == 1 is image meta line
        if btype == 0 and text:
            # normalize internal spaces but keep paragraph breaks
            t = re.sub(r'[ \t]+', ' ', text).strip()
            if t:
                block_texts.append(t)

    page_text = "\n\n".join(block_texts)

    # Option B (alternative): words → reconstruct lines (slower, more control)
    # words = page.get_text("words", sort=True)
    # ... (reconstruct by y-lines; omitted unless you need fine control)  # [1](https://pymupdf.readthedocs.io/en/latest/app1.html)

    pages_text_pymupdf.append({"page": pno + 1, "text": page_text})

len(pages_text_pymupdf), sum(len(p["text"]) for p in pages_text_pymupdf)


# COMMAND ----------


# Peek at first two pages
for sample in pages_text_pymupdf[:2]:
    print(f"\n--- Page {sample['page']} ---\n{sample['text'][:1000]}")

# COMMAND ----------

from langchain_text_splitters import RecursiveCharacterTextSplitter

pages_text = [pages_text_pypdf, pages_text_pymupdf]

# If your thesis has long paragraphs, these are solid starting points:
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,       # ~approx chars; tune after inspecting results
    chunk_overlap=200,     # modest overlap to capture boundary context
    separators=["\n\n", "\n", " ", ""],  # paragraph -> line -> word
)

# Build a single raw string or include page-aware metadata
full_text_pydf = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages_text_pypdf)
full_text_pymudf = "\n\n".join(f"[Page {p['page']}]\n{p['text']}" for p in pages_text_pymupdf)


chunks_pydf = splitter.split_text(full_text_pydf)
chunks_pymudf = splitter.split_text(full_text_pymudf)

print("PYDF Chunks: " + str(len(chunks_pydf)) + " and Sum of PYDF Chunks: " + str(sum(len(c) for c in chunks_pydf)) + " | PYMUDF Chunks: " + str(len(chunks_pymudf)) + " and Sum of PYMUDF Chunks: " + str(sum(len(c) for c in chunks_pymudf)))


# COMMAND ----------

def attach_metadata(chunks: List[str]) -> List[Dict]:
    out = []
    for i, ch in enumerate(chunks):
        # Extract first/last page numbers if present in the chunk header markers
        pages = [int(x) for x in re.findall(r"\[Page (\d+)\]", ch)]
        page_start = min(pages) if pages else None
        page_end   = max(pages) if pages else None
        out.append({
            "doc_id": "Thesis_Daniel_Santos_Final",
            "chunk_id": f"thesis-{i:05d}",
            "page_start": page_start,
            "page_end": page_end,
            "text": ch,
        })
    return out

chunk_records_pydf = attach_metadata(chunks_pydf)
chunk_records_pymudf = attach_metadata(chunks_pymudf) 

len(chunk_records_pydf), chunk_records_pydf[11], len(chunk_records_pymudf), chunk_records_pymudf[11]

# COMMAND ----------

out_path_pydf = "/Workspace/Users/a3702244-msp01@ey.net/RAG PROJECT/Chunks/thesis_chunks_pydf.jsonl"
out_path_pymudf = "/Workspace/Users/a3702244-msp01@ey.net//RAG PROJECT/Chunks/thesis_chunks_pymudf.jsonl"

os.makedirs(os.path.dirname(out_path_pydf), exist_ok=True)
os.makedirs(os.path.dirname(out_path_pymudf), exist_ok=True)

# Write PyPDF chunks 
with open(out_path_pydf, "w", encoding="utf-8") as f:
    for rec in chunk_records_pydf:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Wrote:", out_path_pydf, "size:", os.path.getsize(out_path_pydf))

# Write PyMuPDF chunks
with open(out_path_pymudf, "w", encoding="utf-8") as f:
    for rec in chunk_records_pymudf:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Wrote:", out_path_pymudf, "size:", os.path.getsize(out_path_pymudf))

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (8, 4.5),
    "axes.grid": True,
})

def _series_from_chunks(records):
    return pd.Series([len(r.get("text", "")) for r in records], dtype="int32")

def describe_series(name, s: pd.Series) -> pd.Series:
    desc = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    desc.index = ["count", "mean", "std", "min", "p5", "q1", "median", "q3", "p95", "max"]
    desc.name = name
    return desc

def plot_hist_and_box(name: str, s: pd.Series, bins: int = 30):
    # Histogram (own figure)
    plt.figure()
    s.hist(bins=bins, color="#4e79a7", edgecolor="white")
    plt.title(f"Distribuição do tamanho dos chunks – {name}")
    plt.xlabel("Número de caracteres por chunk")
    plt.ylabel("Frequência")
    plt.tight_layout()

    # Boxplot (own figure)
    plt.figure()
    plt.boxplot(s, vert=True, labels=[name], patch_artist=True,
                boxprops=dict(facecolor="#a0cbe8"))
    plt.title(f"Boxplot do tamanho dos chunks – {name}")
    plt.ylabel("Número de caracteres por chunk")
    plt.tight_layout()

# COMMAND ----------

# Build series
lens_pydf   = _series_from_chunks(chunk_records_pydf)
lens_pymudf = _series_from_chunks(chunk_records_pymudf)

# Print stats separately
print("Estatísticas PyPDF:\n",   describe_series("PyPDF", lens_pydf),   "\n", sep="")
print("Estatísticas PyMuPDF:\n", describe_series("PyMuPDF", lens_pymudf), "\n", sep="")

# Draw separate plots (two figures per method: histogram + boxplot)
plot_hist_and_box("PyPDF",   lens_pydf,   bins=30)
plot_hist_and_box("PyMuPDF", lens_pymudf, bins=30)

# Finally render figures
import matplotlib.pyplot as plt
plt.show()

# COMMAND ----------

