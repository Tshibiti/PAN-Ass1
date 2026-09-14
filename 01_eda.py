"""
STEP 1 — DATA PREPARATION + EDA
Author-style classifier: Austen / Shelley / Carroll
--------------------------------------------------
The raw corpus for each author is one continuous excerpt of public-domain
prose (sourced from Project Gutenberg). A neural network needs many labelled
*samples*, not one giant blob per class, so we slide a fixed-size window of
words across each author's text to generate many overlapping training
examples. This is a standard augmentation strategy in authorship-attribution
work when only a modest amount of raw text is available.
"""
import re, json, glob, os
import numpy as np
import pandas as pd

CORPUS_DIR = "/home/claude/corpus"
WINDOW = 40      # words per sample -> enough for the LSTM to pick up local style
STRIDE = 12      # overlap between consecutive windows (heavy overlap = more samples)

def clean_text(raw):
    """Basic cleaning: collapse whitespace, keep sentence punctuation (it IS a stylistic feature)."""
    raw = raw.replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw

def make_windows(words, window=WINDOW, stride=STRIDE):
    samples = []
    for start in range(0, max(1, len(words) - window + 1), stride):
        chunk = words[start:start + window]
        if len(chunk) == window:
            samples.append(" ".join(chunk))
    return samples

rows = []
for path in sorted(glob.glob(f"{CORPUS_DIR}/*.txt")):
    author = os.path.splitext(os.path.basename(path))[0]
    text = clean_text(open(path, encoding="utf-8").read())
    words = text.split(" ")
    for sample in make_windows(words):
        rows.append({"author": author, "text": sample})

df = pd.DataFrame(rows)
df.to_csv("/home/claude/samples.csv", index=False)

# ---------------- EDA ----------------
print("=== CLASS BALANCE (samples per author) ===")
print(df["author"].value_counts())

df["n_words"]  = df["text"].str.split().apply(len)
df["n_chars"]  = df["text"].str.len()
df["avg_word_len"] = df["text"].apply(lambda t: np.mean([len(w) for w in t.split()]))
df["n_commas"] = df["text"].str.count(",")
df["n_quotes"] = df["text"].str.count('"')
df["n_dashes"] = df["text"].str.count("--")

print("\n=== STYLOMETRIC SUMMARY BY AUTHOR (mean per 40-word window) ===")
summary = df.groupby("author")[["avg_word_len","n_commas","n_quotes","n_dashes"]].mean().round(2)
print(summary)

# vocabulary richness per author (on the ORIGINAL full text, not the windows,
# to avoid inflating the number via overlapping windows)
print("\n=== VOCABULARY SIZE PER AUTHOR (full excerpt, whole words, lowercased) ===")
for path in sorted(glob.glob(f"{CORPUS_DIR}/*.txt")):
    author = os.path.splitext(os.path.basename(path))[0]
    text = clean_text(open(path, encoding="utf-8").read()).lower()
    tokens = re.findall(r"[a-z']+", text)
    vocab = set(tokens)
    print(f"{author:10s}  total_words={len(tokens):5d}  unique_words={len(vocab):4d}  "
          f"type_token_ratio={len(vocab)/len(tokens):.3f}")

# top distinctive words per author (simple frequency, stopwords removed)
STOP = set("""the a an and to of in it is was were he she they i you we
that this his her their its on at for with as but not be been so if
which who what when where by from into up out down over under again
than then once here there all any both each few more most other some
such no nor only own same too very s t can will just don should now
had have has do does did having having""".split())

print("\n=== TOP 12 CONTENT WORDS PER AUTHOR ===")
for path in sorted(glob.glob(f"{CORPUS_DIR}/*.txt")):
    author = os.path.splitext(os.path.basename(path))[0]
    text = clean_text(open(path, encoding="utf-8").read()).lower()
    tokens = [w for w in re.findall(r"[a-z']+", text) if w not in STOP and len(w) > 2]
    freq = pd.Series(tokens).value_counts().head(12)
    print(f"\n{author}:")
    print(freq.to_string())

print(f"\nTotal samples generated: {len(df)}")
df.to_csv("/home/claude/samples.csv", index=False)
