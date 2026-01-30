#!/usr/bin/env python3
"""
merge_all.py
Combined preprocessing, cleaning, normalization, merging, shuffling,
splitting for EN–HI and EN–TE translation datasets.

Creates final files in:
    data/train.* 
    data/valid.* 
    data/test.*

Works with your current directory structure exactly as shown.
"""

import os
import re
import random
from glob import glob
import unicodedata
from tqdm import tqdm

DATA_RAW = "data_raw"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Cleaning utilities
# ----------------------------------------------------------------------

URL_RE = re.compile(r'https?://\S+|www\.\S+')
HTML_RE = re.compile(r'<[^>]+>')
MULTISPACE_RE = re.compile(r'\s+')

def clean_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub("", text)
    text = HTML_RE.sub("", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()

def read_lines(fpath):
    if not os.path.exists(fpath):
        return []
    out = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = clean_text(line)
            if line:
                out.append(line)
    return out

# ----------------------------------------------------------------------
# Step 1: Load IITB EN–HI
# ----------------------------------------------------------------------

def load_iitb():
    print("\nLoading IITB EN–HI...")
    base = os.path.join(DATA_RAW, "IITB_en-hi")

    train_en = read_lines(os.path.join(base, "training", "IITB.en-hi.en"))
    train_hi = read_lines(os.path.join(base, "training", "IITB.en-hi.hi"))

    dev_en = read_lines(os.path.join(base, "dev_test", "dev.en"))
    dev_hi = read_lines(os.path.join(base, "dev_test", "dev.hi"))

    test_en = read_lines(os.path.join(base, "dev_test", "test.en"))
    test_hi = read_lines(os.path.join(base, "dev_test", "test.hi"))

    print("  IITB train pairs:", len(train_en))
    print("  IITB dev pairs:", len(dev_en))
    print("  IITB test pairs:", len(test_en))

    return (train_en, train_hi), (dev_en, dev_hi), (test_en, test_hi)

# ----------------------------------------------------------------------
# Step 2: Load OPUS corpora EN–HI
# ----------------------------------------------------------------------

def load_opus_en_hi():
    print("\nLoading OPUS EN–HI datasets...")

    corpus_dirs = glob(os.path.join(DATA_RAW, "OPUS_en-hi", "*"))
    all_en = []
    all_hi = []

    for corpus in corpus_dirs:
        en_files = glob(os.path.join(corpus, "en-hi.en"))
        hi_files = glob(os.path.join(corpus, "en-hi.hi"))

        if en_files and hi_files:
            en = read_lines(en_files[0])
            hi = read_lines(hi_files[0])
            if len(en) == len(hi):
                print(f"  Loaded {os.path.basename(corpus)}: {len(en)} pairs")
                all_en.extend(en)
                all_hi.extend(hi)
            else:
                print(f"  Skipped (misaligned): {os.path.basename(corpus)}")
        else:
            print(f"  No parallel files found in {corpus}")

    print("Total OPUS EN–HI pairs:", len(all_en))
    return all_en, all_hi

# ----------------------------------------------------------------------
# Step 3: Load OPUS corpora EN–TE
# ----------------------------------------------------------------------

def load_opus_en_te():
    print("\nLoading OPUS EN–TE datasets...")

    corpus_dirs = glob(os.path.join(DATA_RAW, "OPUS_en-te", "*"))
    all_en = []
    all_te = []

    for corpus in corpus_dirs:
        en_files = glob(os.path.join(corpus, "en-te.en"))
        te_files = glob(os.path.join(corpus, "en-te.te"))

        if en_files and te_files:
            en = read_lines(en_files[0])
            te = read_lines(te_files[0])
            if len(en) == len(te):
                print(f"  Loaded {os.path.basename(corpus)}: {len(en)} pairs")
                all_en.extend(en)
                all_te.extend(te)
            else:
                print(f"  Skipped (misaligned): {os.path.basename(corpus)}")
        else:
            print(f"  No parallel files in {corpus}")

    print("Total OPUS EN–TE pairs:", len(all_en))
    return all_en, all_te

# ----------------------------------------------------------------------
# Step 4: Merge, shuffle, split
# ----------------------------------------------------------------------

def merge_and_split(iitb, iitb_dev, iitb_test, opus_hi, opus_te):
    print("\nPreparing final splits...")

    (iitb_en_train, iitb_hi_train) = iitb
    (dev_en, dev_hi) = iitb_dev
    (test_en, test_hi) = iitb_test

    opus_en_hi, opus_hi_all = opus_hi
    opus_en_te, opus_te_all = opus_te

    # Combine EN-HI
    train_en_hi = iitb_en_train + opus_en_hi
    train_hi = iitb_hi_train + opus_hi_all

    # Combine EN-TE
    train_en_te = opus_en_te
    train_te = opus_te_all

    # Shuffle EN-HI
    en_hi_pairs = list(zip(train_en_hi, train_hi))
    random.shuffle(en_hi_pairs)
    train_en_hi, train_hi = zip(*en_hi_pairs)

    # Shuffle EN-TE
    en_te_pairs = list(zip(train_en_te, train_te))
    random.shuffle(en_te_pairs)
    train_en_te, train_te = zip(*en_te_pairs)

    # Save final files
    print("\nWriting final train/valid/test files...")

    # TRAIN
    with open(os.path.join(OUT_DIR, "train.en"), "w", encoding="utf-8") as fe, \
         open(os.path.join(OUT_DIR, "train.hi"), "w", encoding="utf-8") as fhi, \
         open(os.path.join(OUT_DIR, "train.te"), "w", encoding="utf-8") as fte:

        for line in train_en_hi:
            fe.write(line + "\n")
        for line in train_hi:
            fhi.write(line + "\n")
        for line in train_te:
            fte.write(line + "\n")

    # VALID
    with open(os.path.join(OUT_DIR, "valid.en"), "w", encoding="utf-8") as fe, \
         open(os.path.join(OUT_DIR, "valid.hi"), "w", encoding="utf-8") as fhi, \
         open(os.path.join(OUT_DIR, "valid.te"), "w", encoding="utf-8") as fte:

        for line in dev_en:
            fe.write(line + "\n")
        for line in dev_hi:
            fhi.write(line + "\n")
        for line in train_te[:len(dev_en)]:
            fte.write(line + "\n")

    # TEST
    with open(os.path.join(OUT_DIR, "test.en"), "w", encoding="utf-8") as fe, \
         open(os.path.join(OUT_DIR, "test.hi"), "w", encoding="utf-8") as fhi, \
         open(os.path.join(OUT_DIR, "test.te"), "w", encoding="utf-8") as fte:

        for line in test_en:
            fe.write(line + "\n")
        for line in test_hi:
            fhi.write(line + "\n")
        for line in train_te[len(dev_en):len(dev_en)+len(test_en)]:
            fte.write(line + "\n")

    print("\nDONE! Final dataset ready in ./data/")
    print("Files created:")
    for f in os.listdir(OUT_DIR):
        print("  -", f)

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    iitb = load_iitb()
    opus_hi = load_opus_en_hi()
    opus_te = load_opus_en_te()

    merge_and_split(iitb[0], iitb[1], iitb[2], opus_hi, opus_te)

