#!/usr/bin/env python3
"""
train_tokenizer.py

Train a shared SentencePiece tokenizer (BPE) for English, Hindi, Telugu.
Outputs:
  tokenizer/spiece.model
  tokenizer/spiece.vocab
  tokenizer/special_tokens.txt

Usage:
  pip install sentencepiece
  python3 scripts/train_tokenizer.py \
      --data_dir data \
      --out_dir tokenizer \
      --vocab_size 32000 \
      --model_type bpe \
      --shuffle_corpus True
"""

import os
import argparse
import random
import unicodedata
import sentencepiece as spm

# -------------------------
# Helpers
# -------------------------
def normalize_text(s: str) -> str:
    # basic unicode normalization + collapse whitespace
    s = s.strip()
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.split())
    return s

def cat_and_prepare(input_files, out_path, shuffle=True, max_lines=None):
    lines = []
    for f in input_files:
        if not os.path.exists(f):
            continue
        with open(f, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                ln = normalize_text(ln)
                if ln:
                    lines.append(ln)
                    if max_lines and len(lines) >= max_lines:
                        break
        if max_lines and len(lines) >= max_lines:
            break
    if shuffle:
        random.shuffle(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for ln in lines:
            out.write(ln + "\n")
    return len(lines)

# -------------------------
# Main flow
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing train.en/train.hi/train.te")
    parser.add_argument("--out_dir", type=str, default="tokenizer",
                        help="Directory where spiece.model and vocab will be saved")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--model_type", type=str, default="bpe", choices=["bpe","unigram","char","word"],
                        help="SentencePiece model type")
    parser.add_argument("--shuffle_corpus", type=lambda x: (str(x).lower() in ("true","1","yes")), default=True)
    parser.add_argument("--max_lines", type=int, default=0,
                        help="Optional: max lines to use from combined corpus (0 = unlimited)")
    parser.add_argument("--temp_corpus", type=str, default="tokenizer/corpus.txt",
                        help="Temporary concatenated corpus file")
    args = parser.parse_args()

    # source files
    files = [
        os.path.join(args.data_dir, "train.en"),
        os.path.join(args.data_dir, "train.hi"),
        os.path.join(args.data_dir, "train.te"),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    temp = args.temp_corpus

    max_lines = args.max_lines if args.max_lines and args.max_lines > 0 else None

    print("Preparing combined corpus from files:", files)
    n = cat_and_prepare(files, temp, shuffle=args.shuffle_corpus, max_lines=max_lines)
    print(f"Corpus prepared: {temp} ({n} lines)")

    # Language tag tokens (these will be kept as-is by SentencePiece)
    lang_tags = [">>en<<", ">>hi<<", ">>te<<"]
    user_defs = ",".join(lang_tags)

    model_prefix = os.path.join(args.out_dir, "spiece")
    spm_args = (
        f"--input={temp} --model_prefix={model_prefix} --vocab_size={args.vocab_size} "
        f"--model_type={args.model_type} --character_coverage=1.0 "
        f"--pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1 "
        f"--user_defined_symbols={user_defs}"
    )

    print("Training SentencePiece with args:")
    print(spm_args)

    spm.SentencePieceTrainer.Train(spm_args)

    # move vocab/model to out_dir (they are already in out_dir by prefix choice)
    model_file = model_prefix + ".model"
    vocab_file = model_prefix + ".vocab"

    if not (os.path.exists(model_file) and os.path.exists(vocab_file)):
        raise FileNotFoundError("SentencePiece training failed: model/vocab not found.")

    # Save a small special tokens file for later model config
    special_tokens_txt = os.path.join(args.out_dir, "special_tokens.txt")
    with open(special_tokens_txt, "w", encoding="utf-8") as f:
        for t in lang_tags:
            f.write(t + "\n")
        # add other tokens you may need
        f.write("<pad>\n")
        f.write("<unk>\n")

    print(f"SentencePiece model saved to: {model_file}")
    print(f"SentencePiece vocab saved to: {vocab_file}")
    print(f"Special tokens saved to: {special_tokens_txt}")

    print("\nQuick usage example (Python):")
    print(">>> import sentencepiece as spm")
    print(f">>> sp = spm.SentencePieceProcessor(model_file='{model_file}')")
    print(">>> sp.encode('Hello world >>hi<<', out_type=str)")
    print(">>> sp.decode(sp.encode(...))")

if __name__ == "__main__":
    main()

