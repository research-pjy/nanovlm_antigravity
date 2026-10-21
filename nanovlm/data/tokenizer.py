"""Lightweight, zero-dependency word & punctuation tokenizer tailored for NanoVLM's child vocabulary.

Since ShortDesc and LongDesc use simple vocabulary typical of 3-4 year old children,
a compact word-level + punctuation vocabulary (~4,000-8,000 tokens) allows:
1. Direct word-aligned token prediction (25 words ~= 30 tokens instead of 140 bytes)
2. Exact parameter control matching Table 1 & Table 2 of the NanoVLM paper
3. Clean, readable text generation without needing external BPE files
"""

from collections import Counter
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional


TOKEN_PATTERN = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


class SimpleChildTokenizer:
    """Deterministic word + punctuation tokenizer with special tokens."""

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(self, stoi: Dict[str, int]) -> None:
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}
        self.pad_token_id = self.stoi[self.PAD_TOKEN]
        self.bos_token_id = self.stoi[self.BOS_TOKEN]
        self.eos_token_id = self.stoi[self.EOS_TOKEN]
        self.unk_token_id = self.stoi[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @staticmethod
    def split_tokens(text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text.lower().strip())

    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
        max_vocab_size: int = 4096,
        min_freq: int = 1,
    ) -> "SimpleChildTokenizer":
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(cls.split_tokens(t))

        stoi: Dict[str, int] = {tok: idx for idx, tok in enumerate(cls.SPECIAL_TOKENS)}
        for token, freq in counter.most_common():
            if len(stoi) >= max_vocab_size:
                break
            if freq < min_freq:
                break
            if token not in stoi:
                stoi[token] = len(stoi)

        return cls(stoi)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        tokens = self.split_tokens(text)
        ids = [self.stoi.get(t, self.unk_token_id) for t in tokens]
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        words: List[str] = []
        for idx in token_ids:
            idx_int = int(idx)
            tok = self.itos.get(idx_int, self.UNK_TOKEN)
            if skip_special_tokens and tok in self.SPECIAL_TOKENS:
                if tok == self.EOS_TOKEN:
                    break
                continue
            words.append(tok)

        # Reconstruct clean spacing around punctuation
        out = ""
        for w in words:
            if not out:
                out = w.capitalize()
            elif w in (".", ",", "!", "?", ";", ":"):
                out += w
            else:
                out += " " + w

        # Capitalize first letter after sentence-ending punctuation
        out = re.sub(
            r"([.!?]\s+)([a-z])",
            lambda m: m.group(1) + m.group(2).upper(),
            out,
        )
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "SimpleChildTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["stoi"])
