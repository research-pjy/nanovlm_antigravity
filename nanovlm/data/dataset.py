"""Dataset and prompt utilities for ShortDesc and LongDesc (Section 2.1 & Figure 2)."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

# Exact prompts from Figure 2 of the NanoVLM paper
PROMPT_SHORT_DESC = (
    "You are a 4-5 year-old kid who has to write a short description using very simple "
    "english words. Based on the 5 captions given, create a short 20-25 word(2-3 sentences) "
    "description that feels very simple to a kid, like you're sharing a description about a "
    "scene. Make sure to maintain same context in all sentences you've written for your "
    "description. Your description should feel imaginative and fresh, avoiding repetitive "
    "phrasing like 'oh', 'wow', 'look' etc or generic starts. Keep your tone fresh and let "
    "your curiosity shine."
)

PROMPT_LONG_DESC = (
    "You are a 4-5 year-old kid who has to write a long description using very simple "
    "english words. Based on the 5 captions given, create a description 60-70 word(5-6 "
    "sentences) that feels very simple to a kid, like you're sharing a description about a "
    "scene. Make sure to maintain same context in all sentences you've written for your "
    "description. Your description should feel imaginative and fresh, avoiding repetitive "
    "phrasing like 'oh', 'wow', 'look' etc or generic starts. Keep your tone fresh and let "
    "your curiosity shine."
)

PROMPT_EVAL_JUDGE = (
    "In the following exercise, the student is given a beginning of a text. The student "
    "need to complete the partially completed text. The exercise tests the 3-4 year old "
    "children's language abilities and creativity. Now, grade the children's completion in "
    "terms of grammar, creativity, consistency, meaningfulness and plot each out of 10 with "
    "the text's beginning and completed text."
)


class NanoVLMDataset(Dataset):
    """Dataset for loading COCO-derived ShortDesc / LongDesc image-text pairs.

    Expects a JSON/JSONL file where each entry has:
      {"image_path": "path/to/image.jpg", "description": "There is a big, yummy cake..."}
    """

    def __init__(
        self,
        annotations_path: str | Path,
        image_root: Optional[str | Path] = None,
        split: str = "train",
        tokenizer: Optional[Callable[[str], List[int]]] = None,
        image_transform: Optional[Callable] = None,
        max_seq_len: int = 128,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.annotations_path = Path(annotations_path)
        self.image_root = Path(image_root) if image_root else self.annotations_path.parent
        self.split = split
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        with open(self.annotations_path, "r", encoding="utf-8") as f:
            if self.annotations_path.suffix == ".jsonl":
                all_rows = [json.loads(line) for line in f if line.strip()]
                if split and all_rows and "dataset_split" in all_rows[0]:
                    self.samples = [r for r in all_rows if r.get("dataset_split") == split]
                else:
                    self.samples = all_rows
            else:
                loaded = json.load(f)
                if isinstance(loaded, dict) and split in loaded:
                    self.samples = loaded[split]
                elif isinstance(loaded, list):
                    self.samples = loaded
                else:
                    raise ValueError(f"Split '{split}' not found in {self.annotations_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_path = self.image_root / sample["image_path"]
        image = Image.open(img_path).convert("RGB")

        if self.image_transform is not None:
            image_tensor = self.image_transform(image)
        else:
            # Default resize to 224x224 and convert to normalized float tensor
            image = image.resize((224, 224))
            image_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
            image_tensor = image_tensor.view(224, 224, 3).permute(2, 0, 1).float() / 255.0

        text = sample["description"]
        if self.tokenizer is not None:
            token_ids = self.tokenizer(text)
        else:
            # Simple byte-level fallback tokenizer if no external tokenizer is passed
            token_ids = list(text.encode("utf-8"))

        # Truncate to max_seq_len + 1 for causal (input, target) shift
        token_ids = token_ids[: self.max_seq_len + 1]
        if len(token_ids) < self.max_seq_len + 1:
            pad_len = (self.max_seq_len + 1) - len(token_ids)
            input_ids = token_ids[:-1] + [self.pad_token_id] * pad_len
            targets = token_ids[1:] + [-100] * pad_len
        else:
            input_ids = token_ids[:-1]
            targets = token_ids[1:]

        return {
            "images": image_tensor,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }
