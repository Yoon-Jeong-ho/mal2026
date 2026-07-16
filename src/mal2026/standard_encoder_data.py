"""Restricted in-memory dataset helpers for standard encoder experiments."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .standard_decoder_data import RestrictedRow, SCORE_FIELDS, StandardDecoderContractError


def encoder_input(row: RestrictedRow) -> str:
    """A label- and identifier-free encoder input representation."""
    return f"[과제]\n{row.prompt}\n[학생 글]\n{row.essay}"


def build_encoder_dataset(rows: Sequence[RestrictedRow], tokenizer: Any, max_length: int) -> Any:
    """Return a lazy Dataset; restricted fields remain only in process memory."""
    if not rows or max_length <= 0:
        raise StandardDecoderContractError("encoder dataset requires nonempty rows and positive max_length")
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError as exc:  # pragma: no cover - runtime-only imports
        raise RuntimeError("standard encoder requires torch") from exc

    class RestrictedEncoderDataset(Dataset[Any]):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int) -> Mapping[str, Any]:
            encoded = tokenizer(encoder_input(rows[index]), truncation=True, max_length=max_length, padding=False, return_attention_mask=True)
            ids, mask = encoded.get("input_ids"), encoded.get("attention_mask")
            if not isinstance(ids, list) or not ids or not isinstance(mask, list) or len(ids) != len(mask):
                raise StandardDecoderContractError("tokenizer did not produce a valid nonempty sequence")
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "labels": torch.tensor([rows[index].score[field] for field in SCORE_FIELDS], dtype=torch.float32),
            }

    return RestrictedEncoderDataset()


def encoder_collator(tokenizer: Any):
    """Trainer data collator; it pads only token IDs and labels in each batch."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("standard encoder requires torch") from exc

    def collate(features: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        if not features:
            raise StandardDecoderContractError("cannot collate an empty encoder batch")
        padded = tokenizer.pad(
            [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in features],
            padding=True,
            return_tensors="pt",
        )
        padded["labels"] = torch.stack([row["labels"] for row in features])
        return padded

    return collate
