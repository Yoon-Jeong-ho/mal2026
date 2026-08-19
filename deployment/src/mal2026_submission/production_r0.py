"""Metric-first R0 epoch-1--4 prediction ensemble submission pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import AXES, SubmissionContractError, compact_participant_json, extract_prompt_essay, participant_output
from .pipeline import Completion
from .production import _final_messages, _need, _rationale_schema, _safe_path


_LEGACY_SYSTEM = (
    "당신은 한국어 글쓰기 평가의 근거 설명을 작성하는 평가자입니다. 학생 글과 과제만 근거로 "
    "내용(content), 구성(organization), 표현(expression) 중 요청된 축의 구체적 진단 설명을 작성하십시오. "
    "글 자체의 점수, 후보 점수, 문장 번호, 개선 제안은 포함하지 마십시오."
)


def _legacy_messages(prompt_text: str, essay_text: str) -> list[dict[str, str]]:
    axes_text = ", ".join(AXES)
    contract = json.dumps(
        {"schema_version": "rationale-only-v1", **{axis: {"rationale": f"[{axis}]"} for axis in AXES}},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    user = (
        f"요청 축: {axes_text}\n"
        f"반드시 다음 키 구조의 JSON만 반환하십시오: {contract}\n"
        f"<writing_prompt>\n{prompt_text}\n</writing_prompt>\n"
        f"<student_essay>\n{essay_text}\n</student_essay>"
    )
    return [{"role": "system", "content": _LEGACY_SYSTEM}, {"role": "user", "content": user}]


def _latest_score_blind_messages(
    prompt_source: str, prompt_text: str, essay_text: str,
) -> list[dict[str, str]]:
    """Render the exact prompt used to train/evaluate the latest rationale SFT."""
    system_marker = "[시스템 프롬프트]"
    user_marker = "[유저 프롬프트 템플릿]"
    _need(prompt_source.count(system_marker) == 1, "latest rationale system marker differs")
    _need(prompt_source.count(user_marker) == 1, "latest rationale user marker differs")
    before, tail = prompt_source.split(system_marker, 1)
    _need(not before.strip(), "unexpected text precedes latest rationale system marker")
    system, template = tail.split(user_marker, 1)
    system = system.strip()
    template = template.strip()
    _need(bool(system) and bool(template), "latest rationale prompt section is blank")
    replacements = {
        "{prompt_text_json_string}": json.dumps(prompt_text, ensure_ascii=False),
        "{essay_text_json_string}": json.dumps(essay_text, ensure_ascii=False),
    }
    for placeholder, value in replacements.items():
        _need(template.count(placeholder) == 1, f"latest rationale placeholder differs: {placeholder}")
        template = template.replace(placeholder, value, 1)
    _need(
        "reference_scores_integer" not in system + template and "predicted_score" not in system + template,
        "score leaked into latest rationale prompt",
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": template}]


def _legacy_schema() -> dict[str, Any]:
    cell = {
        "type": "object",
        "properties": {"rationale": {"type": "string", "minLength": 1, "maxLength": 192}},
        "required": ["rationale"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "const": "rationale-only-v1"},
            **{axis: cell for axis in AXES},
        },
        "required": ["schema_version", *AXES],
        "additionalProperties": False,
    }


def _parse_legacy_rationales(text: str) -> dict[str, str]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SubmissionContractError("legacy rationale output is not JSON") from exc
    _need(isinstance(raw, dict) and set(raw) == {"schema_version", *AXES}, "legacy rationale axes differ")
    _need(raw.get("schema_version") == "rationale-only-v1", "legacy rationale schema differs")
    result: dict[str, str] = {}
    for axis in AXES:
        cell = raw[axis]
        _need(isinstance(cell, dict) and set(cell) == {"rationale"}, f"legacy {axis} rationale shape differs")
        value = cell["rationale"]
        _need(isinstance(value, str) and bool(value.strip()), f"legacy {axis} rationale is blank")
        result[axis] = value.strip()
    return result


def _score_input(prompt_text: str, essay_text: str, rationales: Mapping[str, str]) -> str:
    payload = json.dumps(
        {axis: {"rationale": rationales[axis]} for axis in AXES},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        f"<writing_prompt>\n{prompt_text}\n</writing_prompt>\n"
        f"<student_essay>\n{essay_text}\n</student_essay>\n"
        f"<evaluation_rationales>\n{payload}\n</evaluation_rationales>"
    )


def _tokenize_score_input(tokenizer: Any, text: str, max_length: int) -> Any:
    """Apply the frozen R0 tokenizer contract used during training/evaluation."""
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )


@dataclass(frozen=True)
class R0RuntimePaths:
    root: Path
    served_model_name: str
    score_base: Path
    score_adapters: tuple[Path, Path, Path, Path]
    score_heads: tuple[Path, Path, Path, Path]
    score_max_length: int
    rationale_base: Path
    blind_adapter: Path
    final_adapter: Path
    final_prompt_kind: str
    final_prompt_path: Path | None
    final_prompt_sha256: str | None
    rationale_max_model_len: int
    rationale_gpu_memory_utilization: float
    blind_max_tokens: int
    final_max_tokens: int
    seed: int

    @classmethod
    def load(cls, root: Path) -> "R0RuntimePaths":
        root = root.resolve()
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SubmissionContractError("R0 runtime manifest is unreadable") from exc
        _need(manifest.get("schema_version") == "mal2026-submission-runtime-v1", "R0 runtime schema differs")
        pipeline_kind = manifest.get("pipeline_kind")
        _need(pipeline_kind in {
            "legacy_r0_prediction_ensemble_to_dpo_rationale",
            "legacy_r0_prediction_ensemble_to_latest_score_blind_rationale",
        }, "R0 runtime pipeline differs")
        score = manifest.get("score")
        rationale = manifest.get("rationale")
        _need(isinstance(score, dict) and isinstance(rationale, dict), "R0 runtime sections differ")
        adapters = score.get("adapter_paths")
        heads = score.get("head_paths")
        _need(isinstance(adapters, list) and len(adapters) == 4, "R0 adapter list differs")
        _need(isinstance(heads, list) and len(heads) == 4, "R0 head list differs")
        value = cls(
            root=root,
            served_model_name=str(manifest.get("served_model_name", "")),
            score_base=_safe_path(root, str(score.get("base_path", "")), "R0 score base"),
            score_adapters=tuple(_safe_path(root, str(path), "R0 score adapter") for path in adapters),  # type: ignore[arg-type]
            score_heads=tuple(_safe_path(root, str(path), "R0 score head", directory=False) for path in heads),  # type: ignore[arg-type]
            score_max_length=int(score.get("max_length", 0)),
            rationale_base=_safe_path(root, str(rationale.get("base_path", "")), "R0 rationale base"),
            blind_adapter=_safe_path(root, str(rationale.get("blind_adapter_path", "")), "R0 blind adapter"),
            final_adapter=_safe_path(root, str(rationale.get("final_adapter_path", "")), "R0 final adapter"),
            final_prompt_kind=str(rationale.get("final_prompt_kind", "legacy_score_conditioned_v1")),
            final_prompt_path=(
                _safe_path(root, str(rationale.get("final_prompt_path", "")), "R0 final prompt", directory=False)
                if rationale.get("final_prompt_path") else None
            ),
            final_prompt_sha256=(
                str(rationale.get("final_prompt_sha256")) if rationale.get("final_prompt_sha256") else None
            ),
            rationale_max_model_len=int(rationale.get("max_model_len", 0)),
            rationale_gpu_memory_utilization=float(rationale.get("gpu_memory_utilization", 0.0)),
            blind_max_tokens=int(rationale.get("blind_max_tokens", 0)),
            final_max_tokens=int(rationale.get("final_max_tokens", 0)),
            seed=int(manifest.get("seed", 42)),
        )
        _need(bool(value.served_model_name), "R0 served model name is blank")
        _need(value.score_max_length == 2048, "R0 score context differs")
        _need(value.rationale_max_model_len >= 4096, "R0 rationale context is too short")
        _need(0.3 <= value.rationale_gpu_memory_utilization <= 0.55, "R0 rationale memory fraction differs")
        _need(value.blind_max_tokens == 512 and value.final_max_tokens == 512, "R0 token budget differs")
        _need(value.final_prompt_kind in {
            "legacy_score_conditioned_v1", "latest_score_blind_v1",
        }, "R0 final prompt kind differs")
        if value.final_prompt_kind == "latest_score_blind_v1":
            _need(value.final_prompt_path is not None, "latest rationale prompt is unavailable")
            _need(
                isinstance(value.final_prompt_sha256, str)
                and sha256(value.final_prompt_path.read_bytes()).hexdigest() == value.final_prompt_sha256,
                "latest rationale prompt checksum differs",
            )
        return value


class R0EnsemblePipeline:
    def __init__(self, paths: R0RuntimePaths) -> None:
        # Avoid a runtime FlashInfer/ninja JIT dependency in the offline image.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer
        from vllm import LLM

        _need(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "submission requires at least one visible GPU")
        self.paths = paths
        self.served_model_name = paths.served_model_name
        self.rationale = LLM(
            model=str(paths.rationale_base), dtype="bfloat16", seed=paths.seed,
            gpu_memory_utilization=paths.rationale_gpu_memory_utilization,
            enforce_eager=True, enable_lora=True, max_lora_rank=32,
            max_model_len=paths.rationale_max_model_len, trust_remote_code=False,
        )
        self.rationale_tokenizer = AutoTokenizer.from_pretrained(
            paths.rationale_base, local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        self.score_tokenizer = AutoTokenizer.from_pretrained(
            paths.score_base, local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        if self.score_tokenizer.pad_token is None:
            _need(self.score_tokenizer.eos_token is not None, "R0 tokenizer has no pad token")
            self.score_tokenizer.pad_token = self.score_tokenizer.eos_token
        base = AutoModel.from_pretrained(
            paths.score_base, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        )
        base.config.use_cache = False
        adapter_names = tuple(f"epoch_{index:02d}" for index in range(1, 5))
        score_backbone = PeftModel.from_pretrained(
            base, paths.score_adapters[0], adapter_name=adapter_names[0], is_trainable=False,
        )
        for name, path in zip(adapter_names[1:], paths.score_adapters[1:], strict=True):
            score_backbone.load_adapter(path, adapter_name=name, is_trainable=False)
        self.score_backbone = score_backbone.cuda().eval()
        hidden = int(self.score_backbone.config.hidden_size)
        _need(hidden == 4096, "R0 hidden size differs")
        heads = []
        for path in paths.score_heads:
            head = nn.Linear(hidden, 3, dtype=torch.float32).cuda().eval()
            state = load_file(str(path), device="cuda")
            _need(set(state) == {"weight", "bias"}, "R0 score head tensors differ")
            head.load_state_dict(state, strict=True)
            heads.append(head)
        self.score_heads = tuple(heads)
        self.adapter_names = adapter_names
        self._torch = torch
        self._functional = functional

    @classmethod
    def from_environment(cls) -> "R0EnsemblePipeline":
        root = Path(os.environ.get("MAL2026_BUNDLE_ROOT", "/opt/mal2026/models"))
        return cls(R0RuntimePaths.load(root))

    def _generate(
        self, messages: list[dict[str, str]], *, adapter_name: str, adapter_id: int,
        adapter_path: Path, max_tokens: int, schema: dict[str, Any], seed: int,
        stop: str | list[str] | None,
    ) -> tuple[str, int, int]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        from vllm.sampling_params import StructuredOutputsParams

        stops = [stop] if isinstance(stop, str) else stop
        sampling = SamplingParams(
            temperature=0.0, top_p=1.0, seed=seed, max_tokens=max_tokens,
            stop=stops, skip_special_tokens=True,
            structured_outputs=StructuredOutputsParams(json=schema, disable_additional_properties=True),
        )
        outputs = self.rationale.chat(
            [messages], sampling_params=sampling,
            lora_request=LoRARequest(adapter_name, adapter_id, str(adapter_path)), use_tqdm=False,
        )
        _need(len(outputs) == 1 and len(outputs[0].outputs) == 1, "R0 rationale generation count differs")
        item = outputs[0].outputs[0]
        _need(item.finish_reason == "stop", "R0 rationale generation did not stop cleanly")
        return item.text, len(outputs[0].prompt_token_ids or []), len(item.token_ids or [])

    def _score(self, prompt_text: str, essay_text: str, rationales: Mapping[str, str]) -> tuple[dict[str, int], int]:
        text = _score_input(prompt_text, essay_text, rationales)
        # Reproduce the frozen R0 training/evaluation tokenizer contract.  In
        # particular, long evaluator inputs are right-truncated to 2,048
        # tokens rather than turning into an HTTP 500 at serving time.
        encoded = _tokenize_score_input(self.score_tokenizer, text, self.paths.score_max_length)
        token_count = int(encoded["input_ids"].shape[1])
        _need(token_count <= self.paths.score_max_length, "R0 score input exceeds audited context")
        encoded = {key: value.cuda() for key, value in encoded.items()}
        torch = self._torch
        predictions = []
        with torch.inference_mode():
            for name, head in zip(self.adapter_names, self.score_heads, strict=True):
                self.score_backbone.set_adapter(name)
                hidden = self.score_backbone(**encoded, return_dict=True).last_hidden_state
                mask = encoded["attention_mask"]
                positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
                final = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
                _need(bool((final >= 0).all().item()), "R0 score input has no token")
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), final]
                logits = head(self._functional.normalize(pooled, p=2, dim=-1).float())
                predictions.append(logits)
            continuous = torch.stack(predictions, dim=0).mean(dim=0).clamp(1.0, 5.0)
            integer = torch.floor(continuous + 0.5).clamp(1, 5).to(torch.int64)[0].tolist()
        _need(len(integer) == 3 and all(type(value) is int and 1 <= value <= 5 for value in integer), "R0 score output differs")
        return {axis: integer[index] for index, axis in enumerate(AXES)}, token_count * 4

    def complete(
        self, messages: Sequence[Mapping[str, Any]], *, max_tokens: int,
        temperature: float, top_p: float, seed: int | None,
        stop: str | list[str] | None,
    ) -> Completion:
        task = extract_prompt_essay(messages)
        if task is None:
            text = "MAL2026 한국어 글쓰기 채점 R0 ensemble 모델입니다."
            return Completion(content=text, completion_tokens=len(self.rationale_tokenizer.encode(text, add_special_tokens=False)))
        _need(temperature == 0.0 and top_p == 1.0, "submission task requires deterministic sampling")
        prompt_text, essay_text = task
        fixed_seed = self.paths.seed if seed is None else seed
        blind_text, blind_prompt_tokens, blind_completion_tokens = self._generate(
            _legacy_messages(prompt_text, essay_text), adapter_name="r0_blind", adapter_id=1,
            adapter_path=self.paths.blind_adapter, max_tokens=self.paths.blind_max_tokens,
            schema=_legacy_schema(), seed=fixed_seed, stop=stop,
        )
        blind = _parse_legacy_rationales(blind_text)
        scores, score_tokens = self._score(prompt_text, essay_text, blind)
        if self.paths.final_prompt_kind == "latest_score_blind_v1":
            _need(self.paths.final_prompt_path is not None, "latest rationale prompt is unavailable")
            final_messages = _latest_score_blind_messages(
                self.paths.final_prompt_path.read_text(encoding="utf-8"), prompt_text, essay_text,
            )
            final_adapter_name = "latest_score_blind_sft"
        else:
            final_messages = _final_messages(prompt_text, essay_text, scores)
            final_adapter_name = "final_dpo"
        final_text, final_prompt_tokens, _ = self._generate(
            final_messages, adapter_name=final_adapter_name, adapter_id=2,
            adapter_path=self.paths.final_adapter, max_tokens=self.paths.final_max_tokens,
            schema=_rationale_schema(1, 384), seed=fixed_seed, stop=stop,
        )
        from .contracts import parse_rationales

        final_rationales = parse_rationales(final_text)
        content = compact_participant_json(participant_output(scores, final_rationales))
        completion_tokens = len(self.rationale_tokenizer.encode(content, add_special_tokens=False))
        _need(completion_tokens <= max_tokens, "R0 final participant JSON exceeds request max_tokens")
        _need(math.isfinite(float(completion_tokens)), "R0 completion token count differs")
        return Completion(
            content=content,
            prompt_tokens=blind_prompt_tokens + blind_completion_tokens + score_tokens + final_prompt_tokens,
            completion_tokens=completion_tokens,
        )
