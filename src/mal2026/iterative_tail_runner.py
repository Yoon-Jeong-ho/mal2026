"""Execution core for the fixed 20-round iterative tail program.

All model selection in this module is train-OOF-only.  Validation is never
opened.  Fold workers emit row-level predictions only beneath the ignored
restricted root; public output contains aggregate metrics and checksums.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mal2026.iterative_tail_features import (
    ScoreBlindFeatureBundle,
    load_score_blind_feature_cache,
    load_score_blind_features,
    write_score_blind_feature_cache,
)
from mal2026.iterative_tail_metrics import (
    AXES,
    compute_iterative_tail_metrics,
    promotion_decision,
)
from mal2026.iterative_tail_models import CandidateSpec, fit_predict
from mal2026.iterative_tail_protocol import IterativeTailProtocol, load_bound_training_rows, load_protocol


RUN_ID = "iterative-tail-refinement-v1-20260801-001"
PUBLIC_ROOT = Path("outputs/iterative-tail-refinement-v1") / RUN_ID
RESTRICTED_ROOT = Path("data/processed/restricted/iterative_tail_refinement_v1") / RUN_ID
CACHE_PATH = RESTRICTED_ROOT / "score_blind_features.npz"
CACHE_MANIFEST_PATH = RESTRICTED_ROOT / "score_blind_features.manifest.json"
BASE_SEED = 2026080101


class IterativeTailRunError(RuntimeError):
    """Raised when a fold, artifact, or round lifecycle contract differs."""


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class ExperimentData:
    source_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    prompt_nums: tuple[str, ...]
    embeddings: np.ndarray
    base: np.ndarray
    targets: np.ndarray
    folds: np.ndarray
    evidence: ScoreBlindFeatureBundle


@dataclass(frozen=True)
class CandidateVariant:
    round_number: int
    variant_id: str
    family: str
    feature_view: str = "none"
    embedding_view: str = "frozen"
    target_source: str = "gold"
    hidden_dim: int = 128
    epochs: int = 80
    learning_rate: float = 1e-3
    ridge_alpha: float = 1.0
    huber_delta: float = 1.0
    ordinal_weight: float = 0.5
    auxiliary_weight: float = 0.25
    contrastive_weight: float = 0.25
    tail_weighting_mode: str = "both"
    tail_weighting_strength: float = 1.0
    uncertainty_coverage: float = 1.0
    max_correction: float = 0.75

    def model_spec(self, *, fold: int, device: str) -> CandidateSpec:
        return CandidateSpec(
            family=self.family, seed=BASE_SEED + fold, device=device,
            hidden_dim=self.hidden_dim, epochs=self.epochs,
            learning_rate=self.learning_rate, ridge_alpha=self.ridge_alpha,
            huber_delta=self.huber_delta, ordinal_weight=self.ordinal_weight,
            auxiliary_weight=self.auxiliary_weight,
            contrastive_weight=self.contrastive_weight,
            tail_weighting_mode=self.tail_weighting_mode,
            tail_weighting_strength=self.tail_weighting_strength,
            uncertainty_coverage=self.uncertainty_coverage,
            max_correction=self.max_correction,
        )


def variants_for_round(round_number: int) -> tuple[CandidateVariant, ...]:
    """Return the predeclared subvariants for one discovery round."""
    if round_number == 1:
        return ()
    if round_number == 2:
        return tuple(CandidateVariant(2, f"alpha-{alpha:g}", "ridge_residual", ridge_alpha=alpha) for alpha in (0.01, 0.1, 1.0, 10.0, 100.0))
    if round_number == 3:
        return tuple(CandidateVariant(3, f"hidden-{hidden}", "mlp_residual", hidden_dim=hidden, epochs=100) for hidden in (256, 128))
    if round_number == 4:
        return (CandidateVariant(4, "coral", "coral", epochs=100),)
    if round_number == 5:
        return (CandidateVariant(5, "huber-ordinal", "joint_huber_ordinal", epochs=100),)
    if round_number == 6:
        return tuple(CandidateVariant(6, f"coverage-{coverage:g}", "uncertainty_gated", epochs=100, uncertainty_coverage=coverage) for coverage in (0.25, 0.5, 0.75, 1.0))
    if round_number == 7:
        return tuple(CandidateVariant(7, f"low-{weight:g}", "tail_effective_number", epochs=100, tail_weighting_mode="low", tail_weighting_strength=weight) for weight in (1.5, 2.0, 3.0))
    if round_number == 8:
        return tuple(CandidateVariant(8, f"high-{weight:g}", "tail_effective_number", epochs=100, tail_weighting_mode="high", tail_weighting_strength=weight) for weight in (1.5, 2.0, 3.0))
    if round_number == 9:
        return (CandidateVariant(9, "equal-band", "equal_band_replay", epochs=100),)
    if round_number == 10:
        return (CandidateVariant(10, "prompt-equal", "joint_huber_ordinal", epochs=100),)
    if round_number == 11:
        return (CandidateVariant(11, "3v4", "auxiliary_3v4", epochs=100, auxiliary_weight=0.5),)
    if round_number == 12:
        return (CandidateVariant(12, "four-threshold", "threshold_calibration"),)
    if round_number == 13:
        return (CandidateVariant(13, "adjacent", "adjacent_contrastive", epochs=100, contrastive_weight=0.25),)
    if round_number == 14:
        return (CandidateVariant(14, "content-v2", "joint_huber_ordinal", feature_view="content_structured", epochs=100),)
    if round_number == 15:
        return (CandidateVariant(15, "org-expression-v2", "joint_huber_ordinal", feature_view="org_expression_structured", epochs=100),)
    if round_number == 16:
        return (CandidateVariant(16, "four-agent-consensus", "joint_huber_ordinal", feature_view="consensus_disagreement", epochs=100),)
    if round_number == 17:
        return (CandidateVariant(17, "consensus-distill", "ridge_residual", feature_view="none", embedding_view="evidence_hash", target_source="round16", ridge_alpha=10.0),)
    if round_number == 18:
        return (CandidateVariant(18, "full-fusion", "joint_huber_ordinal", feature_view="full_fusion", epochs=100),)
    return ()


def prepare_score_blind_cache(protocol: IterativeTailProtocol | None = None) -> Mapping[str, object]:
    protocol = protocol or load_protocol()
    rows = load_bound_training_rows(protocol)
    identifiers = [row.source_id for row in rows]
    if CACHE_PATH.is_file() and CACHE_MANIFEST_PATH.is_file():
        load_score_blind_feature_cache(identifiers, CACHE_PATH, CACHE_MANIFEST_PATH)
        return json.loads(CACHE_MANIFEST_PATH.read_text(encoding="utf-8"))
    bundle = load_score_blind_features(identifiers)
    return write_score_blind_feature_cache(bundle, CACHE_PATH, CACHE_MANIFEST_PATH)


def load_experiment_data(protocol: IterativeTailProtocol | None = None) -> ExperimentData:
    protocol = protocol or load_protocol()
    rows = load_bound_training_rows(protocol)
    identifiers = tuple(row.source_id for row in rows)
    evidence = load_score_blind_feature_cache(identifiers, CACHE_PATH, CACHE_MANIFEST_PATH)
    metadata: dict[str, tuple[str, str]] = {}
    canonical = Path(protocol.bindings["canonical_train_path"])
    with canonical.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            source_id = raw.get("id")
            document_id = raw.get("document_id")
            prompt_num = raw.get("prompt_num")
            if not all(isinstance(value, str) and value for value in (source_id, document_id, prompt_num)):
                raise IterativeTailRunError("canonical train metadata schema differs")
            if source_id in metadata:
                raise IterativeTailRunError("canonical train ID duplicates")
            metadata[source_id] = (document_id, prompt_num)
    if set(metadata) != set(identifiers):
        raise IterativeTailRunError("canonical metadata and embedding IDs differ")
    return ExperimentData(
        source_ids=identifiers,
        document_ids=tuple(metadata[source_id][0] for source_id in identifiers),
        prompt_nums=tuple(metadata[source_id][1] for source_id in identifiers),
        embeddings=np.asarray([row.shared_embedding for row in rows], dtype=np.float32),
        base=np.asarray([row.base_predictions for row in rows], dtype=np.float32),
        targets=np.asarray([row.raw_labels for row in rows], dtype=np.float32),
        folds=np.asarray([row.oof_fold for row in rows], dtype=np.int64),
        evidence=evidence,
    )


def _selected_prediction_path(round_number: int) -> Path:
    return RESTRICTED_ROOT / f"round-{round_number:02d}" / "selected_oof_predictions.jsonl"


def _read_prediction_matrix(path: Path, data: ExperimentData) -> np.ndarray:
    predictions: dict[str, tuple[float, float, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if set(raw) != {"source_id", "fold", "prediction"}:
                raise IterativeTailRunError(f"prediction row schema differs: {path}")
            values = raw["prediction"]
            if not isinstance(values, list) or len(values) != 3:
                raise IterativeTailRunError("prediction axes differ")
            predictions[raw["source_id"]] = tuple(float(value) for value in values)
    if set(predictions) != set(data.source_ids):
        raise IterativeTailRunError(f"prediction population differs: {path}")
    result = np.asarray([predictions[source_id] for source_id in data.source_ids], dtype=np.float32)
    if not np.isfinite(result).all() or np.any(result < 1) or np.any(result > 5):
        raise IterativeTailRunError("prediction values differ")
    return result


def _threshold_predict(
    train_base: np.ndarray, train_targets: np.ndarray, predict_base: np.ndarray, *, device: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Fit four ordered cumulative thresholds independently for each axis."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(BASE_SEED)
    torch_device = torch.device(device)
    base = torch.as_tensor(train_base, dtype=torch.float32, device=torch_device)
    targets = torch.as_tensor(train_targets, dtype=torch.float32, device=torch_device)
    infer = torch.as_tensor(predict_base, dtype=torch.float32, device=torch_device)
    output = []
    initial_hashes, final_hashes = [], []
    for axis in range(3):
        raw_steps = torch.nn.Parameter(torch.full((4,), -0.2, device=torch_device))
        center = torch.nn.Parameter(torch.tensor(3.0, device=torch_device))
        log_scale = torch.nn.Parameter(torch.tensor(-1.5, device=torch_device))
        parameters = [raw_steps, center, log_scale]
        initial_hashes.append(sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in parameters)).hexdigest())
        optimizer = torch.optim.Adam(parameters, lr=0.02)
        for _ in range(300):
            steps = F.softplus(raw_steps) + 0.05
            cuts = torch.cumsum(steps, dim=0)
            cuts = center + cuts - cuts.mean()
            scale = F.softplus(log_scale) + 0.05
            prediction = 1.0 + torch.sigmoid((base[:, axis, None] - cuts[None, :]) / scale).sum(dim=1)
            loss = F.mse_loss(prediction, targets[:, axis])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            steps = F.softplus(raw_steps) + 0.05
            cuts = torch.cumsum(steps, dim=0)
            cuts = center + cuts - cuts.mean()
            scale = F.softplus(log_scale) + 0.05
            output.append(1.0 + torch.sigmoid((infer[:, axis, None] - cuts[None, :]) / scale).sum(dim=1))
        final_hashes.append(sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in parameters)).hexdigest())
    return torch.stack(output, dim=1).clamp(1, 5).cpu().numpy(), tuple(initial_hashes), tuple(final_hashes)


def _prompt_equal_weights(prompts: Sequence[str]) -> np.ndarray:
    counts = Counter(prompts)
    return np.asarray([len(prompts) / (len(counts) * counts[value]) for value in prompts], dtype=np.float32)


def fold_result_paths(variant: CandidateVariant, fold: int) -> tuple[Path, Path]:
    root = RESTRICTED_ROOT / f"round-{variant.round_number:02d}" / variant.variant_id / f"fold-{fold}"
    public = PUBLIC_ROOT / f"round-{variant.round_number:02d}" / variant.variant_id / f"fold-{fold}"
    return root / "predictions.jsonl", public / "result.json"


def run_candidate_fold(
    data: ExperimentData, variant: CandidateVariant, fold: int, *, device: str = "cuda:0",
    smoke: bool = False,
) -> Mapping[str, Any]:
    if fold not in range(5):
        raise IterativeTailRunError("fold must be 0..4")
    train_indices = np.flatnonzero(data.folds != fold)
    predict_indices = np.flatnonzero(data.folds == fold)
    if smoke:
        train_indices = train_indices[:128]
        predict_indices = predict_indices[:32]
    if len(predict_indices) == 0:
        raise IterativeTailRunError("held-out fold is empty")
    extra = data.evidence.view(variant.feature_view)
    train_extra = None if extra is None else extra[train_indices]
    predict_extra = None if extra is None else extra[predict_indices]
    embeddings = data.embeddings
    targets = data.targets
    if variant.embedding_view == "evidence_hash":
        evidence_hash = data.evidence.view("evidence_hash")
        assert evidence_hash is not None
        embeddings = evidence_hash
    if variant.target_source == "round16":
        targets = _read_prediction_matrix(_selected_prediction_path(16), data)
    sample_weights = None
    if variant.round_number == 10:
        sample_weights = _prompt_equal_weights([data.prompt_nums[index] for index in train_indices])
    if variant.family == "threshold_calibration":
        prediction, initial_hashes, final_hashes = _threshold_predict(
            data.base[train_indices], data.targets[train_indices], data.base[predict_indices], device=device,
        )
        family = variant.family
        seed = BASE_SEED
    else:
        result = fit_predict(
            variant.model_spec(fold=fold, device=device),
            embeddings[train_indices], data.base[train_indices], targets[train_indices],
            embeddings[predict_indices], data.base[predict_indices],
            train_extra_features=train_extra, predict_extra_features=predict_extra,
            train_sample_weights=sample_weights,
        )
        prediction = result.predictions
        initial_hashes, final_hashes = result.initial_state_hashes, result.final_state_hashes
        family, seed = result.family, result.seed
    metrics = compute_iterative_tail_metrics(data.targets[predict_indices], prediction)
    if smoke:
        output = PUBLIC_ROOT / "smoke" / f"round-{variant.round_number:02d}-{variant.variant_id}-fold-{fold}.json"
        payload = {
            "schema_version": "mal2026-iterative-tail-smoke-v1", "status": "completed",
            "round": variant.round_number, "variant": variant.variant_id, "fold": fold,
            "records": len(predict_indices), "family": family,
            "macro_rmse": metrics["macro"]["rmse"], "device": device,
            "initial_state_hashes": list(initial_hashes), "final_state_hashes": list(final_hashes),
            "validation_loaded": False, "average_target_used": False,
        }
        _atomic_json(output, payload)
        return payload
    prediction_path, result_path = fold_result_paths(variant, fold)
    _jsonl(prediction_path, (
        {"source_id": data.source_ids[index], "fold": fold, "prediction": [float(value) for value in row]}
        for index, row in zip(predict_indices, prediction, strict=True)
    ))
    payload = {
        "schema_version": "mal2026-iterative-tail-fold-v1", "status": "completed",
        "round": variant.round_number, "variant": variant.variant_id, "fold": fold,
        "records": len(predict_indices), "family": family, "seed": seed,
        "variant_spec": asdict(variant),
        "macro_rmse": metrics["macro"]["rmse"],
        "prediction_path": str(prediction_path), "prediction_sha256": _file_sha256(prediction_path),
        "initial_state_hashes": list(initial_hashes), "final_state_hashes": list(final_hashes),
        "validation_loaded": False, "average_target_used": False,
    }
    _atomic_json(result_path, payload)
    return payload


def _merge_variant(data: ExperimentData, variant: CandidateVariant) -> tuple[np.ndarray, Path]:
    by_id: dict[str, tuple[int, tuple[float, float, float]]] = {}
    for fold in range(5):
        prediction_path, result_path = fold_result_paths(variant, fold)
        if not result_path.is_file():
            raise IterativeTailRunError(f"missing fold result: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("prediction_sha256") != _file_sha256(prediction_path):
            raise IterativeTailRunError("fold result checksum differs")
        with prediction_path.open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                source_id = raw["source_id"]
                if source_id in by_id or raw["fold"] != fold:
                    raise IterativeTailRunError("duplicate or mismatched fold prediction")
                by_id[source_id] = (fold, tuple(float(value) for value in raw["prediction"]))
    if set(by_id) != set(data.source_ids):
        raise IterativeTailRunError("variant OOF coverage differs")
    matrix = np.asarray([by_id[source_id][1] for source_id in data.source_ids], dtype=np.float32)
    merged = RESTRICTED_ROOT / f"round-{variant.round_number:02d}" / variant.variant_id / "oof_predictions.jsonl"
    _jsonl(merged, (
        {"source_id": source_id, "fold": int(data.folds[index]), "prediction": [float(value) for value in matrix[index]]}
        for index, source_id in enumerate(data.source_ids)
    ))
    return matrix, merged


def aggregate_candidate_round(data: ExperimentData, round_number: int) -> Mapping[str, Any]:
    variants = variants_for_round(round_number)
    if not variants:
        raise IterativeTailRunError("candidate round has no variants")
    candidates = []
    matrices: dict[str, np.ndarray] = {}
    merged_paths: dict[str, Path] = {}
    for variant in variants:
        matrix, merged = _merge_variant(data, variant)
        metrics = compute_iterative_tail_metrics(data.targets, matrix)
        candidates.append({
            "variant": variant.variant_id, "family": variant.family,
            "macro_rmse": metrics["macro"]["rmse"],
            "macro_spearman": metrics["macro"]["spearman"],
            "equal_group_rmse": metrics["macro"]["equal_group_rmse"],
            "prediction_sha256": _file_sha256(merged),
        })
        matrices[variant.variant_id] = matrix
        merged_paths[variant.variant_id] = merged
        _atomic_json(PUBLIC_ROOT / f"round-{round_number:02d}" / variant.variant_id / "metrics.json", metrics)
    winner = min(candidates, key=lambda item: (item["macro_rmse"], item["variant"]))
    selected = matrices[str(winner["variant"])]
    selected_path = _selected_prediction_path(round_number)
    _jsonl(selected_path, (
        {"source_id": source_id, "fold": int(data.folds[index]), "prediction": [float(value) for value in selected[index]]}
        for index, source_id in enumerate(data.source_ids)
    ))
    selected_metrics = compute_iterative_tail_metrics(data.targets, selected)
    payload = {
        "schema_version": "mal2026-iterative-tail-round-v1", "status": "completed",
        "round": round_number, "selected_variant": winner["variant"],
        "selection": "minimum train-only five-fold OOF macro continuous RMSE within predeclared variants",
        "variants": candidates, "selected_metrics": selected_metrics,
        "selected_prediction_sha256": _file_sha256(selected_path),
        "validation_loaded": False, "average_target_used": False,
    }
    _atomic_json(PUBLIC_ROOT / f"round-{round_number:02d}" / "aggregate.json", payload)
    return payload


def initialize_baseline_round(data: ExperimentData) -> Mapping[str, Any]:
    selected_path = _selected_prediction_path(1)
    _jsonl(selected_path, (
        {"source_id": source_id, "fold": int(data.folds[index]), "prediction": [float(value) for value in data.base[index]]}
        for index, source_id in enumerate(data.source_ids)
    ))
    metrics = compute_iterative_tail_metrics(data.targets, data.base)
    # Frozen embeddings store float JSON values but the GPU runner uses
    # float32 matrices; the resulting sub-nanounit drift is purely casting.
    if abs(float(metrics["macro"]["rmse"]) - 0.5687802162500409) > 1e-8:
        raise IterativeTailRunError("exact baseline RMSE reproduction failed")
    payload = {
        "schema_version": "mal2026-iterative-tail-round-v1", "status": "completed", "round": 1,
        "selected_variant": "exact-r0-oof", "selected_metrics": metrics,
        "selected_prediction_sha256": _file_sha256(selected_path),
        "validation_loaded": False, "average_target_used": False,
    }
    _atomic_json(PUBLIC_ROOT / "round-01" / "aggregate.json", payload)
    return payload


def _ensemble_weights(targets: np.ndarray, matrices: Sequence[np.ndarray]) -> tuple[float, ...]:
    if len(matrices) == 1:
        return (1.0,)
    best_weights: tuple[float, ...] | None = None
    best_rmse = math.inf
    if len(matrices) == 2:
        grid = ((step / 20.0, 1.0 - step / 20.0) for step in range(21))
    else:
        grid = (
            (left / 20.0, middle / 20.0, 1.0 - (left + middle) / 20.0)
            for left in range(21) for middle in range(21 - left)
        )
    for weights in grid:
        prediction = sum(weight * matrix for weight, matrix in zip(weights, matrices, strict=True))
        rmse = float(compute_iterative_tail_metrics(targets, prediction)["macro"]["rmse"])
        if rmse < best_rmse - 1e-12:
            best_rmse, best_weights = rmse, tuple(float(value) for value in weights)
    assert best_weights is not None
    return best_weights


def run_round19_ensemble(data: ExperimentData, eligible_rounds: Sequence[int]) -> Mapping[str, Any]:
    candidates = []
    for round_number in eligible_rounds:
        aggregate_path = PUBLIC_ROOT / f"round-{round_number:02d}" / "aggregate.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        candidates.append((
            float(aggregate["selected_metrics"]["macro"]["rmse"]),
            round_number,
            _read_prediction_matrix(_selected_prediction_path(round_number), data),
        ))
    candidates.sort(key=lambda item: (item[0], item[1]))
    # One member per broad structural family, maximum three.  Round numbers are
    # fixed representatives of linear, neural/ordinal, and evidence families.
    buckets = ((2,), tuple(range(3, 14)), tuple(range(14, 19)))
    selected = []
    for bucket in buckets:
        options = [item for item in candidates if item[1] in bucket]
        if options:
            selected.append(options[0])
    if not selected:
        selected = [(float(compute_iterative_tail_metrics(data.targets, data.base)["macro"]["rmse"]), 1, data.base)]
    matrices = [item[2] for item in selected]
    weights = _ensemble_weights(data.targets, matrices)
    prediction = np.clip(sum(weight * matrix for weight, matrix in zip(weights, matrices, strict=True)), 1, 5)
    selected_path = _selected_prediction_path(19)
    _jsonl(selected_path, (
        {"source_id": source_id, "fold": int(data.folds[index]), "prediction": [float(value) for value in prediction[index]]}
        for index, source_id in enumerate(data.source_ids)
    ))
    metrics = compute_iterative_tail_metrics(data.targets, prediction)
    payload = {
        "schema_version": "mal2026-iterative-tail-round-v1", "status": "completed", "round": 19,
        "selected_variant": "nonnegative-diverse-ensemble",
        "members": [item[1] for item in selected], "weights": list(weights),
        "selected_metrics": metrics, "selected_prediction_sha256": _file_sha256(selected_path),
        "validation_loaded": False, "average_target_used": False,
    }
    _atomic_json(PUBLIC_ROOT / "round-19" / "aggregate.json", payload)
    return payload


def _fit_bounded_affine(train_pred: np.ndarray, train_target: np.ndarray, test_pred: np.ndarray) -> np.ndarray:
    result = np.empty_like(test_pred)
    for axis in range(3):
        best = (math.inf, 1.0, 0.0)
        for scale in np.linspace(0.8, 1.2, 41):
            for offset in np.linspace(-0.30, 0.30, 61):
                candidate = np.clip(scale * train_pred[:, axis] + offset, 1, 5)
                mse = float(np.mean(np.square(candidate - train_target[:, axis])))
                if mse < best[0] - 1e-15:
                    best = (mse, float(scale), float(offset))
        result[:, axis] = np.clip(best[1] * test_pred[:, axis] + best[2], 1, 5)
    return result


def _paired_bootstrap_macro_rmse(
    targets: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
    *, n_resamples: int = 10000, seed: int = BASE_SEED,
) -> Mapping[str, Any]:
    """Vectorized paired row bootstrap for the fixed final macro-RMSE gate."""
    baseline_squared = np.square(targets.astype(np.float64) - baseline.astype(np.float64))
    candidate_squared = np.square(targets.astype(np.float64) - candidate.astype(np.float64))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    batch_size = 200
    for start in range(0, n_resamples, batch_size):
        stop = min(n_resamples, start + batch_size)
        indices = rng.integers(0, len(targets), size=(stop - start, len(targets)))
        base_rmse = np.sqrt(baseline_squared[indices].mean(axis=1)).mean(axis=1)
        candidate_rmse = np.sqrt(candidate_squared[indices].mean(axis=1)).mean(axis=1)
        draws[start:stop] = base_rmse - candidate_rmse
    improvement = float(compute_iterative_tail_metrics(targets, baseline)["macro"]["rmse"] - compute_iterative_tail_metrics(targets, candidate)["macro"]["rmse"])
    lower, upper = (float(value) for value in np.quantile(draws, (0.025, 0.975)))
    return {
        "unit": "row (document IDs are unique in canonical train)",
        "cluster_count": len(targets), "n_resamples": n_resamples,
        "confidence": 0.95, "seed": seed,
        "delta_direction": "positive_means_candidate_improvement",
        "intervals": {"rmse": {"estimate": improvement, "lower": lower, "upper": upper, "valid_resamples": n_resamples}},
    }


def run_round20_calibration(data: ExperimentData, source_round: int) -> Mapping[str, Any]:
    source = _read_prediction_matrix(_selected_prediction_path(source_round), data)
    prediction = np.empty_like(source)
    for fold in range(5):
        train = data.folds != fold
        test = data.folds == fold
        prediction[test] = _fit_bounded_affine(source[train], data.targets[train], source[test])
    selected_path = _selected_prediction_path(20)
    _jsonl(selected_path, (
        {"source_id": source_id, "fold": int(data.folds[index]), "prediction": [float(value) for value in prediction[index]]}
        for index, source_id in enumerate(data.source_ids)
    ))
    metrics = compute_iterative_tail_metrics(data.targets, prediction)
    payload = {
        "schema_version": "mal2026-iterative-tail-round-v1", "status": "completed", "round": 20,
        "selected_variant": "five-fold-bounded-affine", "source_round": source_round,
        "selected_metrics": metrics, "selected_prediction_sha256": _file_sha256(selected_path),
        "calibration_bounds": {"scale": [0.8, 1.2], "offset": [-0.30, 0.30], "score": [1, 5]},
        "validation_loaded": False, "average_target_used": False, "frozen": True,
    }
    _atomic_json(PUBLIC_ROOT / "round-20" / "aggregate.json", payload)
    return payload


def apply_sequential_promotion(
    data: ExperimentData, *, through_round: int = 20, final_bootstrap: bool = True,
) -> Mapping[str, Any]:
    incumbent_round = 1
    baseline_metrics = json.loads((PUBLIC_ROOT / "round-01" / "aggregate.json").read_text(encoding="utf-8"))["selected_metrics"]
    incumbent_metrics = baseline_metrics
    decisions = []
    promoted_rounds = [1]
    for round_number in range(2, through_round + 1):
        aggregate = json.loads((PUBLIC_ROOT / f"round-{round_number:02d}" / "aggregate.json").read_text(encoding="utf-8"))
        metrics = aggregate["selected_metrics"]
        decision = promotion_decision(incumbent_metrics, metrics)
        decisions.append({"round": round_number, "incumbent_before": incumbent_round, **decision})
        if decision["promote"]:
            incumbent_round = round_number
            incumbent_metrics = metrics
            promoted_rounds.append(round_number)
    incumbent = _read_prediction_matrix(_selected_prediction_path(incumbent_round), data)
    baseline_gain = float(baseline_metrics["macro"]["rmse"] - incumbent_metrics["macro"]["rmse"])
    bootstrap = None
    candidate_minus_baseline = None
    final_pass = False
    if final_bootstrap:
        if len(set(data.document_ids)) != len(data.document_ids):
            raise IterativeTailRunError("vectorized final bootstrap requires the verified unique-document population")
        bootstrap = _paired_bootstrap_macro_rmse(data.targets, data.base, incumbent)
        interval = bootstrap["intervals"]["rmse"]
        candidate_minus_baseline = {
            "estimate": None if interval["estimate"] is None else -float(interval["estimate"]),
            "lower": None if interval["upper"] is None else -float(interval["upper"]),
            "upper": None if interval["lower"] is None else -float(interval["lower"]),
        }
        final_pass = baseline_gain >= 0.01 and candidate_minus_baseline["upper"] is not None and candidate_minus_baseline["upper"] < 0
    payload = {
        "schema_version": "mal2026-iterative-tail-promotion-v1", "status": "completed",
        "baseline_round": 1, "incumbent_round": incumbent_round,
        "promoted_rounds": promoted_rounds, "decisions": decisions,
        "baseline_macro_rmse": baseline_metrics["macro"]["rmse"],
        "incumbent_macro_rmse": incumbent_metrics["macro"]["rmse"],
        "macro_rmse_improvement": baseline_gain,
        "paired_bootstrap_improvement_ci": bootstrap,
        "paired_bootstrap_candidate_minus_baseline_rmse_ci": candidate_minus_baseline,
        "final_gate_pass": final_pass,
        "final_bootstrap_executed": final_bootstrap,
        "validation_loaded": False, "average_target_used": False,
    }
    _atomic_json(PUBLIC_ROOT / "promotion_summary.json", payload)
    return payload


def public_progress() -> Mapping[str, Any]:
    completed = []
    for round_number in range(1, 21):
        path = PUBLIC_ROOT / f"round-{round_number:02d}" / "aggregate.json"
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            completed.append({
                "round": round_number,
                "macro_rmse": raw["selected_metrics"]["macro"]["rmse"],
                "selected_variant": raw.get("selected_variant"),
            })
    return {"run_id": RUN_ID, "completed_rounds": len(completed), "planned_rounds": 20, "rounds": completed}


__all__ = [
    "BASE_SEED", "CACHE_MANIFEST_PATH", "CACHE_PATH", "PUBLIC_ROOT", "RESTRICTED_ROOT",
    "CandidateVariant", "ExperimentData", "IterativeTailRunError", "aggregate_candidate_round",
    "apply_sequential_promotion", "initialize_baseline_round", "load_experiment_data",
    "prepare_score_blind_cache", "public_progress", "run_candidate_fold", "run_round19_ensemble",
    "run_round20_calibration", "variants_for_round",
]
