"""Fail-closed exact-OOF KURE CORAL training with label-density smoothing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .iterative_tail_metrics import AXES, compute_iterative_tail_metrics, metric_improvements
from .kure_axis_contrastive import token_length_audit
from .kure_ordinal_oof import (
    KUREOrdinalOOFConfig, _build_axis_model, derived_seed, load_exact_r0,
    load_raw_axis_gold, seed_runtime, validate_backbone_without_validation,
)
from .kure_phase1_direct_oof import (
    _TextRow, _assert_private_file, _atomic_private_jsonl as _direct_atomic_private_jsonl,
    _atomic_public_json as _direct_atomic_public_json,
    _environment, _git_sha, _secure_directory,
    prediction_band_diagnostics, scheduler_state_conflict, summarize_gpu_telemetry,
    verify_post_prediction_gold_dependencies,
)
from .official_score_matrix import file_sha256, official_half_up
from .ordinal_tail_fixed_feature import CandidateSpec, coral_pmf
from .stage3_coral_promotion import promotion_gate

SCHEMA_VERSION = "mal2026-kure-lds-oof-v1"
METHOD = "coral-lds-gaussian-s025-cap4"
CONFIG_FILE_PATH = Path(__file__).resolve().parents[2] / "configs/kure_lds_oof.v1.json"


class KURELDSOOFError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise KURELDSOOFError(message)


def set_process_title(stage: str) -> str:
    """Expose the active LDS stage/fold/axis in process listings."""
    need(stage and len(stage) <= 96 and all(char.isalnum() or char in "._:-" for char in stage),
         "process-title stage differs")
    import setproctitle
    title=f"mal2026:lds:{stage}"
    setproctitle.setproctitle(title)
    need(setproctitle.getproctitle()==title,"setproctitle did not preserve the requested title")
    return title


def _lexists(path:Path)->bool:
    return path.exists() or path.is_symlink()


def _atomic_private_jsonl(path:Path,rows:Any)->str:
    need(not _lexists(path),f"refusing to overwrite {path}")
    return _direct_atomic_private_jsonl(path,rows)


def _atomic_public_json(path:Path,value:Mapping[str,Any])->str:
    need(not _lexists(path),f"refusing to overwrite {path}")
    return _direct_atomic_public_json(path,value)


def _forbidden(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any("validation" in str(k).lower() or _forbidden(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_forbidden(v) for v in value)
    return isinstance(value, str) and "validation" in value.lower()


@dataclass(frozen=True)
class FitLabelBinding:
    outer_fold: int
    axis: str
    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FitLabelBinding":
        need(set(value) == set(cls.__dataclass_fields__), "fit-label binding fields differ")
        result = cls(int(value["outer_fold"]), str(value["axis"]), str(value["path"]), str(value["sha256"]))
        need(result.outer_fold in range(5) and result.axis in AXES, "fit-label fold/axis differs")
        return result


@dataclass(frozen=True)
class KURELDSOOFConfig:
    schema_version: str; status: str; execution_authorized: bool; steward_preparation_authorized: bool
    steward_task_card_sha256: str; steward_task_card_commit: str
    task_card_path: str; task_card_sha256: str; task_card_commit: str
    preparer_path: str; preparer_sha256: str; preparer_commit: str
    preparation_request_config_sha256: str; steward_authorized_config_sha256: str
    run_id: str
    direct_aggregate_path: str; direct_aggregate_sha256: str; direct_task_card_sha256: str
    direct_config_file_sha256: str; direct_report_config_sha256: str
    source_stage3_config_path: str; source_stage3_config_sha256: str; source_stage3_report_config_sha256: str
    source_stage3_aggregate_sha256: str
    train_path: str; train_sha256: str; fold_manifest_path: str; fold_manifest_sha256: str
    fold_rows_path: str; fold_rows_sha256: str; r0_oof_prediction_path: str; r0_oof_prediction_sha256: str
    label_free_projection_path: str; label_free_projection_sha256: str
    label_free_manifest_path: str; label_free_manifest_sha256: str
    fit_label_manifest_path: str; fit_label_manifest_sha256: str
    fit_label_bindings: tuple[FitLabelBinding, ...]
    output_root: str; restricted_output_root: str
    seed: int; epochs: int; learning_rate: float; weight_decay: float; batch_size: int
    gradient_accumulation_steps: int; max_length: int; raw_rmse_auxiliary_weight: float
    grid_start: float; grid_stop: float; grid_step: float; lds_sigma: float; lds_truncation: float
    weight_floor: float; weight_cap: float; mean_weight_tolerance: float
    smoke_gpu: int; full_gpu_scope: tuple[int, ...]; fold_gpu_mapping: Mapping[str, int]
    telemetry_columns: tuple[str, ...]; smoke_minimum_samples: int; full_minimum_samples: int
    telemetry_interval_seconds: int; axes: tuple[str, ...]; average_target_forbidden: bool
    integration_recovery_policy: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "KURELDSOOFConfig":
        need(isinstance(raw, Mapping) and not _forbidden(raw), "validation fields and paths are forbidden")
        value = dict(raw)
        for key in ("full_gpu_scope", "telemetry_columns", "axes"): value[key] = tuple(value.get(key, ()))
        need(isinstance(value.get("fit_label_bindings"), list), "fit-label bindings must be a list")
        value["fit_label_bindings"] = tuple(FitLabelBinding.from_mapping(x) for x in value["fit_label_bindings"])
        need(set(value) == set(cls.__dataclass_fields__), "LDS config fields differ")
        result = cls(**value); result.validate(require_dependencies=False); return result

    @classmethod
    def from_json(cls, path: str | Path, *, require_dependencies: bool = False) -> "KURELDSOOFConfig":
        try: raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc: raise KURELDSOOFError("LDS config is unreadable") from exc
        result = cls.from_mapping(raw); result.validate(require_dependencies=require_dependencies); return result

    def validate(self, *, require_dependencies: bool) -> None:
        need(self.schema_version == SCHEMA_VERSION and self.run_id == "kure-coral-lds-oof-v1-20260803-001" and self.status in {
            "pending_direct_failure_and_scientific_authorization", "authorized_for_steward_preparation", "authorized"},
             "LDS schema/status differs")
        need(self.execution_authorized is (self.status == "authorized"), "authorization fields disagree")
        need(self.steward_preparation_authorized is (self.status in {"authorized_for_steward_preparation", "authorized"}),
             "steward authorization fields disagree")
        placeholders = (self.steward_task_card_sha256,self.steward_task_card_commit,
                        self.task_card_sha256, self.task_card_commit, self.preparer_sha256, self.preparer_commit,
                        self.preparation_request_config_sha256,self.steward_authorized_config_sha256,self.direct_aggregate_sha256,
                        self.direct_task_card_sha256, self.direct_config_file_sha256, self.direct_report_config_sha256,
                        self.label_free_projection_sha256, self.label_free_manifest_sha256,
                        self.fit_label_manifest_sha256, *(x.sha256 for x in self.fit_label_bindings))
        preparation_digests=(self.steward_task_card_sha256,self.preparer_sha256,self.preparation_request_config_sha256,
                             self.direct_aggregate_sha256,self.direct_task_card_sha256,self.direct_config_file_sha256,
                             self.direct_report_config_sha256,self.label_free_projection_sha256,self.label_free_manifest_sha256)
        fit_digests=(self.steward_authorized_config_sha256,self.fit_label_manifest_sha256,*(x.sha256 for x in self.fit_label_bindings))
        if self.status == "authorized":
            need(all(len(x)==64 for x in (*preparation_digests,*fit_digests)), "authorized digest bindings are incomplete")
            need(len(self.steward_task_card_commit)==len(self.task_card_commit)==len(self.preparer_commit)==40
                 and len(self.task_card_sha256)==64,"authorized commit/card binding differs")
        elif self.status == "authorized_for_steward_preparation":
            need(all(len(x)==64 for x in preparation_digests)
                 and self.task_card_sha256==self.steward_task_card_sha256
                 and self.task_card_commit==self.steward_task_card_commit
                 and len(self.steward_task_card_commit)==len(self.preparer_commit)==40,
                 "steward-authorized lineage bindings are incomplete")
            need(all(x=="" for x in fit_digests), "steward stage must not pre-bind generated fit-label artifacts")
        else:
            need(all(x == "" for x in placeholders), "pending config must retain blank lineage placeholders")
        need(tuple((x.outer_fold,x.axis) for x in self.fit_label_bindings) == tuple((f,a) for f in range(5) for a in AXES), "ordered 5-fold x 3-axis fit-label bindings required")
        need(self.axes == AXES and self.average_target_forbidden is True, "axis/average contract differs")
        need(self.integration_recovery_policy == "identical_hash_missing_fold_resume_only_preserve_partial_immutable_no_complete_rerun_no_scientific_retune", "integration-recovery policy differs")
        need((self.seed, self.epochs, self.learning_rate, self.weight_decay, self.batch_size,
              self.gradient_accumulation_steps, self.max_length, self.raw_rmse_auxiliary_weight)
             == (2026080302, 6, 5e-5, .01, 20, 2, 1536, .25), "Stage3 phase1 schedule differs")
        need((self.grid_start, self.grid_stop, self.grid_step, self.lds_sigma, self.lds_truncation,
              self.weight_floor, self.weight_cap, self.mean_weight_tolerance)
             == (1., 5., .05, .25, .5, .25, 4., 1e-6), "frozen LDS mathematics differs")
        need(self.smoke_gpu == 0 and self.full_gpu_scope == (0,1,2,3)
             and self.fold_gpu_mapping == {"0":0,"1":1,"2":2,"3":3,"4":0}, "GPU mapping differs")
        need(self.telemetry_columns == ("timestamp","index","uuid","name","memory.total","driver_version","utilization.gpu","memory.used")
             and (self.smoke_minimum_samples,self.full_minimum_samples,self.telemetry_interval_seconds)==(1,2,30),
             "telemetry contract differs")
        need(Path(self.output_root).resolve() != Path(self.restricted_output_root).resolve(), "public/restricted roots differ")
        if require_dependencies: self.validate_safe_dependencies()

    def require_execution_authorization(self, *, preflight_all_fit: bool = False) -> None:
        need(self.status == "authorized" and self.execution_authorized, "LDS scientific execution is not authorized")
        self.validate_safe_dependencies(preflight_all_fit=preflight_all_fit)

    def require_steward_authorization(self) -> None:
        need(self.status == "authorized_for_steward_preparation" and self.steward_preparation_authorized
             and not self.execution_authorized, "LDS data-steward preparation is not authorized")
        self.validate_preparation_dependencies()

    def validate_preparation_dependencies(self) -> None:
        need(self.steward_preparation_authorized, "steward dependencies require recorded authorization")
        for path,digest,label,private in (
            (self.preparer_path,self.preparer_sha256,"preparer",False),
            (self.source_stage3_config_path,self.source_stage3_config_sha256,"Stage3 config",False),
            (self.direct_aggregate_path,self.direct_aggregate_sha256,"direct aggregate",False),
            (self.label_free_projection_path,self.label_free_projection_sha256,"label-free projection",True),
            (self.label_free_manifest_path,self.label_free_manifest_sha256,"label-free manifest",True),
        ): _verify(path,digest,label,private=private)
        for commit,path,digest in ((self.preparer_commit,self.preparer_path,self.preparer_sha256),):
            shown=subprocess.run(["git","show",f"{commit}:{path}"],capture_output=True,check=False)
            need(shown.returncode==0 and sha256(shown.stdout).hexdigest()==digest,"commit/file binding differs")
        steward_card=subprocess.run(["git","show",f"{self.steward_task_card_commit}:{self.task_card_path}"],capture_output=True,check=False)
        need(steward_card.returncode==0 and sha256(steward_card.stdout).hexdigest()==self.steward_task_card_sha256,
             "steward authorization card commit binding differs")
        _validate_authorization_text(steward_card.stdout.decode("utf-8"),self.run_id,"steward_preparation")
        if self.status == "authorized_for_steward_preparation":
            _verify(self.task_card_path,self.task_card_sha256,"current steward task card",private=False)
        else:
            _verify(self.task_card_path,self.task_card_sha256,"current scientific task card",private=False)
            final_card=subprocess.run(["git","show",f"{self.task_card_commit}:{self.task_card_path}"],capture_output=True,check=False)
            need(final_card.returncode==0 and sha256(final_card.stdout).hexdigest()==self.task_card_sha256,
                 "scientific authorization card commit binding differs")
        request=subprocess.run(["git","show",f"{self.preparer_commit}:configs/kure_lds_oof.v1.json"],capture_output=True,check=False)
        need(request.returncode==0 and sha256(request.stdout).hexdigest()==self.preparation_request_config_sha256,
             "preparation request binding differs")
        request_raw=json.loads(request.stdout.decode("utf-8"))
        need(request_raw.get("status")=="pending_direct_failure_and_scientific_authorization"
             and request_raw.get("execution_authorized") is False
             and request_raw.get("steward_preparation_authorized") is False,
             "preparation request must be the immutable pending-stage config")
        _validate_direct_failure(self); source=_source(self); validate_backbone_without_validation(source.backbone)
        _load_projection(self)

    def validate_safe_dependencies(self, *, preflight_all_fit: bool = False) -> None:
        need(self.execution_authorized, "safe LDS dependencies require authorization")
        self.validate_preparation_dependencies()
        _validate_recorded_authorization(self,"scientific_execution")
        _verify(self.fit_label_manifest_path,self.fit_label_manifest_sha256,"fit-label manifest",private=True)
        if preflight_all_fit:
            for item in self.fit_label_bindings: _verify(item.path,item.sha256,"fit-label artifact",private=True)
        _validate_fit_manifest(self)


def config_sha256(config: KURELDSOOFConfig) -> str:
    return sha256(json.dumps(asdict(config),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def _config_file_sha256() -> str: return file_sha256(CONFIG_FILE_PATH)


def _verify(path: str, digest: str, label: str, *, private: bool) -> None:
    value=Path(path); need(value.is_file() and not value.is_symlink() and file_sha256(value)==digest,f"{label} binding differs")
    if private: _assert_private_file(value)


def _validate_recorded_authorization(config: KURELDSOOFConfig, scope: str) -> None:
    _validate_authorization_text(Path(config.task_card_path).read_text(encoding="utf-8"),config.run_id,scope)


def _validate_authorization_text(text:str,run_id:str,scope:str)->None:
    need(scope in {"steward_preparation","scientific_execution"},"authorization scope differs")
    prefix={"steward_preparation":"- Steward preparation user statement: ",
            "scientific_execution":"- Scientific execution user statement: "}[scope]
    lines=[line for line in text.splitlines() if line.startswith(prefix)]
    need(len(lines)==1 and run_id in lines[0] and "PENDING_EXPLICIT_USER_AUTHORIZATION" not in lines[0]
         and len(lines[0])>len(prefix)+len(run_id),f"recorded explicit {scope} authorization is absent")


def _source(config: KURELDSOOFConfig) -> KUREOrdinalOOFConfig:
    source=KUREOrdinalOOFConfig.from_json(config.source_stage3_config_path,require_dependencies=False)
    from .kure_ordinal_oof import config_sha256 as source_hash
    need(source_hash(source)==config.source_stage3_report_config_sha256,"Stage3 report config differs")
    need((source.train_path,source.train_sha256,source.fold_manifest_path,source.fold_manifest_sha256,
          source.fold_rows_path,source.fold_rows_sha256,source.r0_oof_prediction_path,source.r0_oof_prediction_sha256,
          source.seed,source.epochs,source.learning_rate,source.weight_decay,source.batch_size,
          source.gradient_accumulation_steps,source.max_length,source.raw_rmse_auxiliary_weight)
         == (config.train_path,config.train_sha256,config.fold_manifest_path,config.fold_manifest_sha256,
             config.fold_rows_path,config.fold_rows_sha256,config.r0_oof_prediction_path,config.r0_oof_prediction_sha256,
             config.seed,config.epochs,config.learning_rate,config.weight_decay,config.batch_size,
             config.gradient_accumulation_steps,config.max_length,config.raw_rmse_auxiliary_weight),"Stage3 phase1 lineage differs")
    need((source.backbone.model_id,source.backbone.model_revision,source.backbone.lora_r,
          source.backbone.lora_alpha,source.backbone.lora_dropout,source.backbone.warmstart_artifact_sha256)
         == ("nlpai-lab/KURE-v1","d14c8a9423946e268a0c9952fecf3a7aabd73bd9",16,32,.05,
             "ffdc985d56c655c03e8964927b127b24f0c5bb7fdde8d89e944941f5419cf25a"),"KURE/AIHub/LoRA lineage differs")
    return source


def _validate_direct_failure(config: KURELDSOOFConfig) -> None:
    report=json.loads(Path(config.direct_aggregate_path).read_text(encoding="utf-8"))
    gate=report.get("common_stage3_promotion_gate")
    need(report.get("schema_version")=="mal2026-kure-phase1-direct-oof-aggregate-v1"
         and report.get("status")=="completed" and report.get("mode")=="full_oof"
         and report.get("run_id")=="kure-phase1-direct-oof-v1-20260803-001"
         and report.get("records")==2000 and report.get("folds")==5
         and report.get("method")=="coral-natural-phase1-direct" and report.get("source_method")=="coral-natural"
         and report.get("training_performed") is False and report.get("calibration_performed") is False
         and report.get("selection_performed") is False and report.get("automatic_stage6_deployment_eligible") is False
         and report.get("protected_output")=="exact_r0"
         and report.get("config_sha256")==config.direct_report_config_sha256
         and report.get("config_file_sha256")==config.direct_config_file_sha256
         and report.get("r0_oof_prediction_sha256")==config.r0_oof_prediction_sha256
         and report.get("task_card_sha256")==config.direct_task_card_sha256
         and report.get("source_stage3_config_sha256")==config.source_stage3_config_sha256
         and report.get("source_stage3_report_config_sha256")==config.source_stage3_report_config_sha256
         and report.get("source_stage3_aggregate_sha256")==config.source_stage3_aggregate_sha256
         and report.get("fold_manifest_sha256")==config.fold_manifest_sha256
         and report.get("fold_rows_sha256")==config.fold_rows_sha256
         and report.get("common_stage3_promotion_gate_passed") is False
         and isinstance(gate,Mapping) and gate.get("eligible") is False
         and report.get("validation_rows_loaded") is False and report.get("average_target_used") is False,
         "completed exact direct failure entry condition is not proven")


def _load_projection(config: KURELDSOOFConfig):
    manifest=json.loads(Path(config.label_free_manifest_path).read_text(encoding="utf-8"))
    need(manifest.get("schema_version")=="mal2026-kure-phase1-direct-input-manifest-v1"
         and manifest.get("status")=="completed" and manifest.get("records")==2000
         and manifest.get("fold_counts")=={str(f):400 for f in range(5)}
         and manifest.get("projection_path")==config.label_free_projection_path
         and manifest.get("projection_sha256")==config.label_free_projection_sha256
         and manifest.get("projection_schema")==["id","document_id","prompt_num","prompt","essay","outer_fold"]
         and manifest.get("labels_present") is False and manifest.get("average_present") is False
         and manifest.get("gold_present") is False and manifest.get("source_train_sha256")==config.train_sha256,
         "label-free projection manifest differs")
    by_fold={f:[] for f in range(5)}; folds={}
    for line in Path(config.label_free_projection_path).read_text(encoding="utf-8").splitlines():
        row=json.loads(line); need(set(row)=={"id","document_id","prompt_num","prompt","essay","outer_fold"},"projection schema differs")
        identifier=row["id"]; fold=row["outer_fold"]
        need(isinstance(identifier,str) and identifier not in folds and fold in range(5),"projection identity differs")
        by_fold[fold].append(_TextRow(identifier,str(row["document_id"]),str(row["prompt_num"]),row["prompt"],row["essay"])); folds[identifier]=fold
    need(len(folds)==2000 and all(len(v)==400 for v in by_fold.values()),"projection coverage differs")
    return {k:tuple(v) for k,v in by_fold.items()},folds


def _validate_fit_manifest(config: KURELDSOOFConfig) -> None:
    manifest=json.loads(Path(config.fit_label_manifest_path).read_text(encoding="utf-8"))
    need(manifest.get("schema_version")=="mal2026-kure-lds-fit-label-manifest-v1" and manifest.get("status")=="completed"
         and manifest.get("records_per_outer_fit")==1600 and manifest.get("axes")==list(AXES)
         and manifest.get("average_indexed") is False and manifest.get("text_present") is False
         and manifest.get("held_ids_present") is False
         and manifest.get("source_train_sha256")==config.train_sha256
         and manifest.get("label_free_projection_sha256")==config.label_free_projection_sha256
         and manifest.get("label_free_manifest_sha256")==config.label_free_manifest_sha256
         and manifest.get("steward_task_card_sha256")==config.steward_task_card_sha256
         and manifest.get("steward_task_card_commit")==config.steward_task_card_commit
         and manifest.get("direct_aggregate_sha256")==config.direct_aggregate_sha256
         and manifest.get("direct_task_card_sha256")==config.direct_task_card_sha256
         and manifest.get("direct_config_file_sha256")==config.direct_config_file_sha256
         and manifest.get("direct_report_config_sha256")==config.direct_report_config_sha256
         and manifest.get("generator_sha256")==config.preparer_sha256
         and manifest.get("generator_git_sha")==config.preparer_commit
         and isinstance(manifest.get("execution_git_sha"),str) and len(manifest["execution_git_sha"])==40
         and manifest.get("preparation_request_config_sha256")==config.preparation_request_config_sha256,
         "fit-label manifest contract differs")
    need(manifest.get("steward_authorized_config_sha256")==config.steward_authorized_config_sha256,
         "fit-label manifest contract differs")
    need(manifest.get("bindings")==[asdict(x)|{"records":1600} for x in config.fit_label_bindings],
         "fit-label manifest binding inventory differs")


def _load_fit_labels(config: KURELDSOOFConfig, projection: Mapping[int,Sequence[_TextRow]], folds: Mapping[str,int],
                     outer_fold: int, axes: Sequence[str]):
    _validate_fit_manifest(config)
    result={}
    selected=[x for x in config.fit_label_bindings if x.outer_fold==outer_fold and x.axis in axes]
    need([(x.outer_fold,x.axis) for x in selected]==[(outer_fold,a) for a in axes],"requested fit-label inventory differs")
    for binding in selected:
        _verify(binding.path,binding.sha256,"fit-label artifact",private=True)
        held={r.identifier for r in projection[binding.outer_fold]}; labels={}
        with Path(binding.path).open(encoding="utf-8") as stream:
            for line in stream:
                row=json.loads(line); need(set(row)=={"source_id","raw_score"},"axis fit-label schema differs")
                identifier=row["source_id"]; need(identifier not in held and folds.get(identifier)!=binding.outer_fold and identifier not in labels,
                                             "held membership entered fit labels")
                raw=float(row["raw_score"]); assert_grid_alignment([raw]); labels[identifier]=raw
        need(len(labels)==1600 and set(labels)==set(folds)-held,"fit-label coverage/exclusion differs")
        result[(binding.outer_fold,binding.axis)]=labels
    return result


def score_grid() -> np.ndarray: return np.linspace(1.,5.,81,dtype=np.float64)


def assert_grid_alignment(values: Sequence[float]) -> np.ndarray:
    result=[]
    for value in values:
        need(math.isfinite(float(value)),"raw score is not finite")
        scaled=Decimal(str(value))*Decimal(20)
        integral=scaled.to_integral_value()
        need(scaled==integral and Decimal(20)<=integral<=Decimal(100),"raw score is not aligned to the .05 grid")
        result.append(int(integral)-20)
    return np.asarray(result,dtype=np.int64)


def gaussian_kernel() -> np.ndarray:
    offsets=np.arange(-10,11,dtype=np.float64)*.05; kernel=np.exp(-.5*(offsets/.25)**2); kernel/=kernel.sum()
    need(len(kernel)==21 and abs(float(kernel.sum())-1.)<1e-15 and np.all(kernel>0),"LDS Gaussian kernel differs")
    return kernel


def smoothed_density(values: Sequence[float]) -> tuple[np.ndarray,np.ndarray]:
    indices=assert_grid_alignment(values); counts=np.bincount(indices,minlength=81).astype(np.float64)
    density=np.convolve(counts,gaussian_kernel(),mode="full")[10:91]
    need(len(density)==81 and np.all(np.isfinite(density)) and np.all(density>=0),"LDS density differs")
    return density,indices


def solve_clipped_mean_one(unscaled: Sequence[float], *, floor: float=.25, cap: float=4., tolerance: float=1e-6) -> tuple[np.ndarray,float]:
    u=np.asarray(unscaled,dtype=np.float64)
    need(u.ndim==1 and len(u)>0 and np.all(np.isfinite(u)) and np.all(u>0),"unscaled weights must be finite positive")
    need(0<floor<=1<=cap and tolerance>0,"weight bounds are infeasible")
    def weights(c: float): return np.clip(c*u,floor,cap)
    lo,hi=0.,1.; bracketed=False
    for _ in range(65):
        if float(weights(hi).mean())>=1.: bracketed=True; break
        hi*=2.
    need(bracketed and math.isfinite(hi),"weight scale bracketing failed within 64 doublings")
    for _ in range(128):
        mid=(lo+hi)/2
        if float(weights(mid).mean())<1.: lo=mid
        else: hi=mid
    c=hi; w=weights(c)
    need(abs(float(w.mean())-1.)<=tolerance and np.all((w>=floor)&(w<=cap)),"bounded mean-one LDS solve failed")
    return w,c


def lds_example_weights(values: Sequence[float]) -> tuple[np.ndarray,Mapping[str,Any]]:
    density,indices=smoothed_density(values); unscaled=1./np.maximum(density[indices],1e-12)
    weights,scale=solve_clipped_mean_one(unscaled)
    return weights,{"kernel_sum":float(gaussian_kernel().sum()),"observed_density_min":float(density[indices].min()),
                    "density_max":float(density.max()),"scale":scale,"weight_min":float(weights.min()),
                    "weight_max":float(weights.max()),"weight_mean":float(weights.mean())}


def weighted_hybrid_loss(logits: Any, rounded: Any, raw: Any, weights: Any, *, floor: float=.25, cap: float=4.) -> Any:
    import torch
    import torch.nn.functional as F
    need(isinstance(logits,torch.Tensor) and isinstance(rounded,torch.Tensor)
         and isinstance(raw,torch.Tensor) and isinstance(weights,torch.Tensor),"weighted loss inputs must be tensors")
    need(logits.ndim==2 and logits.shape[1]==4 and logits.is_floating_point(),"CORAL logits must be finite floating [N,4]")
    n=logits.shape[0]
    need(rounded.ndim==raw.ndim==weights.ndim==1 and rounded.shape==raw.shape==weights.shape==(n,),
         "rounded/raw/weight inputs must be exact [N] vectors")
    need(raw.is_floating_point() and weights.is_floating_point(),"raw labels and LDS weights must be floating tensors")
    need(logits.device==rounded.device==raw.device==weights.device,"weighted loss inputs must share one device")
    need(floor==.25 and cap==4.,"frozen LDS weight bounds differ")
    need(bool(torch.isfinite(logits).all()) and bool(torch.isfinite(raw).all()) and bool(torch.isfinite(weights).all()),
         "weighted loss inputs must be finite")
    rounded_float=rounded.to(dtype=torch.float64)
    need(bool(((rounded_float>=1)&(rounded_float<=5)&(rounded_float==rounded_float.round())).all()),
         "rounded classes must be integral 1..5")
    raw_float=raw.to(dtype=torch.float64); scaled=raw_float*20.
    need(bool(((raw_float>=1)&(raw_float<=5)).all())
         and bool(torch.allclose(scaled,scaled.round(),atol=1e-5,rtol=0)),"raw labels must be .05-grid values in 1..5")
    need(bool(((weights>0)&(weights>=floor)&(weights<=cap)).all()),"LDS weights must be positive and bounded")
    targets=(rounded.view(-1,1)>torch.arange(1,5,device=logits.device).view(1,-1)).to(logits.dtype)
    ordinal=F.binary_cross_entropy_with_logits(logits,targets,reduction="none").mean(1)
    pmf=coral_pmf(logits); expected=(pmf*torch.arange(1,6,device=logits.device,dtype=logits.dtype)).sum(1)
    per_example=ordinal+.25*(expected-raw.to(logits.dtype)).square()
    loss=(weights.to(logits.dtype)*per_example).mean(); need(bool(torch.isfinite(loss)),"weighted hybrid loss is non-finite")
    return loss


class _LDSDataset:
    def __init__(self, rows: Sequence[_TextRow], labels: Mapping[str,float], tokenizer: Any, max_length: int):
        self.ids=[r.identifier for r in rows]; raw=[labels[x] for x in self.ids]
        self.raw_labels=raw; self.labels=[official_half_up(x) for x in raw]
        self.weights,self.lds=lds_example_weights(raw)
        from .kure_axis_contrastive import render_input
        self.encoded=tokenizer([render_input(r) for r in rows],truncation=True,max_length=max_length)
    def __len__(self): return len(self.ids)
    def __getitem__(self,index):
        return {"input_ids":self.encoded["input_ids"][index],"attention_mask":self.encoded["attention_mask"][index],
                "labels":self.labels[index],"raw_labels":self.raw_labels[index],"lds_weights":float(self.weights[index])}


def _train(model: Any,dataset: _LDSDataset,tokenizer: Any,source: KUREOrdinalOOFConfig,output: Path,*,seed:int,max_steps:int=-1):
    from transformers import DataCollatorWithPadding,Trainer,TrainingArguments
    need(not _lexists(output),f"refusing to overwrite {output}"); _secure_directory(output)
    class LDSTrainer(Trainer):
        lora_gradient_observed=False; head_gradient_observed=False
        def compute_loss(self,current_model,inputs,return_outputs=False,**_):
            labels=inputs.pop("labels"); raw=inputs.pop("raw_labels"); weights=inputs.pop("lds_weights")
            result=current_model(**inputs); loss=weighted_hybrid_loss(result["logits"],labels,raw,weights)
            return (loss,result) if return_outputs else loss
        def training_step(self,current_model,inputs,num_items_in_batch=None):
            loss=super().training_step(current_model,inputs,num_items_in_batch)
            torch=__import__("torch")
            named=[(n,p.grad) for n,p in current_model.named_parameters() if p.grad is not None]
            lora=[g for n,g in named if "lora_" in n]; head=[g for n,g in named if n.startswith(("score.","cut_"))]
            if lora and all(torch.isfinite(g).all() for g in lora) and any(float(g.abs().sum())>0 for g in lora): self.lora_gradient_observed=True
            if head and all(torch.isfinite(g).all() for g in head) and any(float(g.abs().sum())>0 for g in head): self.head_gradient_observed=True
            return loss
    args=TrainingArguments(output_dir=str(output),num_train_epochs=source.epochs,max_steps=max_steps,
        per_device_train_batch_size=source.batch_size,gradient_accumulation_steps=source.gradient_accumulation_steps,
        learning_rate=source.learning_rate,weight_decay=source.weight_decay,save_strategy="no",report_to=[],
        seed=seed,data_seed=seed,remove_unused_columns=False)
    trainer=LDSTrainer(model=model,args=args,train_dataset=dataset,data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train(); return trainer


def _predict(model:Any,rows:Sequence[_TextRow],tokenizer:Any,max_length:int,batch_size:int)->np.ndarray:
    import torch
    values=[]; model.eval().to(torch.device("cuda"))
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            from .kure_axis_contrastive import render_input
            encoded=tokenizer([render_input(r) for r in rows[start:start+batch_size]],padding=True,truncation=True,
                              max_length=max_length,return_tensors="pt").to("cuda")
            logits=model(**encoded)["logits"]; pmf=coral_pmf(logits)
            pred=(pmf*torch.arange(1,6,device=logits.device,dtype=pmf.dtype)).sum(1).clamp(1,5)
            values.extend(pred.cpu().float().tolist())
    result=np.asarray(values,dtype=np.float64); need(len(result)==len(rows) and np.all(np.isfinite(result)),"prediction differs")
    return result


def _validate_trainable_state(state:Mapping[str,Any])->None:
    import torch
    coral={"score.weight":(1,1024),"score.bias":(1,),"cut_base":(),"cut_gaps":(3,)}
    need(set(k for k in state if k.startswith(("score.","cut_")))==set(coral),"CORAL trainable inventory differs")
    need(all(tuple(state[k].shape)==shape for k,shape in coral.items()),"CORAL trainable tensor shape differs")
    lora={k:v for k,v in state.items() if "lora_" in k}
    need(len(state)==292 and len(lora)==288 and all(k.endswith(("lora_A.default.weight","lora_B.default.weight")) for k in lora),
         "exact Stage3 LoRA trainable inventory differs")
    shapes=Counter(tuple(v.shape) for v in lora.values())
    need(shapes==Counter({(16,1024):120,(1024,16):120,(4096,16):24,(16,4096):24}),"Stage3 LoRA tensor shapes differ")
    need(all(isinstance(v,torch.Tensor) and v.is_floating_point() and bool(torch.isfinite(v).all()) for v in state.values()),
         "trainable checkpoint contains a non-finite/non-floating tensor")


def _authenticate_checkpoint(path:Path,expected_sha256:str)->Mapping[str,Any]:
    from safetensors import safe_open
    _assert_private_file(path); need(file_sha256(path)==expected_sha256,"checkpoint hash differs")
    with safe_open(path,framework="pt",device="cpu") as handle: state={k:handle.get_tensor(k) for k in handle.keys()}
    _validate_trainable_state(state); return {"tensor_count":len(state),"lora_tensor_count":288,"coral_tensor_count":4}


def _save_state(path:Path,model:Any)->str:
    from safetensors.torch import save
    need(not _lexists(path),f"refusing to overwrite {path}"); _secure_directory(path.parent)
    state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items() if "lora_" in k or k.startswith(("score.","cut_"))}
    _validate_trainable_state(state); payload=save(state)
    tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp"); descriptor=None; temporary_created=False
    try:
        descriptor=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o660)
        temporary_created=True
        view=memoryview(payload)
        while view:
            written=os.write(descriptor,view); need(written>0,"checkpoint temporary write failed"); view=view[written:]
        os.fsync(descriptor); os.close(descriptor); descriptor=None
        need(tmp.is_file() and not tmp.is_symlink(),"checkpoint temporary is not ordinary")
        os.link(tmp,path,follow_symlinks=False); os.chmod(path,0o660); tmp.unlink()
        directory=os.open(path.parent,os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    except BaseException:
        if descriptor is not None: os.close(descriptor)
        if temporary_created: tmp.unlink(missing_ok=True)
        raise
    digest=file_sha256(path); _authenticate_checkpoint(path,digest); return digest


def _smoke_subset(rows:Sequence[_TextRow], labels:Mapping[str,float])->list[_TextRow]:
    buckets={k:[] for k in range(1,6)}
    for row in rows: buckets[official_half_up(labels[row.identifier])].append(row)
    need(all(len(v)>=2 for v in buckets.values()),"smoke fit lacks five rounded classes")
    # Projection order plus class order is deterministic and independent of score magnitude within a class.
    return [row for score in range(1,6) for row in buckets[score][:2]]


def run(config:KURELDSOOFConfig|str|Path,*,outer_fold:int,validate_only:bool=False,smoke:bool=False)->Mapping[str,Any]:
    value=KURELDSOOFConfig.from_json(config) if isinstance(config,(str,Path)) else config
    need(outer_fold in range(5) and (not smoke or outer_fold==0),"outer/smoke fold differs")
    if validate_only:
        set_process_title("validate")
        return {"status":"validated","execution_authorized":value.execution_authorized,"gpu_used":False,"validation_rows_loaded":False,"average_target_used":False}
    value.require_execution_authorization()
    set_process_title(f"{'smoke' if smoke else 'oof'}:f{outer_fold}:load")
    import torch
    from transformers import AutoTokenizer
    need(torch.cuda.is_available(),"LDS training requires explicit GPU launch")
    if smoke: need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0" and torch.cuda.device_count()==1,"smoke requires physical GPU0 only")
    source=_source(value); projection,folds=_load_projection(value)
    held=list(projection[outer_fold]); fit=[r for fold,rows in projection.items() if fold!=outer_fold for r in rows]
    axes=("content",) if smoke else AXES
    all_labels=_load_fit_labels(value,projection,folds,outer_fold,axes)
    content_labels=all_labels[(outer_fold,"content")]; need({r.identifier for r in fit}==set(content_labels),"fit text/label identity differs")
    if smoke: fit=_smoke_subset(fit,content_labels); held=held[:8]
    tokenizer=AutoTokenizer.from_pretrained(source.backbone.model_path,revision=source.backbone.model_revision,
                                             local_files_only=True,trust_remote_code=False,use_fast=True)
    predictions=[]; disclosures=[]
    restricted=Path(value.restricted_output_root)/("smoke/outer-00" if smoke else f"outer-{outer_fold:02d}")
    for axis in axes:
        set_process_title(f"{'smoke' if smoke else 'oof'}:f{outer_fold}:{axis}")
        labels=all_labels[(outer_fold,axis)]; need({r.identifier for r in fit} <= set(labels),"axis fit text/label identity differs")
        seed=derived_seed(value.seed,outer_fold,"coral-natural",axis,"phase1"); seed_runtime(seed)
        train_data=_LDSDataset(fit,labels,tokenizer,value.max_length)
        model,lineage=_build_axis_model(source.backbone,CandidateSpec("coral-natural","coral","natural"))
        trainer=_train(model,train_data,tokenizer,source,restricted/METHOD/axis/"lora",seed=seed,max_steps=2 if smoke else -1)
        need(trainer.lora_gradient_observed and trainer.head_gradient_observed,"finite nonzero LoRA/head gradients not proven")
        predictions.append(_predict(model,held,tokenizer,value.max_length,value.batch_size))
        checkpoint_sha=_save_state(restricted/METHOD/axis/"trainable.safetensors",model)
        disclosures.append({"axis":axis,"phase1_seed":seed,"checkpoint_sha256":checkpoint_sha,"lineage":lineage,
                            "checkpoint_tensor_count":292,"checkpoint_lora_tensor_count":288,"checkpoint_coral_tensor_count":4,
                            "fit_records":len(fit),"lds":train_data.lds,"fit_token_length_audit":token_length_audit(fit,tokenizer,value.max_length),
                            "held_token_length_audit":token_length_audit(held,tokenizer,value.max_length)})
        del trainer,model; torch.cuda.empty_cache()
    matrix=np.column_stack(predictions); set_process_title(f"{'smoke' if smoke else 'oof'}:f{outer_fold}:persist")
    private=restricted/METHOD/"predictions.jsonl"
    private_sha=_atomic_private_jsonl(private,({"source_id":r.identifier,"outer_fold":outer_fold,
        "prediction":{axis:float(matrix[i,j]) for j,axis in enumerate(axes)}} for i,r in enumerate(held)))
    result={"schema_version":SCHEMA_VERSION,"status":"completed","mode":"smoke" if smoke else "outer_fold",
            "nonselectable":smoke,"run_id":value.run_id,"outer_fold":outer_fold,
            "records":len(held),"method":METHOD,"training_performed":True,"calibration_performed":False,
            "selection_performed":False,"axis_bindings":disclosures,"restricted_prediction_sha256":private_sha,
            "config_sha256":config_sha256(value),"config_file_sha256":_config_file_sha256(),
            "task_card_sha256":value.task_card_sha256,"direct_aggregate_sha256":value.direct_aggregate_sha256,
            "r0_oof_prediction_sha256":value.r0_oof_prediction_sha256,"validation_rows_loaded":False,
            "average_target_used":False,"git_sha":_git_sha(),"environment":_environment(),
            "logical_command":f"PYTHONPATH=src {sys.executable} scripts/run_kure_lds_oof.py --config configs/kure_lds_oof.v1.json --outer-fold {outer_fold}"+(" --smoke" if smoke else "")}
    _atomic_public_json(Path(value.output_root)/("smoke/outer-00.json" if smoke else f"outer-{outer_fold:02d}.json"),result)
    return result


def _load_private(path:Path,fold:int)->Mapping[str,tuple[float,float,float]]:
    _assert_private_file(path); result={}
    for line in path.read_text(encoding="utf-8").splitlines():
        row=json.loads(line); need(set(row)=={"source_id","outer_fold","prediction"} and row["outer_fold"]==fold,"prediction row differs")
        p=row["prediction"]; need(set(p)==set(AXES),"prediction axes differ"); values=tuple(float(p[a]) for a in AXES)
        need(all(math.isfinite(x) and 1<=x<=5 for x in values) and row["source_id"] not in result,"prediction value/id differs")
        result[row["source_id"]]=values
    need(len(result)==400,"prediction fold size differs"); return result


def _validate_prediction_fold_membership(predictions:Mapping[str,Any],expected_ids:set[str],fold:int)->None:
    need(len(expected_ids)==400 and set(predictions)==expected_ids,
         f"prediction IDs do not belong exactly to frozen fold {fold}")


def _validate_axis_disclosures(public:Mapping[str,Any],config:KURELDSOOFConfig,fold:int)->None:
    source=_source(config); items=public.get("axis_bindings")
    need(isinstance(items,list) and [x.get("axis") for x in items]==list(AXES),"axis disclosure inventory differs")
    for item,axis in zip(items,AXES,strict=True):
        checkpoint=Path(config.restricted_output_root)/f"outer-{fold:02d}"/METHOD/axis/"trainable.safetensors"
        authenticated=_authenticate_checkpoint(checkpoint,str(item.get("checkpoint_sha256","")))
        lds=item.get("lds",{}); lineage=item.get("lineage",{})
        need(item.get("phase1_seed")==derived_seed(config.seed,fold,"coral-natural",axis,"phase1")
             and item.get("checkpoint_sha256")==file_sha256(checkpoint) and item.get("fit_records")==1600
             and item.get("checkpoint_tensor_count")==authenticated["tensor_count"]==292
             and item.get("checkpoint_lora_tensor_count")==authenticated["lora_tensor_count"]==288
             and item.get("checkpoint_coral_tensor_count")==authenticated["coral_tensor_count"]==4
             and isinstance(lds,Mapping) and abs(float(lds.get("weight_mean",math.nan))-1.)<=config.mean_weight_tolerance
             and .25<=float(lds.get("weight_min",math.nan))<=float(lds.get("weight_max",math.nan))<=4.
             and float(lds.get("observed_density_min",0))>0 and abs(float(lds.get("kernel_sum",0))-1.)<1e-12
             and lineage.get("arm")=="aihub_full_backbone" and lineage.get("pooling")=="cls_l2"
             and lineage.get("artifact_sha256")==source.backbone.warmstart_artifact_sha256,
             "axis seed/checkpoint/LDS/model lineage differs")


def _validate_outer_public(public:Mapping[str,Any],config:KURELDSOOFConfig,fold:int,private_path:Path)->None:
    need(public.get("schema_version")==SCHEMA_VERSION and public.get("status")=="completed"
         and public.get("mode")=="outer_fold" and public.get("nonselectable") is False
         and public.get("run_id")==config.run_id and public.get("outer_fold")==fold and public.get("records")==400
         and public.get("method")==METHOD and public.get("training_performed") is True
         and public.get("calibration_performed") is False and public.get("selection_performed") is False
         and public.get("config_sha256")==config_sha256(config) and public.get("config_file_sha256")==_config_file_sha256()
         and public.get("task_card_sha256")==config.task_card_sha256
         and public.get("direct_aggregate_sha256")==config.direct_aggregate_sha256
         and public.get("r0_oof_prediction_sha256")==config.r0_oof_prediction_sha256
         and public.get("restricted_prediction_sha256")==file_sha256(private_path)
         and public.get("validation_rows_loaded") is False and public.get("average_target_used") is False,
         "outer public identity/lineage contract differs")
    _validate_axis_disclosures(public,config,fold)


def fold_status(config:KURELDSOOFConfig,fold:int)->str:
    """Authenticate immutable complete folds; permit only truly missing folds to resume."""
    aggregate_path=Path(config.output_root)/"aggregate.json"
    need(not _lexists(aggregate_path),"complete LDS aggregate exists or is a symlink; scientific rerun is forbidden")
    public_path=Path(config.output_root)/f"outer-{fold:02d}.json"
    restricted_fold=Path(config.restricted_output_root)/f"outer-{fold:02d}"
    private_path=restricted_fold/METHOD/"predictions.jsonl"
    if not _lexists(public_path) and not _lexists(private_path):
        need(not _lexists(Path(config.output_root)) or (Path(config.output_root).is_dir() and not Path(config.output_root).is_symlink()),
             "public output root is a symlink sentinel")
        need(not _lexists(Path(config.restricted_output_root)) or (Path(config.restricted_output_root).is_dir() and not Path(config.restricted_output_root).is_symlink()),
             "restricted output root is a symlink sentinel")
        public_partial=(list(Path(config.output_root).glob(f"outer-{fold:02d}*"))
                        +list(Path(config.output_root).glob(f".outer-{fold:02d}*"))) if Path(config.output_root).exists() else []
        restricted_partial=(list(Path(config.restricted_output_root).glob(f"outer-{fold:02d}*"))
                            +list(Path(config.restricted_output_root).glob(f".outer-{fold:02d}*"))) if Path(config.restricted_output_root).exists() else []
        need(not public_partial and not restricted_partial,
             "partial fold trainer/checkpoint/artifact exists; recovery amendment required")
        return "missing"
    need(public_path.is_file() and not public_path.is_symlink() and private_path.is_file() and not private_path.is_symlink(),
         "partial fold artifacts cannot be overwritten; recovery amendment required")
    public=json.loads(public_path.read_text(encoding="utf-8")); predictions=_load_private(private_path,fold)
    _validate_outer_public(public,config,fold,private_path)
    projection,_=_load_projection(config); _validate_prediction_fold_membership(predictions,{row.identifier for row in projection[fold]},fold)
    return "completed_immutable"


def aggregate(config:KURELDSOOFConfig|str|Path)->Mapping[str,Any]:
    value=KURELDSOOFConfig.from_json(config) if isinstance(config,(str,Path)) else config
    set_process_title("aggregate"); value.require_execution_authorization()
    projection,folds=_load_projection(value); predictions={}; bindings=[]
    for fold in range(5):
        public_path=Path(value.output_root)/f"outer-{fold:02d}.json"; private_path=Path(value.restricted_output_root)/f"outer-{fold:02d}"/METHOD/"predictions.jsonl"
        need(public_path.is_file() and private_path.is_file(),"outer output incomplete")
        public=json.loads(public_path.read_text()); _validate_outer_public(public,value,fold,private_path)
        current=_load_private(private_path,fold)
        _validate_prediction_fold_membership(current,{row.identifier for row in projection[fold]},fold)
        need(not set(current)&set(predictions),"OOF overlap"); predictions.update(current)
        bindings.append({"outer_fold":fold,"public_sha256":file_sha256(public_path),"restricted_prediction_sha256":file_sha256(private_path)})
    need(len(predictions)==2000,"OOF coverage differs")
    # Only after all durable predictions authenticate may canonical fold/gold/R0 be opened.
    need(set(predictions)==set(folds),"prediction/projection population differs")
    verify_post_prediction_gold_dependencies(value)  # compatible structural fields, no direct authorization use
    from .r0_ordinal_residual import load_embedding_artifact
    _,canonical=load_embedding_artifact(value.fold_manifest_path,value.fold_rows_path)
    need({r.source_id:int(r.oof_fold) for r in canonical}==folds,"canonical fold assignment differs")
    source=_source(value); truth=load_raw_axis_gold(value.train_path,value.train_sha256); r0=load_exact_r0(source); ordered=list(truth)
    truth_a=np.asarray([truth[x] for x in ordered]); r0_a=np.asarray([r0[x] for x in ordered]); candidate=np.asarray([predictions[x] for x in ordered])
    metrics=compute_iterative_tail_metrics(truth_a,candidate); baseline=compute_iterative_tail_metrics(truth_a,r0_a)
    raw_decision=dict(promotion_gate(truth_a,r0_a,candidate,ordered,seed=value.seed))
    raw_decision["candidate_metrics"]=raw_decision.pop("coral_natural_metrics")
    decision=raw_decision
    result={"schema_version":"mal2026-kure-lds-oof-aggregate-v1","status":"completed","mode":"full_oof",
            "run_id":value.run_id,"records":2000,"folds":5,"method":METHOD,
            "metrics":metrics,"exact_r0_metrics":baseline,"improvements_vs_exact_r0":metric_improvements(baseline,metrics),
            "prediction_diagnostics":prediction_band_diagnostics(candidate),"common_stage3_promotion_gate":decision,
            "common_stage3_promotion_gate_passed":decision["eligible"],"automatic_stage6_deployment_eligible":False,
            "protected_output":"exact_r0","fold_bindings":bindings,"config_sha256":config_sha256(value),
            "config_file_sha256":_config_file_sha256(),"task_card_sha256":value.task_card_sha256,
            "direct_aggregate_sha256":value.direct_aggregate_sha256,"r0_oof_prediction_sha256":value.r0_oof_prediction_sha256,
            "validation_rows_loaded":False,"average_target_used":False,"git_sha":_git_sha(),"environment":_environment()}
    _atomic_public_json(Path(value.output_root)/"aggregate.json",result); return result


__all__=["KURELDSOOFConfig","KURELDSOOFError","aggregate","assert_grid_alignment","gaussian_kernel",
         "fold_status","lds_example_weights","run","scheduler_state_conflict","smoothed_density","solve_clipped_mean_one",
         "set_process_title","summarize_gpu_telemetry","weighted_hybrid_loss"]
