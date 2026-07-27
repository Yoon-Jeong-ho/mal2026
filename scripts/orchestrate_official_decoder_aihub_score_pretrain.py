#!/usr/bin/env python3
"""GPU0 smoke gates followed by GPU0--3 FSDP full decoder pretraining."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from mal2026.official_decoder_aihub_pretrain import ARCHITECTURES, DecoderAIHubConfig, artifact_inventory
from mal2026.official_decoder_score import file_sha256

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-standard" / "bin" / "python"
TRAINER = ROOT / "scripts" / "train_official_decoder_aihub_score_pretrain.py"
ALLOWED_GENERATIVE_REUSE_ROOT = (
    ROOT / "outputs/official-decoder-aihub-integer-score-full-pretrain-v1"
    / "official-decoder-aihub-integer-score-full-pretrain-v1-20260727-005"
)

def _command(config: Path, architecture: str, phase: str, smoke: bool, selection: Path | None = None, distributed: bool = False) -> list[str]:
    base = [str(TRAINER), "--config", str(config), "--architecture", architecture, "--phase", phase]
    if selection is not None: base += ["--selection-metadata", str(selection)]
    if smoke: base.append("--smoke")
    return ([str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=4"] if distributed else [str(PYTHON)]) + base

def plan(config_path: Path, config: DecoderAIHubConfig) -> list[dict[str, Any]]:
    root = Path(config.output_root) / config.run_id
    stages = []
    for architecture in ARCHITECTURES:
        smoke_selection = root / f"smoke-{architecture}-selection" / "training_complete.json"
        selection = root / f"{architecture}-selection" / "training_complete.json"
        stages.extend([
            {"architecture": architecture, "phase": "selection", "stage": "gpu0_one_update_smoke", "gpus": [0], "command": _command(config_path, architecture, "selection", True)},
            {"architecture": architecture, "phase": "refit", "stage": "gpu0_one_update_smoke", "gpus": [0], "command": _command(config_path, architecture, "refit", True, smoke_selection)},
        ])
        if architecture == "generative":
            stages.append({"architecture": architecture, "phase": "selection", "stage": "fsdp4_one_update_preflight", "gpus": [0,1,2,3], "command": _command(config_path, architecture, "selection", True, distributed=True)})
        stages.extend([
            {"architecture": architecture, "phase": "selection", "stage": "fsdp4_full_parameter", "gpus": [0,1,2,3], "command": _command(config_path, architecture, "selection", False, distributed=True)},
            {"architecture": architecture, "phase": "refit", "stage": "fsdp4_full_parameter", "gpus": [0,1,2,3], "command": _command(config_path, architecture, "refit", False, selection, True)},
        ])
    return stages

def _gpu_processes(gpus: list[int]) -> list[str]:
    result = subprocess.run(["nvidia-smi", f"--id={','.join(map(str,gpus))}", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], cwd=ROOT, text=True, capture_output=True, check=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

def _result_from_completion(architecture: str, completion: Path, *, source_run_id: str) -> dict[str, Any]:
    payload = json.loads(completion.read_text(encoding="utf-8"))
    state = payload["state"]
    return {
        "architecture": architecture,
        "source_run_id": source_run_id,
        "selected_global_step": payload["selection"]["selected_event"]["global_step"],
        "selection_metrics": {key: payload["selection"]["selected_event"][key] for key in ("macro_integer_rmse", "macro_integer_spearman", "macro_continuous_rmse")},
        "artifact_path": state["artifact_path"], "artifact_sha256": state["artifact_sha256"],
        "state_metadata_path": state["metadata_path"], "state_metadata_sha256": state["metadata_sha256"],
        "completion_path": str(completion.resolve()), "completion_sha256": file_sha256(completion),
    }

def _validate_reused_generative(config: DecoderAIHubConfig, source_root: Path) -> dict[str, Any]:
    if source_root.resolve() != ALLOWED_GENERATIVE_REUSE_ROOT.resolve() or not source_root.is_dir() or source_root.is_symlink():
        raise RuntimeError("generative reuse root is outside the one declared recovery lineage")
    selection_path = source_root / "generative-selection/training_complete.json"
    refit_path = source_root / "generative-refit/training_complete.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    refit = json.loads(refit_path.read_text(encoding="utf-8"))
    for payload, phase in ((selection, "selection"), (refit, "refit")):
        if payload.get("status") != "completed" or payload.get("architecture") != "generative" or payload.get("phase") != phase:
            raise RuntimeError(f"reused generative {phase} completion differs")
        if payload.get("identity") != config.identity("generative") or payload.get("canonical_validation_access") is not False:
            raise RuntimeError(f"reused generative {phase} protocol differs")
        if payload.get("run_id") != source_root.name or payload.get("training_method") != "full_parameter":
            raise RuntimeError(f"reused generative {phase} lineage differs")
    if selection["selection"]["selected_event"] != refit["selection"]["selected_event"]:
        raise RuntimeError("reused generative selected event differs between selection and refit")
    if refit["trainer"]["global_step"] != selection["selection"]["selected_event"]["global_step"]:
        raise RuntimeError("reused generative refit did not replay the selected step")
    state = refit.get("state", {})
    artifact = Path(state.get("artifact_path", ""))
    metadata = Path(state.get("metadata_path", ""))
    if artifact.resolve() != (source_root / "generative-refit/full_model").resolve() or metadata.resolve() != (source_root / "generative-refit/full_model_state.json").resolve():
        raise RuntimeError("reused generative state paths differ")
    if file_sha256(metadata) != state.get("metadata_sha256"):
        raise RuntimeError("reused generative metadata checksum differs")
    inventory, artifact_sha = artifact_inventory(artifact)
    metadata_payload = json.loads(metadata.read_text(encoding="utf-8"))
    if artifact_sha != state.get("artifact_sha256") or metadata_payload.get("inventory") != inventory:
        raise RuntimeError("reused generative artifact inventory differs")
    return _result_from_completion("generative", refit_path, source_run_id=source_root.name)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reuse-generative-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = DecoderAIHubConfig.from_json(args.config, require_dependencies=not args.dry_run)
    stages = plan(args.config, config)
    if args.reuse_generative_from is not None:
        if args.reuse_generative_from.resolve() != ALLOWED_GENERATIVE_REUSE_ROOT.resolve():
            raise RuntimeError("dry/real generative reuse root differs")
        stages = [stage for stage in stages if stage["architecture"] != "generative"]
    if args.dry_run:
        print(json.dumps({"status":"dry_run_passed","gpu_started":False,"gpu_scope":{"smoke":[0],"full":[0,1,2,3]},"training_method":"full_parameter","distributed_strategy":"fsdp_full_shard_auto_wrap","fsdp_version":config.fsdp_version,"canonical_validation_access":False,"reused_generative_root":str(args.reuse_generative_from.resolve()) if args.reuse_generative_from else None,"stages":stages}, indent=2, sort_keys=True))
        return
    reused_generative = _validate_reused_generative(config, args.reuse_generative_from) if args.reuse_generative_from is not None else None
    root = Path(config.output_root) / config.run_id
    if root.exists(): raise RuntimeError(f"refusing to reuse run root: {root}")
    root.mkdir(parents=True); logs = root / "logs"; logs.mkdir()
    manifest_path = root / "manifest.json"; ledger = root / "ledger.jsonl"
    manifest = {"schema_version":"mal2026-official-decoder-aihub-pretrain-run-v1","status":"running","started_at":datetime.now(timezone.utc).isoformat(),"git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"gpu_scope":{"smoke":[0],"full":[0,1,2,3],"authorization":"default MAL2026 GPU scope"},"training_method":"full_parameter","fsdp_version":config.fsdp_version,"canonical_validation_access":False,"reused_completed_architectures":{"generative":reused_generative} if reused_generative else {}}
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    env=os.environ.copy(); env.update({"PYTHONPATH":str(ROOT/"src"),"TOKENIZERS_PARALLELISM":"false","WANDB_DISABLED":"true"})
    for index,stage in enumerate(stages):
        if (processes:=_gpu_processes(stage["gpus"])): raise RuntimeError(f"GPU conflict; no process altered: {processes}")
        env["CUDA_VISIBLE_DEVICES"]=",".join(map(str,stage["gpus"]))
        with ledger.open("a") as handle: handle.write(json.dumps({"event":"start",**stage},sort_keys=True)+"\n")
        log=logs/f"{index:02d}-{stage['architecture']}-{stage['phase']}-{stage['stage']}.log"
        with log.open("x") as handle: completed=subprocess.run(stage["command"],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
        with ledger.open("a") as handle: handle.write(json.dumps({"event":"completed" if completed.returncode==0 else "failed","exit_code":completed.returncode,"log":str(log),**stage},sort_keys=True)+"\n")
        if completed.returncode: raise RuntimeError(f"stage failed: {stage['architecture']} {stage['phase']}")
    results=[]
    for architecture in ARCHITECTURES:
        if architecture == "generative" and reused_generative is not None:
            results.append(reused_generative)
            continue
        completion=root/f"{architecture}-refit"/"training_complete.json"; payload=json.loads(completion.read_text())
        results.append(_result_from_completion(architecture, completion, source_run_id=config.run_id))
    aggregate={"schema_version":"mal2026-official-decoder-aihub-pretrain-aggregate-v1","status":"completed","run_id":config.run_id,"score_fields":list(config.score_fields),"integer_target_used":True,"average_target_used":False,"training_method":"full_parameter","downstream_adaptation":"fresh_MAL_LoRA","canonical_validation_access":False,"reused_completed_architectures":["generative"] if reused_generative else [],"results":results,"privacy":"aggregate_only_no_rows_text_ids_or_predictions"}
    aggregate_path=root/"aggregate_results.json"; aggregate_path.write_text(json.dumps(aggregate,indent=2,sort_keys=True)+"\n")
    manifest.update({"status":"completed","completed_at":datetime.now(timezone.utc).isoformat(),"aggregate_sha256":file_sha256(aggregate_path)}); manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")

if __name__=="__main__": main()
