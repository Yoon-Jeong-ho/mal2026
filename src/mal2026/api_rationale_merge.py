"""Compose three separately generated axis rationales into one private artifact."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .api_rationale_data import AXES, RESTRICTED_ROOT, APIRationaleContractError, aggregate_input_provenance, load_generated_rationales, sha256_file
from .api_rationale_sft import SUPPORTED_MODELS


OUTPUT_ROOT = RESTRICTED_ROOT / "decoder_generation_v1"


class APIRationaleMergeError(APIRationaleContractError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition: raise APIRationaleMergeError(message)


def _sha(path: Path) -> str: return sha256_file(path)


def _atomic(path: Path, value: Mapping[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"); partial.replace(path)


@dataclass(frozen=True)
class APIRationaleMergeConfig:
    schema_version: str
    run_id: str
    base_key: str
    source: str
    content_generation_dir: str
    organization_generation_dir: str
    expression_generation_dir: str
    output_dir: str

    @classmethod
    def from_json(cls, path: Path) -> "APIRationaleMergeConfig":
        raw=json.loads(path.read_text(encoding="utf-8")); _need(isinstance(raw,dict) and set(raw)==set(cls.__dataclass_fields__),"merge config has unknown or missing fields")
        value=cls(**raw);value.validate();return value

    def validate(self) -> None:
        _need(self.schema_version=="mal2026-api-rationale-merge-v1" and self.base_key in SUPPORTED_MODELS and self.source=="validation", "merge config identity differs")
        output=Path(self.output_dir); _need(output.is_absolute() and output.parent==OUTPUT_ROOT.resolve() and not output.exists(),"merge output must be fresh restricted direct child")
        expected=f"api-rationale-generation-v1-{self.base_key}-axis_triplet-validation-002"; _need(self.run_id==expected and output.name==expected,"merge lineage differs")
        for field,axis in ((self.content_generation_dir,"content"),(self.organization_generation_dir,"organization"),(self.expression_generation_dir,"expression")):
            path=Path(field);_need(path.is_absolute() and path.parent==OUTPUT_ROOT.resolve() and path.name==f"api-rationale-generation-v1-{self.base_key}-{axis}-validation-002","axis-generation lineage differs")


def run_api_rationale_merge(config: APIRationaleMergeConfig) -> dict[str, Any]:
    config.validate()
    pieces={"content":load_generated_rationales(Path(config.content_generation_dir),source="validation",task="content"),"organization":load_generated_rationales(Path(config.organization_generation_dir),source="validation",task="organization"),"expression":load_generated_rationales(Path(config.expression_generation_dir),source="validation",task="expression")}
    ids=set(pieces["content"]);_need(ids==set(pieces["organization"])==set(pieces["expression"]) and len(ids)==400,"axis generated populations differ")
    output=Path(config.output_dir);output.mkdir(mode=0o700,parents=True)
    records=output/"generated_rationales.jsonl"
    with records.open("x",encoding="utf-8") as handle:
        for identifier in sorted(ids):
            item={"source_id":identifier,"rationale":{axis:pieces[axis][identifier][axis] for axis in AXES},"parse_valid":True,"failure_category":None,"attempts":0}
            handle.write(json.dumps(item,ensure_ascii=False,separators=(",",":"),sort_keys=True)+"\n")
    report={"status":"completed","run_id":config.run_id,"source":"validation","base_key":config.base_key,"task":"axis_triplet","counts":{"expected":400,"observations":400,"parse_valid":400},"hard_gates":{"complete_records":True,"all_outputs_parse_valid":True,"zero_transport_or_schema_failures":True,"axis_populations_identical":True},"failure_categories":{},"generated_rationales_sha256":_sha(records),"source_axis_generation_report_sha256":{axis:_sha(Path(getattr(config,f"{axis}_generation_dir"))/"aggregate_generation_report.json") for axis in AXES},"input_provenance":aggregate_input_provenance(),"raw_prompts_or_model_completions_persisted":False}
    _atomic(output/"aggregate_generation_report.json",report)
    manifest={"schema_version":config.schema_version,"status":"completed","run_id":config.run_id,"source":"validation","base_key":config.base_key,"task":"axis_triplet","config":asdict(config),"aggregate_report_sha256":_sha(output/"aggregate_generation_report.json"),"source_writing_scores_read_or_prompted":False,"candidate_scores_read_or_prompted":False,"raw_prompts_or_model_completions_persisted":False}
    _atomic(output/"manifest.json",manifest);return report
