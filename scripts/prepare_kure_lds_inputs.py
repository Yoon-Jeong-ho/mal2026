#!/usr/bin/env python3
"""Data-steward-only creation of 15 outer-fit/axis raw-label artifacts."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from hashlib import sha256
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; AXES=("content","organization","expression")
sys.path.insert(0,str(ROOT/"src"))
from mal2026.kure_lds_oof import KURELDSOOFConfig
def need(x,msg):
    if not x: raise RuntimeError(msg)
def digest(path):
    path=Path(path); need(path.is_file() and not path.is_symlink(),f"ordinary file required: {path}")
    h=sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def secure(parent):
    anchor=(ROOT/"data/processed/restricted").resolve(); parent=parent.resolve()
    need(anchor==parent or anchor in parent.parents,"output outside restricted anchor")
    chain=[parent]
    while chain[-1]!=anchor: chain.append(chain[-1].parent)
    for p in reversed(chain): p.mkdir(exist_ok=True); os.chmod(p,0o770); need(p.stat().st_mode&7==0,"world ACL differs")
def verify_private(path,anchor):
    path=path.resolve(); anchor=anchor.resolve(); need(path.is_file() and not path.is_symlink() and path.stat().st_mode&7==0,"private source ACL differs")
    cursor=path.parent; need(anchor==cursor or anchor in cursor.parents,"private source outside anchor")
    while True:
        need(cursor.is_dir() and not cursor.is_symlink() and cursor.stat().st_mode&7==0,"private parent ACL differs")
        if cursor==anchor: break
        cursor=cursor.parent
def publish(path,rows):
    need(not path.exists() and not path.is_symlink(),f"refusing to overwrite {path}"); secure(path.parent)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent,text=True); tmp=Path(tmp)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
            f.flush(); os.fsync(f.fileno())
        os.chmod(tmp,0o660); os.link(tmp,path); tmp.unlink(); os.chmod(path,0o660)
        d=os.open(path.parent,os.O_DIRECTORY); os.fsync(d); os.close(d)
    except BaseException: tmp.unlink(missing_ok=True); raise
    return digest(path)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,required=True); args=ap.parse_args()
    config_path=args.config.resolve(); need(config_path==ROOT/"configs/kure_lds_oof.v1.json","canonical LDS config is required")
    parsed=KURELDSOOFConfig.from_json(config_path); parsed.require_steward_authorization(); config=json.loads(config_path.read_text(encoding="utf-8"))
    need(config["status"]=="authorized_for_steward_preparation" and config["steward_preparation_authorized"] is True
         and config["execution_authorized"] is False,"steward preparation requires its exclusive intermediate state")
    need(digest(Path(__file__).resolve())==config["preparer_sha256"],"executed preparer hash differs")
    git_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    projection=ROOT/config["label_free_projection_path"]; projection_sha=digest(projection)
    verify_private(projection,ROOT/"data/processed/restricted")
    if config["label_free_projection_sha256"]: need(projection_sha==config["label_free_projection_sha256"],"projection hash differs")
    folds={}; held={f:set() for f in range(5)}
    for line in projection.read_text(encoding="utf-8").splitlines():
        row=json.loads(line); need(set(row)=={"id","document_id","prompt_num","prompt","essay","outer_fold"},"projection schema differs")
        need(row["id"] not in folds and row["outer_fold"] in range(5),"projection membership differs")
        folds[row["id"]]=row["outer_fold"]; held[row["outer_fold"]].add(row["id"])
    need(len(folds)==2000 and all(len(x)==400 for x in held.values()),"projection coverage differs")
    train=ROOT/config["train_path"]; verify_private(train,ROOT/"eval"); need(digest(train)==config["train_sha256"],"canonical train hash differs")
    # Membership is fully frozen above before any score object is indexed.
    scores={}
    with train.open(encoding="utf-8") as f:
        for line in f:
            row=json.loads(line); identifier=row["id"]
            need(identifier in folds and identifier not in scores and isinstance(row.get("score"),dict),"canonical identity/schema differs")
            score=row["score"]; need(all(axis in score for axis in AXES),"three axes missing")
            # Never access score['average']; only these explicit keys are indexed.
            scores[identifier]={axis:float(score[axis]) for axis in AXES}
    need(set(scores)==set(folds),"canonical score population differs")
    bindings=[]; expected={(x["outer_fold"],x["axis"]):x for x in config["fit_label_bindings"]}
    need(set(expected)=={(f,a) for f in range(5) for a in AXES},"15 binding inventory differs")
    for fold in range(5):
        fit_ids=[identifier for identifier,assigned in folds.items() if assigned!=fold]
        need(len(fit_ids)==1600 and not held[fold].intersection(fit_ids),"held exclusion differs")
        for axis in AXES:
            item=expected[(fold,axis)]; path=ROOT/item["path"]
            value_sha=publish(path,({"source_id":identifier,"raw_score":scores[identifier][axis]} for identifier in fit_ids))
            bindings.append({"outer_fold":fold,"axis":axis,"path":item["path"],"sha256":value_sha,"records":1600})
    generator=Path(__file__).resolve(); manifest={
      "schema_version":"mal2026-kure-lds-fit-label-manifest-v1","status":"completed","records_per_outer_fit":1600,
      "axes":list(AXES),"bindings":bindings,"average_indexed":False,"text_present":False,"held_ids_present":False,
      "source_train_sha256":config["train_sha256"],"label_free_projection_sha256":projection_sha,
      "label_free_manifest_sha256":config["label_free_manifest_sha256"],
      "steward_task_card_sha256":config["steward_task_card_sha256"],
      "steward_task_card_commit":config["steward_task_card_commit"],"direct_aggregate_sha256":config["direct_aggregate_sha256"],
      "direct_task_card_sha256":config["direct_task_card_sha256"],
      "direct_config_file_sha256":config["direct_config_file_sha256"],
      "direct_report_config_sha256":config["direct_report_config_sha256"],
      "generator_path":str(generator.relative_to(ROOT)),"generator_sha256":digest(generator),
      "generator_git_sha":config["preparer_commit"],"execution_git_sha":git_sha,
      "preparation_request_config_sha256":config["preparation_request_config_sha256"],
      "steward_authorized_config_sha256":digest(config_path)}
    manifest_path=ROOT/config["fit_label_manifest_path"]; manifest_sha=publish(manifest_path,[manifest])
    print(json.dumps({"status":"completed","projection_sha256":projection_sha,"fit_label_manifest_sha256":manifest_sha,
                      "fit_label_bindings":bindings},sort_keys=True))
if __name__=="__main__": main()
