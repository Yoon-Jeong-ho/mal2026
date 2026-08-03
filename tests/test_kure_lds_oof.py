from __future__ import annotations
from dataclasses import asdict,replace
import inspect,json,os,subprocess,tempfile,unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
import mal2026.kure_lds_oof as m
from mal2026.kure_lds_oof import (KURELDSOOFConfig,KURELDSOOFError,assert_grid_alignment,
    gaussian_kernel,lds_example_weights,smoothed_density,solve_clipped_mean_one,weighted_hybrid_loss)

class KURELDSOOFTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.raw=json.loads(Path("configs/kure_lds_oof.v1.json").read_text())
 def config(self): return KURELDSOOFConfig.from_mapping(self.raw)
 def trainable_state(self):
  state={"score.weight":torch.zeros(1,1024),"score.bias":torch.zeros(1),"cut_base":torch.zeros(()),"cut_gaps":torch.zeros(3)}
  for i in range(120): state[f"layer1024.{i}.lora_A.default.weight"]=torch.zeros(16,1024); state[f"layer1024.{i}.lora_B.default.weight"]=torch.zeros(1024,16)
  for i in range(24): state[f"layer4096.{i}.lora_A.default.weight"]=torch.zeros(16,4096); state[f"layer4096.{i}.lora_B.default.weight"]=torch.zeros(4096,16)
  return state
 def test_pending_config_and_exact_inventory(self):
  c=self.config(); self.assertEqual((c.status,c.execution_authorized,c.steward_preparation_authorized),("pending_direct_failure_and_scientific_authorization",False,False))
  self.assertEqual([(x.outer_fold,x.axis) for x in c.fit_label_bindings],[(f,a) for f in range(5) for a in m.AXES])
  self.assertTrue(all(x.sha256=="" for x in c.fit_label_bindings))
  self.assertEqual((c.task_card_sha256,c.preparer_sha256,c.direct_aggregate_sha256,c.fit_label_manifest_sha256),("","","",""))
  self.assertEqual(c.integration_recovery_policy,"identical_hash_missing_fold_resume_only_preserve_partial_immutable_no_complete_rerun_no_scientific_retune")
 def test_pending_run_and_aggregate_fail_closed(self):
  c=self.config(); self.assertEqual(m.run(c,outer_fold=0,validate_only=True)["gpu_used"],False)
  with self.assertRaisesRegex(KURELDSOOFError,"not authorized"): m.run(c,outer_fold=0)
  with self.assertRaisesRegex(KURELDSOOFError,"not authorized"): m.aggregate(c)
 def test_two_stage_authorization_shape_is_unambiguous(self):
  raw=json.loads(json.dumps(self.raw)); raw.update({"status":"authorized_for_steward_preparation","steward_preparation_authorized":True})
  for key in ("steward_task_card_sha256","task_card_sha256","preparer_sha256","preparation_request_config_sha256","direct_aggregate_sha256",
              "direct_task_card_sha256","direct_config_file_sha256","direct_report_config_sha256",
              "label_free_projection_sha256","label_free_manifest_sha256"): raw[key]="a"*64
  raw["steward_task_card_commit"]=raw["task_card_commit"]=raw["preparer_commit"]="b"*40
  steward=KURELDSOOFConfig.from_mapping(raw); self.assertFalse(steward.execution_authorized)
  with self.assertRaisesRegex(KURELDSOOFError,"scientific execution"): steward.require_execution_authorization()
  final=json.loads(json.dumps(raw)); final.update({"status":"authorized","execution_authorized":True})
  final["steward_authorized_config_sha256"]="9"*64
  final["fit_label_manifest_sha256"]="c"*64
  for item in final["fit_label_bindings"]: item["sha256"]="d"*64
  configured=KURELDSOOFConfig.from_mapping(final); self.assertTrue(configured.execution_authorized)
  with self.assertRaisesRegex(KURELDSOOFError,"data-steward"): configured.require_steward_authorization()
 def test_pending_preparer_dynamically_refuses_before_artifacts(self):
  target=Path(self.raw["fit_label_manifest_path"]); before=target.exists()
  result=subprocess.run([str(Path(".venv-standard/bin/python")),"scripts/prepare_kure_lds_inputs.py","--config","configs/kure_lds_oof.v1.json"],capture_output=True,text=True)
  self.assertNotEqual(result.returncode,0); self.assertIn("not authorized",result.stderr); self.assertEqual(target.exists(),before)
 def test_decimal_grid_alignment_is_exact(self):
  np.testing.assert_array_equal(assert_grid_alignment([1,1.05,3.25,5]),[0,1,45,80])
  for bad in (0.95,5.05,1.001,float("nan"),1.1+1e-12):
   with self.assertRaises(KURELDSOOFError): assert_grid_alignment([bad])
 def test_gaussian_kernel_and_constant_zero_density(self):
  k=gaussian_kernel(); self.assertEqual(len(k),21); self.assertAlmostEqual(float(k.sum()),1,14); self.assertTrue(np.all(k>0)); self.assertTrue(np.allclose(k,k[::-1]))
  density,indices=smoothed_density([1.0]); self.assertEqual(indices.tolist(),[0]); self.assertAlmostEqual(density[0],k[10]); self.assertEqual(float(density[11]),0.)
 def test_lower_crossing_bisection_and_infeasible_inputs(self):
  u=np.asarray([.1,.2,1.,2.,10.]); w,c=solve_clipped_mean_one(u)
  self.assertLessEqual(abs(float(w.mean())-1),1e-6); self.assertTrue(np.all((w>=.25)&(w<=4)))
  lo=np.nextafter(c,0.); self.assertLessEqual(float(np.clip(lo*u,.25,4).mean()),1.)
  # The implementation is fixed at 128 updates, upper-endpoint selection, and 64 permitted doublings.
  source=inspect.getsource(solve_clipped_mean_one); self.assertIn("range(128)",source); self.assertIn("range(65)",source); self.assertIn("c=hi",source)
  for bad in ([1.,0.], [1.,float("inf")], []):
   with self.assertRaises(KURELDSOOFError): solve_clipped_mean_one(bad)
  with self.assertRaises(KURELDSOOFError): solve_clipped_mean_one([1.],floor=1.1)
 def test_lds_weights_are_fit_only_bounded_mean_one(self):
  values=[1.0]*2+[2.0]*10+[3.0]*30+[4.0]*40+[5.0]*3
  weights,report=lds_example_weights(values); self.assertEqual(len(weights),len(values)); self.assertAlmostEqual(float(weights.mean()),1,6)
  self.assertGreater(report["observed_density_min"],0); self.assertGreaterEqual(weights.min(),.25); self.assertLessEqual(weights.max(),4)
 def test_weighted_loss_numeric_and_gradient(self):
  logits=torch.tensor([[0.,0.,0.,0.],[1.,.5,-.5,-1.]],requires_grad=True); rounded=torch.tensor([3,5]); raw=torch.tensor([3.2,4.8]); weights=torch.tensor([.5,1.5])
  loss=weighted_hybrid_loss(logits,rounded,raw,weights)
  targets=(rounded[:,None]>torch.arange(1,5)[None,:]).float(); ordinal=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction="none").mean(1)
  pmf=m.coral_pmf(logits); expected=(pmf*torch.arange(1,6)).sum(1); expected_loss=(weights*(ordinal+.25*(expected-raw).square())).mean()
  self.assertTrue(torch.allclose(loss,expected_loss)); loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all()); self.assertGreater(float(logits.grad.abs().sum()),0)
 def test_weighted_loss_rejects_broadcast_nonfinite_offgrid_and_bounds(self):
  valid=[torch.zeros(2,4),torch.tensor([2,4]),torch.tensor([2.0,4.0]),torch.tensor([.5,1.5])]
  invalid=[
   [torch.zeros(2,5),*valid[1:]], [valid[0],torch.tensor([[2],[4]]),*valid[2:]],
   [valid[0],valid[1],torch.tensor([2.01,4.0]),valid[3]], [valid[0],torch.tensor([2.5,4.]),*valid[2:]],
   [valid[0],valid[1],valid[2],torch.tensor([0.1,1.])], [torch.tensor([[float("nan")]*4]*2),*valid[1:]],
   [valid[0],valid[1],torch.tensor([2,4]),valid[3]],
  ]
  for args in invalid:
   with self.assertRaises(KURELDSOOFError): weighted_hybrid_loss(*args)
 def test_training_does_not_reuse_scalar_stage3_loss(self):
  source=inspect.getsource(m._train)+inspect.getsource(m.weighted_hybrid_loss)
  self.assertNotIn("hybrid_loss(",source.replace("weighted_hybrid_loss(","")); self.assertNotIn("_train_phase",source)
  self.assertIn("training_step",source); self.assertIn("gradient_observed",source)
 def test_fit_label_schema_and_held_exclusion_are_fail_closed(self):
  source=inspect.getsource(m._load_fit_labels); self.assertIn('{"source_id","raw_score"}',source); self.assertIn("identifier not in held",source)
  prep=Path("scripts/prepare_kure_lds_inputs.py").read_text(); self.assertNotIn('score["average"]',prep); self.assertIn('"average_indexed":False',prep)
  self.assertIn("not held[fold].intersection(fit_ids)",prep)
 def test_checkpoint_publish_authentication_no_clobber_acl_and_finite_inventory(self):
  class Fake:
   def __init__(self,state): self.state=state
   def state_dict(self): return self.state
  anchor=Path("data/processed/restricted")
  with tempfile.TemporaryDirectory(dir=anchor) as td:
   path=Path(td)/"fold"/"trainable.safetensors"; digest=m._save_state(path,Fake(self.trainable_state()))
   self.assertEqual(m._authenticate_checkpoint(path,digest)["tensor_count"],292); self.assertEqual(path.stat().st_mode&7,0)
   with self.assertRaisesRegex(KURELDSOOFError,"overwrite"): m._save_state(path,Fake(self.trainable_state()))
   bad=self.trainable_state(); bad["score.weight"][0,0]=float("nan")
   with self.assertRaisesRegex(KURELDSOOFError,"non-finite"): m._save_state(Path(td)/"bad.safetensors",Fake(bad))
   target=Path(td)/"target"; target.write_text("safe"); link=Path(td)/"link.safetensors"; link.symlink_to(target)
   with self.assertRaisesRegex(KURELDSOOFError,"overwrite"): m._save_state(link,Fake(self.trainable_state()))
   self.assertEqual(target.read_text(),"safe")
   temp_target=Path(td)/"temp-target"; temp_target.write_text("safe")
   final=Path(td)/"blocked.safetensors"; tmp=final.with_name(f".{final.name}.{os.getpid()}.tmp"); tmp.symlink_to(temp_target)
   with self.assertRaises(FileExistsError): m._save_state(final,Fake(self.trainable_state()))
   self.assertEqual(temp_target.read_text(),"safe"); self.assertTrue(tmp.is_symlink())
 def test_manifest_lineage_is_dynamically_authenticated(self):
  c=self.config(); bindings=tuple(replace(x,sha256="d"*64) for x in c.fit_label_bindings)
  configured=replace(c,status="authorized",execution_authorized=True,steward_preparation_authorized=True,
   steward_task_card_sha256="8"*64,steward_task_card_commit="9"*40,
   task_card_sha256="a"*64,task_card_commit="b"*40,preparer_sha256="c"*64,preparer_commit="e"*40,
   preparation_request_config_sha256="f"*64,steward_authorized_config_sha256="0"*64,
   direct_aggregate_sha256="1"*64,direct_task_card_sha256="2"*64,
   direct_config_file_sha256="3"*64,direct_report_config_sha256="4"*64,label_free_projection_sha256="5"*64,
   label_free_manifest_sha256="6"*64,fit_label_manifest_sha256="7"*64,fit_label_bindings=bindings)
  manifest={"schema_version":"mal2026-kure-lds-fit-label-manifest-v1","status":"completed","records_per_outer_fit":1600,
   "axes":list(m.AXES),"average_indexed":False,"text_present":False,"held_ids_present":False,
   "source_train_sha256":configured.train_sha256,"label_free_projection_sha256":configured.label_free_projection_sha256,
   "label_free_manifest_sha256":configured.label_free_manifest_sha256,
   "steward_task_card_sha256":configured.steward_task_card_sha256,"steward_task_card_commit":configured.steward_task_card_commit,
   "direct_aggregate_sha256":configured.direct_aggregate_sha256,
   "direct_task_card_sha256":configured.direct_task_card_sha256,"direct_config_file_sha256":configured.direct_config_file_sha256,
   "direct_report_config_sha256":configured.direct_report_config_sha256,"generator_sha256":configured.preparer_sha256,
   "generator_git_sha":configured.preparer_commit,"execution_git_sha":"f"*40,
   "preparation_request_config_sha256":configured.preparation_request_config_sha256,
   "steward_authorized_config_sha256":configured.steward_authorized_config_sha256,
   "bindings":[asdict(x)|{"records":1600} for x in bindings]}
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/"manifest.json"; path.write_text(json.dumps(manifest)); configured=replace(configured,fit_label_manifest_path=str(path))
   m._validate_fit_manifest(configured)
   for key,value in (("held_ids_present",True),("text_present",True),("label_free_manifest_sha256","0"*64),
                     ("direct_aggregate_sha256","0"*64),("steward_task_card_sha256","0"*64),
                     ("steward_authorized_config_sha256","1"*64)):
    bad=dict(manifest); bad[key]=value; path.write_text(json.dumps(bad))
    with self.assertRaises(KURELDSOOFError): m._validate_fit_manifest(configured)
 def test_direct_gate_requires_completed_failed_exact_bound_report(self):
  c=self.config()
  good={"schema_version":"mal2026-kure-phase1-direct-oof-aggregate-v1","status":"completed","mode":"full_oof",
        "run_id":"kure-phase1-direct-oof-v1-20260803-001","records":2000,"folds":5,"method":"coral-natural-phase1-direct",
        "source_method":"coral-natural","training_performed":False,"calibration_performed":False,"selection_performed":False,
        "automatic_stage6_deployment_eligible":False,"protected_output":"exact_r0","config_sha256":"c"*64,
        "r0_oof_prediction_sha256":c.r0_oof_prediction_sha256,"task_card_sha256":"a"*64,"config_file_sha256":"b"*64,
        "source_stage3_config_sha256":c.source_stage3_config_sha256,"source_stage3_report_config_sha256":c.source_stage3_report_config_sha256,
        "source_stage3_aggregate_sha256":c.source_stage3_aggregate_sha256,"fold_manifest_sha256":c.fold_manifest_sha256,"fold_rows_sha256":c.fold_rows_sha256,
        "common_stage3_promotion_gate_passed":False,"common_stage3_promotion_gate":{"eligible":False},"validation_rows_loaded":False,"average_target_used":False}
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/"direct.json"; path.write_text(json.dumps(good)); bound=replace(c,direct_aggregate_path=str(path),direct_task_card_sha256="a"*64,direct_config_file_sha256="b"*64,direct_report_config_sha256="c"*64)
   m._validate_direct_failure(bound)
   for key,value in (("status","running"),("records",1999),("method","bad"),("training_performed",True),
                     ("automatic_stage6_deployment_eligible",True),("common_stage3_promotion_gate_passed",True),("r0_oof_prediction_sha256","0"*64)):
    bad=dict(good); bad[key]=value; path.write_text(json.dumps(bad))
    with self.assertRaises(KURELDSOOFError): m._validate_direct_failure(bound)
 def test_safe_dependency_path_excludes_gold_until_aggregate_predictions(self):
  safe=inspect.getsource(KURELDSOOFConfig.validate_safe_dependencies)
  for name in ("train_path","fold_manifest_path","fold_rows_path","r0_oof_prediction_path"): self.assertNotIn(f"self.{name}",safe)
  aggregate=inspect.getsource(m.aggregate); self.assertLess(aggregate.index("predictions.update"),aggregate.index("verify_post_prediction_gold_dependencies"))
  self.assertLess(aggregate.index("predictions.update"),aggregate.index("load_raw_axis_gold"))
 def test_stage3_lineage_and_seed_contract(self):
  c=self.config(); source=m._source(c)
  self.assertEqual((source.backbone.model_id,source.backbone.model_revision),("nlpai-lab/KURE-v1","d14c8a9423946e268a0c9952fecf3a7aabd73bd9"))
  self.assertEqual((source.backbone.lora_r,source.backbone.lora_alpha,source.backbone.lora_dropout),(16,32,.05))
  self.assertEqual((c.epochs,c.learning_rate,c.weight_decay,c.batch_size,c.gradient_accumulation_steps,c.max_length),(6,5e-5,.01,20,2,1536))
 def test_method_neutral_gate_and_exact_coverage_order(self):
  source=inspect.getsource(m.aggregate); self.assertIn('raw_decision["candidate_metrics"]',source); self.assertIn('pop("coral_natural_metrics")',source)
  self.assertLess(source.index('need(len(predictions)==2000'),source.index('promotion_gate(')); self.assertIn('"automatic_stage6_deployment_eligible":False',source)
 def test_resume_policy_rejects_complete_aggregate_and_partial_fold(self):
  c=self.config()
  with tempfile.TemporaryDirectory() as td:
   cc=replace(c,output_root=td,restricted_output_root=str(Path(td)/"restricted"))
   self.assertEqual(m.fold_status(cc,0),"missing")
   Path(td,"outer-00.json").write_text("{}")
   with self.assertRaisesRegex(KURELDSOOFError,"partial fold"): m.fold_status(cc,0)
   Path(td,"outer-00.json").unlink(); Path(td,"aggregate.json").write_text("{}")
   with self.assertRaisesRegex(KURELDSOOFError,"rerun is forbidden"): m.fold_status(cc,0)
  with tempfile.TemporaryDirectory() as td:
   cc=replace(c,output_root=td,restricted_output_root=str(Path(td)/"restricted")); aggregate=Path(td)/"aggregate.json"
   aggregate.symlink_to(Path(td)/"missing-target")
   with self.assertRaisesRegex(KURELDSOOFError,"symlink"): m.fold_status(cc,0)
  with tempfile.TemporaryDirectory() as td:
   cc=replace(c,output_root=td,restricted_output_root=str(Path(td)/"restricted")); partial=Path(cc.restricted_output_root)/"outer-00"/m.METHOD/"content"/"lora"
   partial.mkdir(parents=True)
   with self.assertRaisesRegex(KURELDSOOFError,"recovery amendment"): m.fold_status(cc,0)
 def test_fold_membership_swap_is_dynamically_rejected(self):
  expected={f"id-{i}" for i in range(400)}; predictions={key:(3.,3.,3.) for key in expected}
  m._validate_prediction_fold_membership(predictions,expected,0)
  swapped=dict(predictions); swapped.pop("id-0"); swapped["fold1-id"]=(3.,3.,3.)
  with self.assertRaisesRegex(KURELDSOOFError,"frozen fold 0"): m._validate_prediction_fold_membership(swapped,expected,0)
 def test_outer_public_contract_dynamic_tamper(self):
  c=self.config()
  with tempfile.NamedTemporaryFile() as f:
   private=Path(f.name); base={"schema_version":m.SCHEMA_VERSION,"status":"completed","mode":"outer_fold","nonselectable":False,
    "run_id":c.run_id,"outer_fold":0,"records":400,"method":m.METHOD,"training_performed":True,"calibration_performed":False,
    "selection_performed":False,"config_sha256":m.config_sha256(c),"config_file_sha256":m._config_file_sha256(),
    "task_card_sha256":c.task_card_sha256,"direct_aggregate_sha256":c.direct_aggregate_sha256,
    "r0_oof_prediction_sha256":c.r0_oof_prediction_sha256,"restricted_prediction_sha256":m.file_sha256(private),
    "validation_rows_loaded":False,"average_target_used":False}
   with mock.patch.object(m,"_validate_axis_disclosures"):
    m._validate_outer_public(base,c,0,private)
    for key,value in (("run_id","bad"),("outer_fold",1),("training_performed",False),("calibration_performed",True),
                      ("config_file_sha256","0"*64),("direct_aggregate_sha256","1"*64),("average_target_used",True)):
     bad=dict(base); bad[key]=value
     with self.assertRaises(KURELDSOOFError): m._validate_outer_public(bad,c,0,private)
 def test_pending_launcher_gate_precedes_fake_nvidia_and_artifacts(self):
  with tempfile.TemporaryDirectory() as td:
   sentinel=Path(td)/"called"; fake=Path(td)/"nvidia-smi"; fake.write_text(f"#!/bin/sh\ntouch {sentinel}\nexit 99\n"); fake.chmod(0o755)
   result=subprocess.run(["bash","scripts/run_kure_lds_oof_gpu0_3.sh","smoke"],env={**os.environ,"PATH":f"{td}:{os.environ['PATH']}"},capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0); self.assertFalse(sentinel.exists()); self.assertIn("not authorized",result.stderr)
 def test_launcher_dangling_output_sentinels_precede_nvidia_and_gpu_launch(self):
  original=Path("scripts/run_kure_lds_oof_gpu0_3.sh").read_text()
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); fake_python=root/"python"; gpu_launch=root/"gpu-launch"; nvidia_called=root/"nvidia-called"
   fake_python.write_text(f'''#!/bin/sh
case "$*" in
 *--check-authorization*) exit 0;;
 *--fold-status*) echo '{{"status": "missing"}}'; exit 0;;
 *--scheduler-state*) exit 0;;
 *) touch "{gpu_launch}"; exit 97;;
esac
'''); fake_python.chmod(0o755)
   fake_nvidia=root/"nvidia-smi"; fake_nvidia.write_text(f"#!/bin/sh\ntouch '{nvidia_called}'\nexit 98\n"); fake_nvidia.chmod(0o755)
   for mode,relative in (("smoke","smoke/outer-00.json"),("full","aggregate.json"),("full","logs/gpu-telemetry-full-test-attempt.csv"),
                        ("full","logs/full-gpu0-folds0-4.attempt-test-attempt.log")):
    run_root=root/f"run-{mode}-{relative.replace('/','-')}"; sentinel=run_root/relative; sentinel.parent.mkdir(parents=True); sentinel.symlink_to(root/"missing")
    text=original.replace('PYTHON="$ROOT/.venv-standard/bin/python"',f'PYTHON="{fake_python}"')
    text=text.replace('RUN_DIR="$ROOT/outputs/kure-coral-lds-oof-v1/kure-coral-lds-oof-v1-20260803-001"',f'RUN_DIR="{run_root}"')
    copied=Path("scripts")/f".test-lds-launcher-{os.getpid()}.sh"; copied.write_text(text); copied.chmod(0o700)
    env={**os.environ,"PATH":f"{root}:{os.environ['PATH']}","MAL2026_ATTEMPT_TAG":"test-attempt"}
    try: result=subprocess.run(["bash",str(copied),mode],env=env,capture_output=True,text=True)
    finally: copied.unlink(missing_ok=True)
    self.assertNotEqual(result.returncode,0); self.assertFalse(nvidia_called.exists()); self.assertFalse(gpu_launch.exists())
    self.assertRegex(result.stderr,"sentinel|rerun is forbidden|refusing to overwrite")
   prior_root=root/"prior-run"; prior=(prior_root/"logs/gpu-telemetry-full-prior-attempt.csv"); prior.parent.mkdir(parents=True); prior.write_text("immutable-prior")
   text=original.replace('PYTHON="$ROOT/.venv-standard/bin/python"',f'PYTHON="{fake_python}"')
   text=text.replace('RUN_DIR="$ROOT/outputs/kure-coral-lds-oof-v1/kure-coral-lds-oof-v1-20260803-001"',f'RUN_DIR="{prior_root}"')
   copied=Path("scripts")/f".test-lds-launcher-prior-{os.getpid()}.sh"; copied.write_text(text); copied.chmod(0o700)
   nvidia_called.unlink(missing_ok=True); gpu_launch.unlink(missing_ok=True)
   try: result=subprocess.run(["bash",str(copied),"full"],env={**os.environ,"PATH":f"{root}:{os.environ['PATH']}","MAL2026_ATTEMPT_TAG":"fresh-attempt"},capture_output=True,text=True)
   finally: copied.unlink(missing_ok=True)
   self.assertNotEqual(result.returncode,0); self.assertTrue(nvidia_called.exists()); self.assertFalse(gpu_launch.exists()); self.assertEqual(prior.read_text(),"immutable-prior")
   nvidia_called.unlink(missing_ok=True)
   copied=Path("scripts")/f".test-lds-launcher-tag-{os.getpid()}.sh"; copied.write_text(text); copied.chmod(0o700)
   try: result=subprocess.run(["bash",str(copied),"full"],env={**os.environ,"PATH":f"{root}:{os.environ['PATH']}","MAL2026_ATTEMPT_TAG":"../unsafe"},capture_output=True,text=True)
   finally: copied.unlink(missing_ok=True)
   self.assertNotEqual(result.returncode,0); self.assertIn("unsafe MAL2026_ATTEMPT_TAG",result.stderr); self.assertFalse(nvidia_called.exists())
 def test_launcher_telemetry_cleanup_and_training_ledger_contract(self):
  text=Path("scripts/run_kure_lds_oof_gpu0_3.sh").read_text(); self.assertIn("training_performed':True",text)
  self.assertIn("wait_tracked_pid",text); self.assertIn("pkill -TERM -P",text); self.assertIn("gpu-telemetry",text); self.assertIn("--fold-status",text)
  self.assertLess(text.index("--fold-status"),text.index('COORD_DIR="$ROOT/outputs/reservations'))
  self.assertIn("'attempt_tag':attempt",text); self.assertIn("'telemetry_summary_path'",text); self.assertIn("MAL2026_ATTEMPT_TAG",text)

if __name__=="__main__": unittest.main()
