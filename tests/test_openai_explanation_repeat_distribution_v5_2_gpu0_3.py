"""GPU-free contract tests for the migrated v5.2 launcher."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("repeat_v5_2_gpu0_3", ROOT / "scripts/run_openai_explanation_repeat_distribution_v5_2_gpu0_3.py")
RUNNER = importlib.util.module_from_spec(SPEC); assert SPEC and SPEC.loader; SPEC.loader.exec_module(RUNNER)


def test_migrated_config_locks_gpu_scope_parallel_and_full_envelope() -> None:
    smoke = RUNNER.config(3, [0]); full = RUNNER.config(2000, [0, 1, 2, 3])
    assert smoke["runtime"]["physical_gpus"] == [0]
    assert full["runtime"]["physical_gpus"] == [0, 1, 2, 3]
    assert full["runtime"]["parallel_requests_per_server"] == 4
    assert full["runtime"]["context_size"] // 4 == full["runtime"]["slot_context_size"] == 4096
    assert full["selection"]["split"] == "train" and full["selection"]["max_essays"] == 2000
    assert full["protocol"]["selection_artifact_permitted"] is False


def test_launcher_requires_preflight_and_smoke_gates_before_full() -> None:
    launcher = (ROOT / "scripts/run_openai_explanation_repeat_distribution_v5_2_gpu0_3.sh").read_text(encoding="utf-8")
    assert "MAL2026_V52_PREFLIGHT_REPORT" in launcher
    assert "MAL2026_V52_SMOKE_REPORT" in launcher
    assert "resolve_v5_2_gpu0_3_priority.py" in launcher
