from importlib.util import module_from_spec, spec_from_file_location
from decimal import Decimal
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("openai_distribution_audit", ROOT / "scripts" / "audit_openai_distribution_metadata.py")
assert SPEC and SPEC.loader
AUDIT = module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_average_is_derived_from_components_outside_a_model() -> None:
    population = AUDIT.Population("synthetic")
    population.add(score={"content": Decimal("1"), "organization": Decimal("2"), "expression": Decimal("3")}, essay_length=250, task_group="opaque")
    report = AUDIT.population_report(population)
    assert report["mean_scores"]["average"] == 2.0
    assert report["external_average_band_counts"]["[2,3)"] == 1


def test_divergence_reports_zero_and_sparse_support() -> None:
    result = AUDIT.divergence(AUDIT.Counter({"a": 20}), AUDIT.Counter({"b": 1}), left_total=20, right_total=1)
    assert result["total_variation_distance"] == 1.0
    assert result["overlap_min_mass"] == 0.0
    assert result["zero_cells_left"] == 1
    assert result["zero_cells_right"] == 1
    assert result["sparse_cells_min_count_lt_20"] == 2
