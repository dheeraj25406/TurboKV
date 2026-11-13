"""Unit tests for synthetic document and workload dataset generation."""

from turbokv.datasets.retrieval_dataset import (
    RetrievalDatasetGenerator,
    CALIBRATION_FACTS,
    HELDOUT_FACTS,
)


def test_retrieval_dataset_separation():
    # Calibration facts and held-out facts must be completely disjoint
    calib_set = {f[2] for f in CALIBRATION_FACTS}
    heldout_set = {f[2] for f in HELDOUT_FACTS}
    assert len(calib_set.intersection(heldout_set)) == 0


def test_workload_generator_build_item():
    gen = RetrievalDatasetGenerator(tokenizer=None)
    item = gen.build_item(
        item_id="test_1",
        fact_tuple=CALIBRATION_FACTS[0],
        target_tokens=500,
        position_fraction=0.5,
        split="calibration",
    )

    assert item.item_id == "test_1"
    assert item.ground_truth_answer == CALIBRATION_FACTS[0][2]
    assert item.position_bucket == "middle"
    assert item.split == "calibration"
    assert item.target_fact in item.full_prompt
    assert item.question in item.full_prompt


def test_workload_generator_positions():
    gen = RetrievalDatasetGenerator(tokenizer=None)
    items = gen.generate_suite(context_lengths=[1024], split="calibration", positions=[0.1, 0.5, 0.9])
    assert len(items) == 3
    buckets = [it.position_bucket for it in items]
    assert buckets == ["beginning", "middle", "end"]
