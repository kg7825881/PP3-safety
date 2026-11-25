"""Dataset preparation tests: label parsing, class mapping and split leakage."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from ppe_guard.dataset import (
    TARGET_CLASSES,
    assign_splits,
    class_mapping,
    group_duplicates,
    parse_labels,
    prepare,
)

NAMES = yaml.safe_load(Path("configs/construction-ppe.yaml").read_text())["names"]


def test_parse_labels_reports_bad_lines():
    text = "0 .5 .5 .2 .2\n0 .5 .5 .2 .2\n1 .5 .5 0 .2\n2 .5 .5\nx .5 .5 .1 .1\n"
    boxes, problems = parse_labels(text)
    assert len(boxes) == 1
    assert len(problems) == 4


def test_class_mapping_uses_names_not_positions():
    m = class_mapping(NAMES)
    assert m == {5: 0, 0: 1, 2: 2, 6: 3, 11: 4}


def test_class_mapping_rejects_missing_class():
    with pytest.raises(ValueError, match="no_vest"):
        class_mapping({k: v for k, v in NAMES.items() if v != "no_vest"})


def test_group_duplicates_is_transitive():
    g = group_duplicates([0b0, 0b1111, 0b11111111, 2**64 - 1], max_distance=4)
    assert g[0] == g[1] == g[2] != g[3]


def test_splits_never_share_a_group():
    groups = [0, 0, 1, 1, 2, 3, 4, 0]
    splits = ["train"] * 7 + ["val"]
    out = assign_splits(splits, groups, test_count=2, seed=1)
    assert out[0] == out[1] == "dropped"  # duplicates of a val image
    kept: dict[int, set[str]] = {}
    for group, split in zip(groups, out, strict=True):
        if split != "dropped":
            kept.setdefault(group, set()).add(split)
    assert all(len(s) == 1 for s in kept.values())
    assert out.count("test") >= 2


def test_prepare_end_to_end(tmp_path):
    src = tmp_path / "src"
    rng = np.random.default_rng(0)
    for split, n in (("train", 8), ("val", 2)):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        for i in range(n):
            img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(src / "images" / split / f"{i}.jpg"), img)
            # Person (kept -> 0) and gloves (dropped)
            (src / "labels" / split / f"{i}.txt").write_text("5 .5 .5 .4 .4\n1 .2 .2 .1 .1\n")
    out = tmp_path / "out"
    r = prepare(src, NAMES, out, tmp_path / "r.json", test_count=3, seed=0)
    assert r["images_per_split"] == {"train": 5, "test": 3, "val": 2}
    label = next((out / "labels" / "train").glob("*.txt")).read_text()
    assert label == "0 0.500000 0.500000 0.400000 0.400000\n"
    assert list(yaml.safe_load((out / "data.yaml").read_text())["names"].values()) == (
        TARGET_CLASSES
    )


def test_prepare_refuses_to_delete_unrelated_directory(tmp_path):
    src = tmp_path / "src"
    (src / "images" / "train").mkdir(parents=True)
    (src / "labels" / "train").mkdir(parents=True)
    cv2.imwrite(str(src / "images" / "train" / "a.jpg"), np.zeros((32, 32, 3), np.uint8))
    (src / "labels" / "train" / "a.txt").write_text("5 .5 .5 .4 .4\n")
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        prepare(src, NAMES, precious, tmp_path / "r.json")
    assert (precious / "keep.txt").exists()
