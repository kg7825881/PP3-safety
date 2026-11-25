# Days 3–4 — PPE Dataset, Fine-Tuning and Evaluation

**Goal:** teach a pretrained detector five PPE classes and measure honestly where it fails.

```
Construction-PPE ─▶ prepare-data ─▶ ppe5 (train/val/test) ─▶ train ─▶ best.pt ─▶ evaluate (test)
  12 classes         audit, remap,      678 / 200 / 120       MPS        │
                     dedupe, split                                       └▶ error_analysis (val)
```

## 1. Data
Ultralytics Construction-PPE (Nov 2026): 1,000 images, 14,863 boxes, 12 classes, CC BY 4.0.
Its README states it combines human-collected and **generated** images.

| Decision | Why |
|---|---|
| Keep 5 of 12 classes | Compliance rules need person, helmet, vest, no_helmet, no_vest; fewer classes, simpler problem |
| Carve 120 test images from train | Source has no test split; val picks the best epoch, so its score is optimistic |
| Group near-duplicates (dHash, ≤ 4 bits) | The same scene in train and test inflates scores; 2 train images matched val and were dropped |
| Map classes by **name** | A reordered label file would silently corrupt training if mapped by position |

Instances, train / val / test: person 1717/481/243, helmet 1107/294/173, vest 1284/340/191,
no_helmet 566/179/68, no_vest 386/130/50. No malformed labels found.
The exact file lists are in `benchmarks/dataset_audit.json`.

## 2. Training
Transfer learning from COCO-pretrained YOLO26n (2.39 M parameters), 640 px, batch 16, MPS.
100-epoch limit, early stopping patience 20. **1.99 hours; best epoch 80** by validation.

- Validation mAP50 plateaued around epoch 41 (0.829).
- Train box loss kept falling (0.872 → 0.831 from epoch 91 to 100) while validation mAP stayed flat:
  mild overfitting. More epochs would not help; more or better data, or resolution, might.
- MPS training is not bit-reproducible: PyTorch warns one op has no deterministic version.

## 3. Results at 640 px

| Class | Val P / R / mAP50 | **Test P / R / mAP50** |
|---|---|---|
| all | 0.935 / 0.761 / 0.831 | **0.927 / 0.829 / 0.888** |
| person | 0.951 / 0.911 / 0.954 | 0.978 / 0.955 / 0.975 |
| helmet | 0.936 / 0.791 / 0.859 | 0.954 / 0.821 / 0.926 |
| vest | 0.960 / 0.874 / 0.919 | 0.983 / 0.919 / 0.969 |
| no_helmet | 0.918 / 0.598 / 0.706 | 0.808 / 0.632 / 0.716 |
| no_vest | 0.911 / 0.632 / 0.715 | 0.911 / 0.819 / 0.855 |

P and R are at the confidence that maximizes mean F1. CPU and MPS evaluation agree exactly
to three decimals.

**Why is test better than validation?** Unusual, so it was checked. Test has fewer tiny objects
(17% under 16 px vs 24% in val; for violations 29% vs 37%). Nearest-train perceptual distances
are similar for both, so there is no sign of near-duplicate leakage. Hashing cannot rule out
subtler similarity, such as images generated from similar prompts. Quote both numbers.

Speed at 640 px on MPS is unchanged by fine-tuning (about 13–14 ms per frame).

## 4. Error analysis (validation only)
Recall at confidence 0.25, IoU ≥ 0.5, by object size (square root of box area at 640 px scale):

| Class | < 16 px | 16–32 px | ≥ 32 px |
|---|---|---|---|
| no_helmet | 29/81 (36%) | 15/23 (65%) | 71/75 (95%) |
| no_vest | 2/32 (6%) | 7/18 (39%) | 75/80 (94%) |
| helmet | 69/121 (57%) | 29/32 (91%) | 141/141 (100%) |

**Size explains most misses.** Lowering the threshold barely helps: no_helmet recall only moves
from 0.642 (conf 0.25) to 0.654 (0.10) while precision falls from 0.906 to 0.741.

Feeding the same weights larger input (960 px, **no retraining**) at confidence 0.25:

| Class | 640 recall / precision | 960 recall / precision |
|---|---|---|
| no_helmet | 0.642 / 0.906 | 0.693 / 0.827 |
| no_vest | 0.646 / 0.923 | 0.715 / 0.877 |
| helmet | 0.813 / 0.895 | 0.891 / 0.916 |

More pixels help recall, with lower precision on violations. Training at 960 px is the next
experiment to try. See `benchmarks/finetuned_960.md` for its latency cost.

## 5. Limitations
1. Real-site accuracy is unproven; the images are partly generated.
2. Only 68 no_helmet and 50 no_vest test instances, so those scores are rough.
3. One training run, one seed: run-to-run variance is unknown.
4. `conf_threshold: 0.25` in `configs/ppe.yaml` was chosen on validation.
5. A far-away worker's violation is often missed. Place cameras so heads are at least ~32 px
   (at 640 px input scale), and let tracking combine evidence across frames (Day 7–8).

## 6. Code walkthrough
- `dataset.py`: `parse_labels` validates boxes; `class_mapping` remaps by name; `dhash` encodes
  each image in 64 bits (is each pixel brighter than its neighbor?); `group_duplicates` uses
  union-find; `assign_splits` assigns whole groups; `prepare` writes the dataset and audit.
- `train.py`: the Ultralytics trainer with our settings; returns the best-validation weights.
- `evaluate.py`: per-class metrics to JSON and Markdown.
- `scripts/error_analysis.py`: matches predictions to labels greedily, most confident first.

## 7. Reproduce
Download `construction-ppe.zip` from Ultralytics' assets release (SHA-256 `e5817aa9…c1232`), extract to
`data/datasets/construction-ppe-v1/`, then:

```bash
.venv/bin/ppe-guard prepare-data
.venv/bin/ppe-guard train
cp runs/train/ppe_yolo26n/weights/best.pt weights/ppe_yolo26n.pt
.venv/bin/ppe-guard evaluate
.venv/bin/python scripts/error_analysis.py
```

## 8. Interview questions
**Q: Precision or recall for safety?** A missed violation can injure someone, so recall matters
most; but too many false alarms get the system ignored. We keep detection sensitive and let
time-based rules filter one-frame errors.

**Q: What is mAP50 vs mAP50-95?** AP is the area under a class's precision-recall curve; mAP averages
over classes. mAP50 counts a match at IoU ≥ 0.5. mAP50-95 averages IoU 0.50 to 0.95, so it also
rewards tight boxes.

**Q: Why not report validation accuracy?** It chose the best epoch, so it is biased upward. Test
influenced neither weights nor epoch choice.

**Q: Your test beat validation. Isn't that leakage?** I checked: fewer tiny objects in test, and
similar distance to train as validation. I report both and state what hashing can't detect.

**Q: Why transfer learning?** With 678 images, training from scratch would overfit. COCO weights already
know edges, shapes and people; person reached 0.93 mAP50 after one short epoch.

**Q: How would you improve it?** Train at 960 px (errors are mostly small objects), add real site
images of violations, and repeat with several seeds to measure variance.

**Q: Why learn no_helmet instead of inferring it from a missing helmet?** A missing detection may just
be occlusion. An explicit class is positive evidence of a bare head. The rules can use both.
