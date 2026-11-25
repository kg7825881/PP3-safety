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

More pixels help recall, with lower precision on violations. That motivated the next section.

## 5. Retraining at 960 px
Same splits, COCO starting weights, seed and early stopping as the 640 run. **Batch was 8, not 16**
(memory), so batch size is a second difference. 3.65 hours; stopped at epoch 88, best epoch 68.

**Validation, where the choice was made.** The middle column is a control: the 640 weights fed
960 px input, which separates the effect of *input size* from the effect of *retraining*.

| | 640 weights @ 640 | 640 weights @ 960 | Retrained @ 960 |
|---|---|---|---|
| mAP50 / mAP50-95 | 0.831 / 0.618 | 0.867 / 0.630 | **0.875 / 0.663** |
| no_helmet mAP50 | 0.706 | 0.750 | **0.772** |
| no_vest mAP50 | 0.715 | 0.766 | **0.774** |

Recall at precision >= 0.90, with a 95% interval from 2,000 paired image resamples
(`scripts/compare_models.py`):

| Class | 640 → retrained 960 | Gain (95% CI) | Gain over the 960 control (95% CI) |
|---|---|---|---|
| person | 0.921 → 0.952 | +0.031 (0.011 to 0.059) | +0.010 (−0.008 to 0.032) |
| helmet | 0.810 → 0.881 | +0.071 (0.012 to 0.107) | −0.010 (−0.034 to 0.007) |
| vest | 0.894 → 0.935 | +0.041 (0.013 to 0.073) | −0.003 (−0.012 to 0.007) |
| no_helmet | 0.648 → 0.693 | +0.045 (0.000 to 0.095) | +0.034 (−0.012 to 0.082) |
| no_vest | 0.662 → 0.715 | +0.054 (0.013 to 0.111) | +0.008 (−0.029 to 0.074) |

**Reading it honestly:**
- 960 beats 640; no_helmet is borderline because its interval touches zero.
- **Most of the recall gain comes from larger input, not retraining.** Over the control, every interval
  includes zero, so recall evidence for retraining itself is inconclusive. Its clearest measured
  difference is tighter boxes (mAP50-95 0.630 → 0.663, no interval computed).
- The interval covers which images were sampled, not training randomness (one run, one seed).

Recall by size at confidence 0.25, 640 → retrained 960 (buckets at 640 px scale):

| Class | < 16 px | 16–32 px | ≥ 32 px |
|---|---|---|---|
| no_helmet | 29/81 → 43/81 | 15/23 → 15/23 | 71/75 → 72/75 |
| no_vest | 2/32 → 7/32 | 7/18 → 9/18 | 75/80 → 77/80 |
| helmet | 69/121 → 85/121 | 29/32 → 29/32 | 141/141 → 140/141 |

Tiny objects improved most, but at a fixed threshold precision on violations also fell
(no_helmet 0.906 → 0.839, no_vest 0.923 → 0.894), so use the equal-precision table to compare.

**Speed (MPS).** Two runs each, back to back in one session: 10.4 ms at 640 and 16.1 ms at 960,
repeatable within 0.1 ms. Earlier sessions measured about 13–14 and 23.6 ms, cause unknown; compare
only within a session. Full loop on the synthetic 960×540 clip: 72 → 57 FPS (other video sizes
untested). **Not tested:** long runs on this fanless laptop, or 960 px after CoreML export.

**Decision: 960 px**, made on validation. The test set was scored afterward:

| Class | Test at 640: P / R / mAP50 | **Test at 960: P / R / mAP50** |
|---|---|---|
| all | 0.927 / 0.829 / 0.888 | **0.949 / 0.891 / 0.930** |
| person | 0.978 / 0.955 / 0.975 | 0.983 / 0.975 / 0.984 |
| helmet | 0.954 / 0.821 / 0.926 | 0.949 / 0.890 / 0.939 |
| vest | 0.983 / 0.919 / 0.969 | 0.973 / 0.956 / 0.978 |
| no_helmet | 0.808 / 0.632 / 0.716 | 0.921 / 0.735 / 0.837 |
| no_vest | 0.911 / 0.819 / 0.855 | 0.918 / 0.898 / 0.913 |

Test mAP50-95: 0.693 → 0.740. With 68 no_helmet and 50 no_vest test objects, treat their gains as rough.

**Threshold.** `conf_threshold` stays 0.25. On validation it gives no_helmet recall/precision
0.726/0.839, versus about 0.693/0.90 at 0.39. It favors recall and relies on time-based rules to filter false
alarms (Day 7–8); revisit once those rules exist.

## 6. Limitations
1. Real-site accuracy is unproven; the images are partly generated.
2. Only 68 no_helmet and 50 no_vest test instances, so those scores are rough.
3. One training run per setting, one seed: run-to-run variance is unknown.
4. Validation picked the best epoch and the input size, so its scores are optimistic.
5. A far-away worker's violation is often missed. Place cameras so heads are at least ~32 px
   (at 640 px input scale), and let tracking combine evidence across frames (Day 7–8).

## 7. Code walkthrough
- `dataset.py`: `parse_labels` validates boxes; `class_mapping` remaps by name; `dhash` encodes
  each image in 64 bits (is each pixel brighter than its neighbor?); `group_duplicates` uses
  union-find; `assign_splits` assigns whole groups; `prepare` writes the dataset and audit.
- `train.py`: the Ultralytics trainer with our settings; returns the best-validation weights.
- `evaluate.py`: per-class metrics to JSON and Markdown.
- `scripts/error_analysis.py`: matches predictions to labels greedily, most confident first.

## 8. Reproduce
Download `construction-ppe.zip` from Ultralytics' assets release (SHA-256 `e5817aa9…c1232`), extract to
`data/datasets/construction-ppe-v1/`, then:

```bash
.venv/bin/ppe-guard prepare-data
.venv/bin/ppe-guard train
cp runs/train/ppe_yolo26n/weights/best.pt weights/ppe_yolo26n.pt
.venv/bin/ppe-guard train --imgsz 960 --batch 8 --name ppe_yolo26n_960
cp runs/train/ppe_yolo26n_960/weights/best.pt weights/ppe_yolo26n_960.pt
.venv/bin/ppe-guard evaluate -w weights/ppe_yolo26n_960.pt --imgsz 960 --split val --tag eval_val_960
.venv/bin/python scripts/error_analysis.py --weights weights/ppe_yolo26n_960.pt --imgsz 960 --tag errors_val_960_trained
.venv/bin/python scripts/compare_models.py
.venv/bin/python scripts/compare_models.py --a weights/ppe_yolo26n.pt 960 --tag compare_val_same_input
.venv/bin/ppe-guard evaluate -w weights/ppe_yolo26n_960.pt --imgsz 960 --tag eval_test_960
```

## 9. Interview questions
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

**Q: How would you improve it?** More real site images of violations first, because tiny
violations are still missed at 960 px. Then several seeds to measure variance, and a larger model.

**Q: Did retraining at 960 help, or just the bigger input?** I ran a control: the old weights at
960 px. Most of the recall gain came from input size; retraining's own recall gain was not
distinguishable from zero. The retrained model had the best validation mAP50 and mAP50-95 at the same
input size (so similar cost), and I chose it on that.

**Q: Why compare recall at equal precision?** At one fixed threshold, a model can gain recall just by
producing more boxes. Fixing precision at 0.90 asks: for the same false-alarm rate among alerts, who
finds more violations?

**Q: Why a paired bootstrap?** Resampling whole images keeps boxes from the same scene together, and scoring both
models on the same resample cancels out image difficulty. It doesn't capture training randomness.

**Q: Why learn no_helmet instead of inferring it from a missing helmet?** A missing detection may just
be occlusion. An explicit class is positive evidence of a bare head. The rules can use both.
