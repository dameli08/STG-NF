# RetailS Dataset Audit (STG-NF)

All findings below were verified **directly against the raw files** on
2026-07-30 using the RetailS root `/home/damelikassym/RetailS/RetailS`. Where an
earlier audit in the sibling `shopformer_repro` project disagrees, the raw data
wins (see §7).

## 1. Layout & counts

| Split | Pose dir | Files | GT dir | GT files |
|-------|----------|-------|--------|----------|
| train | `RetailS_train/pose/train` | 942 | — (normal only) | — |
| test_realworld | `RetailS_test_realworld/pose/test` | 47 | `.../gt/test_frame_mask` | 47 |
| test_staged | `RetailS_test_staged/pose/test` | 624 | `.../gt/test_frame_mask` | 624 |

**All 942 + 47 + 624 JSON files parse cleanly (0 invalid).**

## 2. Pose JSON structure (verified)

```
{
  "<person_id>": {            # OUTER key = person / track id
    "<frame_id>": {           # INNER key = frame id
      "keypoints": [x0,y0,c0, x1,y1,c1, ... , x16,y16,c16],   # length 51 = 17×3
      "scores": null          # per-frame scalar score is absent (null)
    },
    ...
  },
  ...
}
```

- **Keypoints**: native **COCO-17**, `(x, y, confidence)` interleaved. Converted
  internally to **COCO-18** (+neck) to match STG-NF's graph.
- **Per-joint confidence** lives in `keypoints[2::3]` (range `0.002 – 1.003`,
  mean `≈0.53`, no zeros). The top-level `scores` field is `null`, so a per-frame
  score is derived as the **mean of the joint confidences**.

## 3. Coordinates & resolution

`x ∈ [145, 1078]`, `y ∈ [15.6, 1919]` ⇒ RetailS is **portrait ≈ 1080 × 1920**
(phone-style camera). Config `dataset.vid_res = [1080, 1920]`. (Segment
normalization divides by the y-std afterwards, so the exact resolution mostly
cancels; it still matters for the confidence-weighting scale and for keeping the
pre-normalization magnitudes sane.)

## 4. Tracks, ordering, gaps

- **Persons per clip**: train median **45** (busy retail floor, max 143); test
  median **1** (max 5–6) — test clips are essentially single-subject.
- **Track length** (frames per person, train): median **65**, min 15, max 3186.
  **86.7%** of tracks ≥ 24 frames and **98%** ≥ 16 → `seg_len = 24` is viable.
- **Temporal gaps**: ~70% of tracks contain at least one missing-frame gap.
  STG-NF tolerates ≤ 2 missing frames per window (`is_seg_continuous`); we add an
  optional **linear interpolation of small (≤ `max_missing`) gaps** so more valid
  windows survive without fabricating long stretches of motion.
- Frame ids are integer strings; sequences are built in **ascending frame order**.

## 5. Ground truth

- GT masks are `.npy`, **frame-level**, values `{0, 1}` with **`1 = abnormal`
  (shoplifting)**. Length equals the clip's frame count and is `> max_frame_id`
  for **every** test clip (0 mismatches).
- Abnormal frame fraction: **realworld 0.317**, **staged 0.497** — both splits are
  well balanced, so AUC-ROC / AUC-PR are both meaningful.
- Pose file stem == GT file stem (e.g. `01_0222.json` ↔ `01_0222.npy`).

## 6. Compatibility with STG-NF assumptions

| STG-NF assumption | RetailS reality | Adaptation |
|-------------------|-----------------|------------|
| `person → frame → {keypoints, scores}` JSON | ✅ identical | reuse parser |
| COCO-17 keypoints | ✅ (51 floats) | reuse COCO18 conversion |
| Frame-level `.npy` GT, `1 = abnormal` | ✅ | drop ShanghaiTech `1-gt` inversion? No — see below |
| Per-frame `scores` present | ❌ `null` | derive from joint confidences |
| Filenames `*_tracked_person.json`, `scene_clip` | ❌ different naming | filename-agnostic adapter keyed by stem |
| Normal-only train, few-shot abnormal optional | ✅ normal-only train | unsupervised `R = 0` |
| Resolution 856×480 (ShanghaiTech) | ❌ 1080×1920 | configurable `vid_res` |

**GT polarity note.** ShanghaiTech masks encode `1 = anomaly` and the official
code inverts them (`1 - gt`) so that `1 = normal` and the *normality* score
(`−nll`) aligns with the label. RetailS masks also use `1 = abnormal`, so we keep
the **same inversion** to score normality, or equivalently evaluate the anomaly
score `nll` against the original `1 = abnormal` labels. Our evaluator computes
metrics on the **anomaly** score vs. the raw `1 = abnormal` GT, which is
polarity-unambiguous and matches the paper's AUC.

## 7. Correction of a prior audit

The sibling `shopformer_repro/docs/retails_dataset_audit.md` claims (a) the JSON
is nested `frame → track`, (b) 769/942 train files are invalid, and (c) GT masks
are track-index labels of length `max_track_id + 1`. **All three are false** on
the current data: nesting is `person → frame`, 0 files are invalid, and GT length
tracks the frame count. That audit was produced by a buggy reader and is
superseded by this document.

## 8. Splits & leakage

RetailS ships **official** train / test_realworld / test_staged splits at the file
level, so there is **no random splitting** and no need to split by
session/camera. Training consumes only `RetailS_train` (normal behaviour);
evaluation consumes the two test folders. Because splits are disjoint physical
folders, there is **no train/test leakage** through shared clips or people.
