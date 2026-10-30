# Bbox correctness analysis

PR #15 exact-parity FAIL is preserved. These are descriptive measurements, with no relaxed acceptance.

| Case | Changed coordinates | Max absolute delta | Minimum valid IoU | Invalid boxes |
| --- | ---: | ---: | ---: | ---: |
| english | 4 | 2 | 0.966666667 | 0 |
| chinese | 0 | 0 | 1.000000000 | 0 |
| mixed | 10 | 3 | 0.941071429 | 0 |
| table | 0 | 0 | 1.000000000 | 0 |
| math | 9 | 3 | 0.925397957 | 0 |
| repeat_unicode_recovery | 4 | 2 | 0.966666667 | 0 |
| pages_2 | 1 | 1 | 0.992424242 | 0 |
| pages_3 | 5 | 2 | 0.970822567 | 0 |
| pages_5 | 4 | 1 | 0.986111111 | 0 |
| table_2_pages | 4 | 3 | 0.995661605 | 0 |

All ten pairs preserve exact non-bbox bytes and canonical page/block/label/content order. Full token equality fails in eight pairs. Per-box labels, signed deltas, coordinates and IoU are retained in bbox-correctness.json. No box is reversed, zero-area or outside 0..999. Boundary contact alone cannot establish clipping against the source image; annotated text geometry is unavailable.

The pinned model renderer maps coordinates using int(coordinate / 999 * image_axis_pixels). The Markdown converter separately clamps image crop references to 0..1000. A delta of 3 therefore means 3/999 of the corresponding page dimension, before integer rounding. This can move a reconstructed edge, change overlap/containment or a crop boundary. High IoU does not establish text containment or downstream safety.

No semantic acceptance threshold is proposed from these observations. A new protocol would need independently annotated synthetic geometry, predeclared containment/crop/order criteria and boundary adversarial cases, followed by a new experiment. BF16 operator atol/rtol 0.03/0.03 provides no bbox acceptance rule.
