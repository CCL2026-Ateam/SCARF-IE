# Multilayer Relation Extra OPS Summary

Base pipeline: strict3 prune_unseen -> entity OPS -> entity all + relation all augment -> final OPS(entity=0.75, relation=0.95) -> second entity all + relation all augment.
Extra #1 uses entity+relation OPS. Later layers use relation-only OPS: entity_policy=keep_all, apply_entity_fix=false, relation_policy=drop_threshold, relation_drop_threshold=0.95, apply_relation_fix=false.

## holdout_08_09

Best total: relation-only extra OPS #6 = 0.469976.

| stage | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN | d_total prev | d_RE prev | d_rel TP | d_rel FP | cum_total | cum_rel TP | cum_rel FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| second augment baseline | 0.466282 | 0.700104 | 0.310400 | 987/604/263 | 169/493/280 | base | base | base | base | +0.000000 | +0 | +0 |
| entity+relation extra OPS #1 | 0.468219 | 0.701611 | 0.312625 | 986/595/264 | 168/477/281 | +0.001938 | +0.002224 | -1 | -16 | +0.001938 | -1 | -16 |
| relation-only extra OPS #2 | 0.468424 | 0.701380 | 0.313120 | 986/596/264 | 167/468/282 | +0.000205 | +0.000495 | -1 | -9 | +0.002142 | -2 | -25 |
| relation-only extra OPS #3 | 0.468811 | 0.701380 | 0.313765 | 986/596/264 | 167/465/282 | +0.000387 | +0.000645 | +0 | -3 | +0.002530 | -2 | -28 |
| relation-only extra OPS #4 | 0.469711 | 0.701380 | 0.315265 | 986/596/264 | 167/459/282 | +0.000900 | +0.001500 | +0 | -6 | +0.003430 | -2 | -34 |
| relation-only extra OPS #5 | 0.469844 | 0.701380 | 0.315486 | 986/596/264 | 167/458/282 | +0.000132 | +0.000220 | +0 | -1 | +0.003562 | -2 | -35 |
| relation-only extra OPS #6 | 0.469976 | 0.701380 | 0.315707 | 986/596/264 | 167/457/282 | +0.000133 | +0.000221 | +0 | -1 | +0.003695 | -2 | -36 |

Output dirs:
- entity+relation extra OPS #1: `C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f\work\test_b_validation_scaling\outputs\entity_ops_then_all_relation_layer\extra_ops1_after_second_layer_text_label_holdout_08_09`
- relation-only extra OPS #2: `C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f\work\test_b_validation_scaling\outputs\entity_ops_then_all_relation_layer\relation_only_ops2_after_extra_ops1_text_label_holdout_08_09`
- relation-only extra OPS #3: `C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f\work\test_b_validation_scaling\outputs\entity_ops_then_all_relation_layer\relation_only_ops3_after_ops2_text_label_holdout_08_09`
- relation-only extra OPS #4: `C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f\work\test_b_validation_scaling\outputs\entity_ops_then_all_relation_layer\relation_only_ops4_after_ops3_text_label_holdout_08_09`
- relation-only extra OPS #5: `C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f\work\test_b_validation_scaling\outputs\entity_ops_then_all_relation_layer\relation_only_ops5_after_ops4_text_label_holdout_08_09`
- relation-only extra OPS #6: `C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f\work\test_b_validation_scaling\outputs\entity_ops_then_all_relation_layer\relation_only_ops6_after_ops5_text_label_holdout_08_09`

## 5fold_0_1_4_6_7

Best total: relation-only extra OPS #2 = 0.479410.

| stage | total | NER | RE | entity TP/FP/FN | relation TP/FP/FN | d_total prev | d_RE prev | d_rel TP | d_rel FP | cum_total | cum_rel TP | cum_rel FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| second augment baseline | 0.476288 | 0.711983 | 0.319158 | 2457/1426/612 | 429/1263/640 | base | base | base | base | +0.000000 | +0 | +0 |
| entity+relation extra OPS #1 | 0.478122 | 0.711812 | 0.322328 | 2448/1408/621 | 425/1200/644 | +0.001834 | +0.003171 | -4 | -63 | +0.001834 | -4 | -63 |
| relation-only extra OPS #2 | 0.479410 | 0.711812 | 0.324476 | 2448/1408/621 | 425/1178/644 | +0.001289 | +0.002148 | +0 | -22 | +0.003122 | -4 | -85 |
| relation-only extra OPS #3 | 0.478669 | 0.711812 | 0.323241 | 2448/1408/621 | 420/1156/649 | -0.000741 | -0.001235 | -5 | -22 | +0.002382 | -9 | -107 |

Output dirs:
- entity+relation extra OPS #1: `C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f\work\test_b_validation_scaling\outputs\entity_ops_then_all_relation_layer\extra_ops1_after_second_layer_text_label_5fold`
- relation-only extra OPS #2: `C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f\work\test_b_validation_scaling\outputs\entity_ops_then_all_relation_layer\relation_only_ops2_after_extra_ops1_text_label_5fold`
- relation-only extra OPS #3: `C:\Users\Zzl410410\Documents\Codex\2026-06-24\019ef53b-8c1b-7a92-ad59-c7c456388b6f\work\test_b_validation_scaling\outputs\entity_ops_then_all_relation_layer\relation_only_ops3_after_ops2_text_label_5fold`

## Readout

- holdout_08_09 keeps improving through relation-only #6, but #5 and #6 each remove only one extra relation FP and add about +0.00013 total; this is effectively a saturation zone.
- 5fold_0_1_4_6_7 peaks at relation-only #2: relation-only #3 drops 5 relation TP, so total falls despite another 22 relation FP removed.
- Practical default remains entity+relation extra OPS #1, then one relation-only extra OPS layer (#2). For holdout-only tuning, #6 is currently best, but it is not robustly supported by 5fold.
