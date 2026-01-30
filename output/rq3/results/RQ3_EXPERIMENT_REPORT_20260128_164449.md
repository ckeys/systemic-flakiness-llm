# RQ3 Experiment Report

Generated: 20260128_164449

Total Samples: 50

Number of Runs: 1

## Summary

| Method | BLEU-4 | CodeBLEU | Edit Sim. |
|--------|--------|----------|-----------|
| B1_zero_shot | 0.0674 | 0.1724 | 0.1760 |
| B2_individual | 0.0684 | 0.1592 | 0.1564 |
| B3_no_clustering | 0.0482 | 0.1501 | 0.1409 |
| Ours | 0.0598 | 0.1415 | 0.1417 |

## Method Comparison

### Ours vs B1 (Zero-Shot)
- BLEU-4 improvement: +-0.0076
- CodeBLEU improvement: +-0.0309

### Ours vs B2 (Individual)
- BLEU-4 improvement: +-0.0086
- CodeBLEU improvement: +-0.0177

## LLM Cost Summary

- Total API Calls: 285
- Total Tokens: 1,150,514
- Total Cost: $0.2088 USD

### Cost by Component

- B1_zero_shot_repair: $0.0220 (38 calls)
- B2_individual_diagnosis: $0.0266 (50 calls)
- B2_individual_repair: $0.0369 (50 calls)
- collective_diagnosis: $0.0487 (48 calls)
- Ours_collective_repair: $0.0746 (99 calls)