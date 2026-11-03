| Task | Params | Seeds | Epochs | Test token acc, teacher-forced (%) | **Test exact match, greedy (%)** | OOD-length exact match, greedy (%) |
|---|---:|---:|---|---:|---:|---:|
| copy | 933,396 | 3 | 10, 8, 8 | 100.0 ± 0.0 | **99.8 ± 0.2** | 2.2 ± 2.6 |
| reverse | 933,396 | 3 | 14, 16, 10 | 100.0 ± 0.0 | **99.8 ± 0.1** | 1.2 ± 1.7 |
| addition | 931,086 | 3 | 19, 21, 33 | 100.0 ± 0.0 | **99.9 ± 0.1** | 0.0 ± 0.0 |

Mean ± std over seeds. Per-seed test exact match: copy [100.0, 99.6, 99.75]; reverse [99.9, 99.9, 99.6]; addition [99.95, 99.95, 99.8].

**Attention alignment** (seed 0, 500 test examples): fraction of output steps whose head-averaged last-layer cross-attention argmax lands on the source token being emitted: copy **100.0%**, reverse **100.0%**.

**Exact match by source length** (seed 0):

- copy: len 5: 100%, len 6: 100%, len 7: 100%, len 8: 100%, len 9: 100%, len 10: 100%, len 11: 100%, len 12: 100%, len 13: 3%, len 14: 0%, len 15: 0%, len 16: 0%
- reverse: len 5: 100%, len 6: 100%, len 7: 100%, len 8: 100%, len 9: 100%, len 10: 100%, len 11: 100%, len 12: 100%, len 13: 14%, len 14: 0%, len 15: 0%, len 16: 0%
- addition: len 5: 100%, len 6: 100%, len 7: 100%, len 9: 0%
