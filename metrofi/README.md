# MetroFi

Only two experiments are retained. Both use **node positional encoding (PE) + RRWP + no sampled features (nosf)**; the only main difference is whether real coordinates are provided.

| Model | Config | Checkpoint | Coordinates |
|---|---|---|:---:|
| Conditional | `configs/config_metrofi_cond.py` | `checkpoints/metrofi_cond_pe_nosf_epoch999.ckpt` | Yes |
| Unconditional | `configs/config_metrofi_uncond.py` | `checkpoints/metrofi_uncond_pe_nosf_epoch999.ckpt` | No |

## Official test results

Both models are evaluated on the same 943 test graphs with the same true mask and sampler settings.

| Model | MSE ↓ | MAE ↓ | Pooled Wasserstein ↓ | Pooled JS ↓ |
|---|---:|---:|---:|---:|
| Conditional | **0.0206** | **0.1050** | **0.0187** | **0.0219** |
| Unconditional | 0.0263 | 0.1194 | 0.0322 | 0.0276 |

**Conclusion: real coordinates help; the conditional model achieves 21.8% lower MSE than the unconditional model.**

Results: `metrofi/results/official_test/`

Reproduce:

```bash
bash metrofi/eval_official_test.sh conditional
bash metrofi/eval_official_test.sh unconditional
```
