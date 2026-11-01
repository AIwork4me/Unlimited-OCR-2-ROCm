# MarathonDocBench full Windows results — NOT RUN

**NOT AUTHORIZED / NOT RUN; no complete Windows benchmark score.** This file is a blocked campaign receipt, not evidence of a completed full evaluation.

| Accounting | Value |
| --- | ---: |
| Required documents | 100 |
| Attempted / successful / failed | 0 / 0 / 0 |
| NOT_RUN documents | 100 |
| Required / processed pages | 18272 / 0 |
| Success rate | null |
| Wall time, throughput, token count, peak allocator/RSS | null |
| Windows Overall | null |
| Official reference Overall | 77.06 |
| Windows minus official | null |

Zero failed attempts means no attempts, not 100% reliability. Full dataset bytes/hash/decode/order/GT qualification passed on Windows, but is not inference or scoring PASS. Unknown benchmark hardware measurements and run identities are null in [benchmark-status.json](../../eval/windows/benchmark-status.json); requested/historical target profile is separate.

| Metric | Windows full result | Baidu paper cell |
| --- | --- | ---: |
| TextEdit | null | 0.104 |
| T-TextEdit | null | 0.123 |
| TableTEDS | null | 74.81 |
| T-TableTEDS | null | 71.11 |
| FormulaCDM | null | 87.93 |
| ReadOrderEdit | null | 0.147 |
| FigureF1 | null | 71.79 |
| HeadingTEDS | null | 48.20 |

Paper cells for edit metrics are raw lower-is-better distances; other cells are higher-is-better percentages. Common scorer reports edit metrics as `100*(1-distance)`. The paper Overall is quoted directly: never recompute it from rounded cells. All eight Windows fields and Overall remain null.

Source: [Baidu Unlimited-OCR-2 paper, page10 Tables1/2](https://huggingface.co/baidu/Unlimited-OCR-2/blob/main/Unlimited-OCR-2.pdf), visually checked against the retained original paper fixture. For immutable model/source context see [pinned Baidu revision](https://huggingface.co/baidu/Unlimited-OCR-2/tree/9cf077c4b206e0e2a31879df937b0b0bcba3661c). The paper URL is a publisher link, not an immutable revision claim.

Strict official comparability is NOT ESTABLISHED. Native uses growing caches, five-image/8192-token production limits and default ngram0; official customized SGLang uses ten-page KV-tail windows, custom ngram35/window1024 and different budgets. Public MPDocBench420 differs from the paper433 denominator. Native Windows full eight-metric scorer is blocked by rendering tools/POSIX assumptions. Existing Linux/WSL scorer results do not certify Windows scoring.

Release gates: reviewed common ngram/budget/completion integration; Windows bounded real-document identity/resource/failure policy; qualified official scorer path; frozen full dataset/environment identities; explicit separate campaign authorization. No full campaign or unauthorized larger budget was executed. See [Windows protocol](WINDOWS_BENCHMARK_PROTOCOL.md) and [dataset readiness](DATASET_READINESS.md).
