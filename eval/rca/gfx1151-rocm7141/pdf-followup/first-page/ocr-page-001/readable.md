Baidu 百度

t

ERNIE

# Unlimited OCR Works 2

Pioneering the OCR-3.0 Era: From Long- to Unlimited-Horizon Parsing

Baidu ERNIE Team

## Abstract

Unlimited OCR ushers the era of long-horizon parsing, substantially mitigating the growth of working memory as document lengthens. However, as document length scales to hundreds of pages, the visual context grows substantially, sharply increasing the computational cost of each decoding step and limiting the scalability of ultra-long document parsing. Inspired by human book transcription via localized focus without re-scanning preceding pages, we introduce Unlimited OCR 2, a document parsing model that looks only at what it transcribes. Building on Unlimited OCR, we upgrade all the decoder attention modules to Sparse Reference Sliding Window Attention (SR-SWA), bounding active reference tokens to a fixed capacity and thus sustaining constant computational costs across arbitrarily long documents. By combining SR-SWA with Unlimited-Horizon Decoding Paradigm, Unlimited OCR 2 enables document parsing over virtually unlimited lengths without prohibitive memory overhead. At the hundred-page scale, Unlimited OCR 2 achieves $7.0 \times$ and $5.6 \times$ per-step speedups over Full Attention and R-SWA, respectively. Furthermore, it delivers state-of-the-art end-to-end performance on both MPDocBench and MarathonDocBench, our newly proposed benchmark targeting hundred-page document parsing. Ultimately, the capacity for uninterrupted, one-shot document parsing signifies a fundamental paradigm shift into the OCR 3.0 era.

![](page1_202_574_481_801.jpg)

![](page1_506_574_795_801.jpg)

Figure 1 | R-SWA vs SR-SWA. R-SWA computes attention over all visual tokens across pages, inducing extreme visual redundancy; SR-SWA sparsely attends to relevant visual regions, aligning machine decoding with human reading intuition.