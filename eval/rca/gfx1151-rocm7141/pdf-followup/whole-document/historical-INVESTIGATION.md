# Unlimited-OCR-2：官方 Transformers 全档 PDF 实机追踪

结论：指定 PDF 的 15 页已在 Windows Radeon 8060S / gfx1151 上，通过一次官方原生 `infer_multi` 调用完整生成。保留现有 SAM/CLIP 局部 GPU MATH 适配时，输出 13,836 token、15 个非空页段，原生结束原因为 `stop`。未适配的独立对照实例在第一页 SAM 的第一次 SDPA 中首次出现 HIP 错误。这支持同一已确认的 AOTriton runtime/images 打包契约缺陷归因。此结论限于本机和本版本栈；OCR 准确率尚未验收。

本报告和所有 PDF、PNG、OCR 正文、原始日志均保留在本地忽略目录，未上传 GitHub。生产源码、官方模型源码和权重均未修改。

## 输入与真实调用链

- 输入：`<USERPROFILE>\Desktop\Unlimited-OCR-2-ROCm\Unlimited-OCR-2.pdf`，15 页，SHA-256 `1eb5885f4ddfa4ec6fbff375092377f30b5d6a5e034ec4936213f125bf2f9444`。
- 官方源码：[baidu/Unlimited-OCR README，固定提交 4d0bebab8714e0f3b0c811cd44bf422ddb49daae](https://github.com/baidu/Unlimited-OCR/blob/4d0bebab8714e0f3b0c811cd44bf422ddb49daae/README.md)。从其 PDF 示例按 AST 原样提取 `pdf_to_images`；其余 SGLang 客户端代码未执行。选择的推理后端为 README 同时记录的 Transformers `AutoModel` / `infer_multi`，并未运行 SGLang 服务。
- PDF 读取：原函数使用 PyMuPDF、300 DPI；本地渲染器固定 `pymupdf==1.27.2.2`，安装在单独 CPU 渲染 venv，未改变 GPU 模型环境。每页 PNG 实际尺寸 2481×3508。这里 CPU 仅执行 PDF 光栅化，模型仍在 GPU。
- 一次调用：15 张已哈希、按页排序的 PNG 全部传入同一个 `model.infer_multi(..., image_files=all_15_pages, ...)`。没有按 5 页分批或逐页生成。原生 `infer_multi` 使用 `crop_mode=False`，模型构建出 4,115 prompt token，其中 4,095 image token（15×273），15 个 1024×1024 全局视图。
- 跟踪边界：PDF 读取 → 15 次官方 `Image.open` → `_uocr_build_inputs` → `infer_multi/infer` → `_uocr_generate` → SAM/CLIP → 实际 ATen SDPA → 解码器 prefill / 原生 DASH → 自回归生成 → 原始和可读输出。
- 固定模型：`baidu/Unlimited-OCR-2` 提交 `9cf077c4b206e0e2a31879df937b0b0bcba3661c`。两个对照进程加载前分别校验 code/config/tokenizer/weights；运行后再次校验全部文件，均与锁文件一致。
- 固定栈：Python 3.10.20；torch 2.12.0+rocm7.14.1；torchvision 0.27.0+rocm7.14.1；torchaudio 2.11.0+rocm7.14.1；HIP 7.14.60850；Windows 11；gfx1151；驱动 32.0.22032.6002。

之前用 Pillow 直接打开 PDF 的失败，只能说明该封装入口缺少 PDF 读取步骤，不能据此判断模型不支持 PDF。本次已纠正该验证范围。

## 未适配对照：最早错误发生在哪里

独立诊断实例原样调用官方模型，未改变生产 loader 或移除其适配器。运行 ID `eaadf93f-28d4-4bb2-b575-04dce902596e`，24.141 秒后退出 1。

1. 15 页图像全部打开，输入构建完成。
2. 第一页 SAM 的 `deepencoder.py:836` 首次 SDPA，BF16 Q/K/V 均为 `[25,12,196,64]`，stride `[150528,12544,64,1]`，加性 mask `[25,12,196,196]`。
3. CPU profiler 记录实际 `aten::_scaled_dot_product_efficient_attention`。HIP 从调用前 0 变为调用后 1（invalid argument）；同步返回正常但错误仍保留。
4. 随后在 `deepencoder.py:890` 的 `window_unpartition(...).permute(...).contiguous()` 抛异常。这里是延迟报错位置。CLIP 未到达，DASH 调用为 0，第一故障发生在解码器计算前。

详见 [首次错误](unadapted/first-failure.json)、[完整进程清单](unadapted/manifest.json)、[按边界追踪](unadapted/trace.jsonl)、[stderr](unadapted.stderr.log)。所有失败和警告均保留。

## 保留现有 MATH 适配：全档运行结果

运行 ID `8c816946-cf2f-4b09-9ebd-7e7fb815c962`，退出 0；开始时间 2026-10-28 08:19:03 UTC，最后生成结束 08:40:44 UTC。

| 项目 | 实测 |
|---|---|
| 单次原生请求输入 | 全部 15 页 |
| SAM / CLIP 完成数 | 15 / 15 |
| 视觉 SDPA | 540 次；每次 HIP 前后均为 0 |
| 具名 profiler 观察 | SAM 与 CLIP 首次调用均为 MATH |
| 原生解码器 | 12 层均到达；12 次原生 DASH prefill，seq=4115，BF16 GPU，同步成功 |
| 最终结束 | `finish_reason=stop`；expected_pages=15；num_pages_out=15 |
| 输出 | 13,836 token；15 个非空页段；15 个 PAGE 标记；无末尾空页段 |
| 总耗时（包含本次加载/追踪） | 1,300.985 秒，约 21 分 41 秒 |
| 官方生成计时 / prefill | 1,276.409 秒 / 11.960 秒 |
| PyTorch GPU allocation 峰值 | 9,298,788,352 bytes，约 8.66 GiB |
| PyTorch GPU reservation 峰值 | 9,542,041,600 bytes，约 8.89 GiB |
| 进程 RSS 峰值 | 9,560,043,520 bytes，约 8.90 GiB |

APU GPU allocation/reservation 是 PyTorch 分配器指标，不等同于独立显存或总系统内存消耗。以上计时来自有追踪的单次实机运行，不是稳定吞吐基准。

使用原生 greedy 解码；输出预算 39,321 token（官方评测指南的 0.3×131,072）；max_length=131,072；原生 page guard 和 max_page_tokens=20,000 保持。未禁用 DASH/SR-SWA，未替换模型，未把模型移到 CPU。没有引入 SGLang 专用的生成算法。

原始输出 SHA-256：`2fde62d850f3e07cb914c792e818f9378cbea287fb80423ab6beec44130cf1c4`。

- [原始输出](math/raw.txt)
- [可读 Markdown](math/readable.md)
- [最终原始清单](math/manifest.json)
- [完整边界追踪](math/trace.jsonl)
- [生成 token ID](math/token-ids.json)
- [完整 stdout](math.stdout.log) / [stderr](math.stderr.log)

使用原 PDF 文本层作为额外的页序诊断参考：对每个输出页段和全部 15 页参考文本，计算归一化字符 5-gram 集合 Dice。15/15 页段的最高相似页均与对应输入页号一致，分数 0.677–0.994。该参考文本从未输入模型；这不是人工校对 ground truth，不是 CER/WER，也没有设定 OCR 准确率验收阈值。因此状态明确为 `INFERENCE_COMPLETED_NOT_ACCURACY_SCORED`。详见 [完整页序诊断](full-order-diagnostic.json)。

## 与已确认根因的关系

`0.13 images` 指 **AOTriton 预编译 GPU 内核资源包**，不指 PDF 页图或 OCR 输入图像。本栈 runtime 构建标识为 0.12.50 / `05f26c2d…`，而随 wheel 提供的 images 属于 0.13 / `6e00ef3e…`。旧 loader 寻找松散 `.aks2` 文件，现有 images 使用 ZIP 存储；先前 images-only A→B→A 和 HIP launch 追踪已证明这个精确打包契约缺陷。

本次运行后再次校验相关 5 个 DLL、8 个 images 文件和 runtime header，全部与先前 RCA 原始主环境记录一致。实际 PDF 调用链出现相同默认 efficient SDPA 的首次 HIP 错误，与既有因果证据相互印证。这里不是一次新的 images-only 因果 A/B；没有替换本次生产环境内核资源。

保留局部 GPU MATH 可以绕过这条不兼容的 fused-attention loader 路径，完整 PDF 已运行完成；永久打包修复仍由 ROCm/PyTorch Windows wheel 维护方处理。未验证重建后的官方修复 wheel，也未验证其他 GPU/OS 或训练、反向传播。现有适配器仍应保留。

## 来源字节审计与复核

只读独立审查认可全档方法，并核对了两个对照的全部 15 页顺序、相同模型源码/权重、首次 SDPA 错误、540 次清洁 SDPA、原生 DASH 和原始环境文件一致性。其审阅发生在生成结束前；最终 EOS、页数、token、输出哈希与环境不变性由运行后的 `finalize_evidence.py` 再次断言核验（退出 0）。

审查发现两个来源表述问题，已在 [source-byte-audit.json](source-byte-audit.json) 透明记录，未重写原始 render manifest 或原始快照：

1. 本地 README 拷贝比官方内容多一个末尾 LF；其实际 Git blob 为 `34054be…`。仅去掉该额外 LF 后得到另存的 canonical 快照，其 Git blob 正确匹配官方 `8782f6e14af3fc0b052e2bfc245dd88e33756084`。两份快照内 reader 函数 AST 完全相同。
2. `function_sha256=61a853…` 表示实际编译使用的规范 LF 源文本；保存的 CRLF 函数文件字节 SHA 为 `090a056…`。两个摘要均已另行记录并注明语义。

运行后汇总：[summary.json](summary.json)。复核脚本：[finalize_evidence.py](finalize_evidence.py)。此跟进未改变生产代码，未重新宣称全套 CPU CI；Git tracked tree 保持干净。原有 PR #3 未合并。
