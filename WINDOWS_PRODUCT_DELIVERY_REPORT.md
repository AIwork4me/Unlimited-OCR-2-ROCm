# Windows Radeon 产品交付报告（2026-11-01）

## 1. Windows Product Status

当前状态是**部分产品交付，完整官方 Benchmark 仍 BLOCKED**。用户可审阅并使用已准备的原始 AMD Windows Native 安装流程、离线 artifact 集、CPU 预检、数据集完整性收据和公共 runner 的 Windows 入口。两个全新环境 `.b/w1`、`.b/w2` 的离线安装和 pip check 已完成，安装阶段明确跳过 GPU doctor。两套环境各13-case Golden 均完成，合计26次完整合成 OCR 请求经主控重新读取原始产物独立验证 PASS。首次身份在环境检查阶段失败、0次 OCR；失败账本和异常完整保留，CPU 复现修正后使用另行审阅的新身份，没有重置旧账本。不能据此声明 Windows ALL_READY。

稳定路线固定为 Native Transformers、BF16、vision scoped GPU MATH、原生 DASH/Triton/SR-SWA 与原始权重。torch2.12.0+rocm7.14.1、torchvision0.27.0+rocm7.14.1、torchaudio2.11.0+rocm7.14.1 不漂移；原始 torch archive 为415d2dbd…，不是同版本的6a67d158…候选 wheel。详见 [Windows readiness](docs/windows/WINDOWS_NATIVE_READINESS.md)。

## 2. Native Inference Validation

历史真实 Windows11 / Ryzen AI MAX+395 / Radeon8060S gfx1151 上，单页英文、中文、混合文本、表格、简单算术，以及原生2/3/5页重复推理的 A–E 门禁 PASS；范围与原始记录见 [历史验证](docs/VALIDATION.md)。这些证据保留原身份，不能复制成当前提交、当前新环境的验证。

本次代码的最新 CPU 检查在 `5436f80233477cf53f0e5eda4406c27a9e09c035` 上为389 PASS、8 skipped；Ruff、format、Windows/Linux-target mypy、build、pip check、source archive 检查 PASS。Linux-target mypy 只是静态检查。物理推理对应 fresh clone 的 `3556e3867a086437a43a645fa907689f4dd25260`；后续5436只改 PowerShell 缓存入口与测试。两套新环境的设备/BF16/Triton、官方模型加载、DASH/SR-SWA、vision GPU MATH、HIP、13-case 内容/结构均 PASS。26次生成均原生 stop，全部 fixture CER=0。跨环境13对精确 token/bbox 及套内6对 cold/warm 严格比较均 PASS；这是本次配对观察，历史严格差异仍保留，不能推广为确定性已永久修复。详见 [本次基线](docs/windows/NATIVE_BASELINE_VALIDATION.md) 和 [脱敏收据](eval/windows/golden-validation.json)。授权上限仍为两套既有13-case suite、合计26次完整请求，且前置门禁失败即停止。

## 3. Official Benchmark Results

| 完整数据集 | 必须文档 / 页数 | 已尝试文档 / 已处理页 | 八项指标 | Overall |
| --- | --- | --- | --- | --- |
| MPDocBench public | 420 / 3135 | 0 / 0 | 全部 null | null |
| MarathonDocBench | 100 / 18272 | 0 / 0 | 全部 null | null |

两项完整 GPU campaign 均 NOT_AUTHORIZED / NOT_RUN，没有完整实测成绩。Windows CPU 对全部页面的 bytes、SHA256、完整解码、尺寸、GT、顺序和 scope digest 复核均 PASS，缺失/多余/错误为0；相同内容组3/19仅作信息记录，不删除合法页面。数据就绪不等于 Benchmark PASS。[公共 dataset 收据](eval/windows/dataset-readiness.json)、[campaign 状态收据](eval/windows/benchmark-status.json)、[MPDocBench 状态](docs/windows/MPDOCBENCH_FULL_RESULTS.md)、[Marathon 状态](docs/windows/MARATHON_FULL_RESULTS.md) 保留完整分母和 null。

## 4. Baidu Official Comparison

官方论文 MPDocBench Overall 原表精确值84.59，展示可写84.6；Marathon 为77.06。Windows Overall 未产生，因此相对官方差值均为 null。论文 page10 Tables1/2 已对保留的原始 PDF 作视觉核验，单项原表值见上述两份结果状态文档。[Baidu 原论文](https://huggingface.co/baidu/Unlimited-OCR-2/blob/main/Unlimited-OCR-2.pdf) 是发布方链接，模型/代码另固定在 [不可变 revision](https://huggingface.co/baidu/Unlimited-OCR-2/tree/9cf077c4b206e0e2a31879df937b0b0bcba3661c)。不得用四舍五入后的单元格重算论文 Overall。

协议严格等价尚未建立：生产 Native 仍受5图/8192-token 限制，使用增长缓存与默认 ngram0；官方 customized SGLang 使用10页窗口、KV-tail continuation、ngram35/window1024，以及39321/235929级别预算。PR #24 的公共 ngram 集成仍 OPEN，不能视作已合并默认。论文 MPDocBench433文档也不能替换 public420分母。详见 [Windows protocol](docs/windows/WINDOWS_BENCHMARK_PROTOCOL.md)。

## 5. Reliability and Performance

完整数据集文档成功率、OOM率、截断率、吞吐、GPU peak、进程 RSS、生成 tokens 和总耗时均为 null；0失败是0尝试，不是100%可靠性。合成回归为26/26完成，56次输入页使用，3396个生成 tokens，逐case同步耗时合计330.608秒；OOM/截断/HIP错误均0，另保留1次0-infer环境检查失败。两套分别166.618/163.990秒；allocator峰值 allocated 8.589 GiB / reserved 10.510 GiB，进程 peak working set 最大11.212 GiB，1Hz采样最低系统可用76.430 GiB。它们不是全量文档吞吐或独立显存用量。不得把安装时长、CPU dataset decode 或历史短文档速度当成本次完整 GPU 性能。

Windows runner 复用公共 ledger，增加 suspended-create / assign Job / resume、KILL_ON_JOB_CLOSE、后代清理和 accounting；普通超时、中断、异常结果、非零退出、不可读结果有真实 CPU fake-worker 回归。三项独立 P2 缺陷已修正并再审。仍存在硬终止发生在进程创建与 Job assignment 之间时的未归属 suspended child 窗口，不能宣称任意父进程死亡都可完全清理。LongPathsEnabled=0 保持不变；262字符的保留失败证明深目录仍有风险，应使用短的仓库内路径。APU GPU allocator 是共享内存分配，不能当成独立 VRAM 或系统总内存消耗。

## 6. Reproduction

所有命令从主仓库根运行。此次 fresh local clone `.worktrees/wi` 与已审代码 commit `3556e3867a086437a43a645fa907689f4dd25260` 一致。下面预检不创建环境、不安装、不调用 GPU；`$Venv` 必须未使用，本机 w1/w2 已存在，应换短的新名称。完整已测安装流程、[43-artifact SHA256 清单](eval/windows/wheelhouse-SHA256SUMS.json)及重新获取步骤见 [安装复现指南](docs/windows/WINDOWS_INSTALL_REPRODUCTION.md)。

```powershell
$Python310 = (& py -3.10 -c "import sys; print(sys.executable)")
$Source = Join-Path $PWD '.worktrees/wi'
$Venv = Join-Path $PWD '.b/w-new'
$Wheelhouse = Join-Path $PWD '.b/wh'
& (Join-Path $Source 'scripts/setup_windows_rocm7141.ps1') -Python $Python310 -VenvPath $Venv -Wheelhouse $Wheelhouse -Offline -PreflightOnly
```

原始9个 AMD archive 的 hash/tag/member path 已核验，w1/w2最长目标219字符。仅有这9个 archive 不构成完整离线依赖集；rocm/rocm-bootstrap 原始来源也是 AMD，不能只依赖 PyPI。新环境安装始终拒绝已有目录；缓存、临时目录、模型数据均在根目录内，环境变量仅进程级设置并恢复。模型 cache 显式使用 `.b/h`；执行 trust_remote_code 前复核固定 revision 的代码/config/tokenizer/weights。没有全局 driver/SDK/security/PATH/registry 修改。

## 7. GitHub Changes

当前交付是本地可审阅准备，没有新分支远程推送或新 PR。四个本地代码提交为 `4abba4f`（Job supervision 与 publication）、`c431027`（原始 wheel / fresh offline install）、`3556e3867a086437a43a645fa907689f4dd25260`（PowerShell 入口、冻结 GT 绑定），以及 `5436f80233477cf53f0e5eda4406c27a9e09c035`（继承 HF 缓存的仓库内隔离与恢复测试）。[PR 基线审计](docs/windows/PR_BASELINE_AUDIT.md) 记录所有指定 PR 的冻结状态。原目录42项 dirty 文件与 index、历史406项及全部既有 seals、旧模型/原始及候选 wheel 和已安装成员均已重新按字节验证保持不变；旧 Draft #28/#29 仍为原 head、未合并。原有 Draft #28/#29 未改动，README、Linux 推理/模型/评分代码与 Linux 结果未修改。

保留 initial349 PASS /8 skipped /1 publication ancestor failure。修复后只按选定 root 的相对路径跳过私有目录，并经审阅更新旧 source-fixture whitelist hash；不是删除失败或绕过扫描。CPU/构建证据在私有 checks-r2 / checks-r3，最终文档/JSON 仍需主控对最终提交运行公开范围和 source archive 安全门禁。

## 8. Remaining Blockers

当前 clean-env 合成 Golden 已通过，但它未验证真实长文档；生产5图/8192限制和公共 ngram/budget/completion 集成不足以完成完整官方协议。Windows scorer 的75项 Python 依赖 metadata 匹配并不代表可运行八指标：xelatex、Ghostscript CLI、latexmlc 缺失，官方源码仍有 POSIX/fork/SIGALRM 假设。既有 g05 scorer PASS 来自 WSL/Linux，不能改标 Windows PASS。

完整数据集 campaign 尚未获独立资源/时限/失败恢复授权；官方严格可比性未建立；通用新 Windows 主机获取整套 artifact 与编译/渲染 prerequisites 还需要独立复现。Native Windows SGLang 为 BLOCKED_WITH_ACTIONABLE_FIX / NOT_TESTED，存在 torch2.13/cu13/flash依赖冲突，不能用 --no-deps 或 generic SGLang 代替。见 [可行性报告](docs/windows/WINDOWS_SGLANG_FEASIBILITY.md)。因此“新用户完整官方 benchmark 并看到实际分差”的最终验收尚未达成。

## 9. Linux Codex Handoff

交付公共 ledger/scorer.METRICS 兼容的 Windows accounting、固定身份与 null 状态收据，无第二套 scorer。本次合成 Golden 脱敏收据 SHA256 为 `8d2fda03ad3e979ecf0f33e15b15a7f48ccc1cf1cf94f360b833769f52be4410`；原始推理/权重/GT仍私有。dataset summary SHA256 为 `541fa1fae3a987825958f794d9dac851670412d0178057ee2c90252862e80233`，与实际 Windows summary 字节一致。完整分母、缺失指标、未运行硬件测量和官方 delta 都显式保留；requested target profile 与真实 run 字段分开。

可复用 runner/schema/manifest 和 source archive 测试；独立 Linux CPU 官方 scorer 是后续 Windows predictions 的可选评分端，但任何私有 GT/预测转移都需另行授权与身份核验，本次没有传输。Linux GPU PASS 不转成 Windows PASS。原始文本、GT、图像、权重、token/tensor、主机路径和机器标识均不进入公共报告。详见 [handoff](docs/windows/WINDOWS_HANDOFF.md)。

## 10. Next Action

唯一建议：完成共同的 ngram/budget/completion 集成后，准备并独立审阅**一次有界、最多5页的真实文档验证 policy**，固定执行身份、实际 scorer 路径、时限/共享内存停止阈值、失败封存和证据位置，再取得明确 GPU 运行批准。此行动不授权39321-token、15+页或完整420/100文档 campaign；当前仍不执行这些任务。预先限定的候选范围与尚缺的身份门禁见 [真实文档准入准备](docs/windows/REAL_DOCUMENT_ADMISSION.md)。
