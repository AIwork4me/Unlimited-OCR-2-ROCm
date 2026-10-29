# Windows ROCm / PyTorch 2.12 验证交接

日期：2026-10-29，Asia/Shanghai。

## 结论与范围

**FIX_VERIFIED。** 本地从冻结源码构建的 Windows cp310 wheel，在实体
Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151 上通过既有 BF16 forward SDPA 矩阵。

| 项目 | 原始基线 | 候选 |
| --- | ---: | ---: |
| 必需 fused 条件 | 21 FAIL | 21 FUSED_PASS |
| 显式 MATH 控制 | 4 MATH_PASS | 4 MATH_PASS |
| copy-only 控制 | 1 CONTROL_PASS | 1 CONTROL_PASS |
| 指定的可选 masked-FLASH 探针 | 1 UNSUPPORTED | 1 UNSUPPORTED |

全部 27 个 fixture SHA-256 一致，除输出路径外的参数一致。六组关键条件各运行
三个独立 Python 进程。所有数值检查在原始阈值下通过，failing/nonfinite 元素为 0；
HIP 检查点均为 0，copy 和 arithmetic 下游操作成功。

FLASH 与 EFFICIENT 的日志均记录成功的 `hipModuleLoadDataEx`、
`hipModuleGetFunction(..., attn_fwd)`、非零函数句柄、grid `(1,2,1)`、
block `(128,1,1)`、`hipSuccess`，以及 PAL 的 `kernel : attn_fwd`。
该结论没有使用隐式 MATH fallback 作为 fused 正确性的证明。

范围限定为本地构建的这个 artifact、Windows gfx1151、CPython 3.10、SDK 7.14.1、
驱动 32.0.22032.6002 和本轮 BF16 forward 矩阵。新发布的 AMD wheel、其他 dtype、
backward、其他硬件和完整 OCR 模型流程未验证；没有定位单个“修复提交”。

## 冻结身份与交付入口

| 对象 | 精确身份 |
| --- | --- |
| 现有 PR | <https://github.com/AIwork4me/Unlimited-OCR-2-ROCm/pull/4> |
| 分支 | `validation/pytorch212-current-windows-gfx1151` |
| PyTorch release/2.12 | `60dd50f143fd9b10ac23cde8cdd34be61232b139` |
| TheRock recipe | `6d34c9b6f3e752b8f718b0c81f3070df77ce3283` |
| AOTriton runtime | 0.14.50，`7a6963a7265d2cf0fc48c6a63464386134d32483` |
| 官方 image source signature | `c11b5b886a10a00f925f74133d38b3505e67ccd9` |
| 官方 gfx115x archive SHA-256 | `e4bda76fd6857a733cd7b4b3eb9ccb21c0f47b565ce304777a61bbb13d12ad6e` |
| wheel 文件名 | `torch-2.12.0+rocm7.14.1-cp310-cp310-win_amd64.whl` |
| wheel 大小 | 388,478,127 字节 |
| wheel SHA-256 | `6a67d15861538ee217afdccd1ccc09ff318d7f44bff29a10f74c0d51730e6e1e` |
| 诊断代码字节绑定提交 | `3e641d97af952d3fa70a3fbec223d6faed41b047` |
| 原始复现器来源 | RCA `6b643f51d4078c4bb841f6a48734cf2206090311` |

runtime 与 image 的 source SHA 是两个独立身份，不要求二者相等。应分别绑定到冻结
recipe 和 artifact 字节，再通过实体 GPU 验证兼容性。不能仅凭相同版本号认定正确。

原始基线和新候选的 torch 版本字符串同为 `2.12.0+rocm7.14.1`，实际源码和二进制
不同。安装和验收必须使用精确文件路径、wheel hash、source SHA 和加载身份。

## 本机目录与环境

| 路径 | 用途 |
| --- | --- |
| `C:/uocr` | PR #4 的工作树；后续 Git 操作显式指定此目录 |
| `C:/b/pytorch` | 冻结 PyTorch 源码；保留官方 HIPIFY 生成内容 |
| `C:/b/therock` | 冻结构建器 |
| `C:/b/be` | 构建环境，SDK 7.14.1 已安装并初始化 |
| `C:/b/old` | 原始基线环境；保留原始 torch/vision/audio 三件套 |
| `C:/b/new` | 全新候选环境；安装本次 wheel，不依赖旧 vision/audio |
| `C:/b/qa` | CPU 检查环境 |
| `C:/b/out/wheels` | 最终候选 wheel；不要与 `C:/b/wheels` 下旧同名 wheel 混淆 |
| `C:/b/wheels` | 已有 SDK 和历史 wheel，继续复用 |
| `C:/b/resume` | 原始日志、恢复收据、profile、配对分析和构建记录 |
| `C:/b/candidate` | 已完成的 27 进程候选矩阵原始输出 |
| `C:/b/pytorch/build` | 可复用的增量编译产物；无须重新 clean/build |

先查看 `build-after-psimd-process.json`：官方构建器最终退出 **0**，并报告
`Builds all completed`。`candidate-matrix-process.json` 记录完整矩阵执行时间和命令。
不要因为会话被中断便重启构建；先检查进程、日志末尾和 process JSON。

## 过程中的关键经验

### 1. 依赖图包含 setup、配置、预处理、编译和安装

`setup.py` 会检查所有声明子模块的标记文件，即使该模块不参与当前目标。
CUTLASS 必须恢复到 `da5e086dab31d63815acafdac9a9c5893b1c69e2`，但 CUDA-only
内核在 USE_CUDA=OFF 下不需要完整物化。可以使用精确 pin 的稀疏 checkout。

“不编译”不等于“不需要”：

- oneDNN 配置会访问 tests/examples 的 CMake 文件。
- 官方 HIPIFY 会读取 MSLK 的 `include/mslk/utils/tuning_cache.cuh`。
- Kineto profiling 关闭时，公共 profiler 头仍无条件包含 `ActivityType.h`。
- FP16 的安装脚本仍安装 PSIMD 的 `include/psimd.h`。

本次后两项分别造成实际编译诊断与 install 失败。补全现有固定提交后继续成功，
没有手写源代码修补。最终 37 个根 gitlink 和三个 setup/native 嵌套 pin 全部一致。
具体 SHA、物化方式和剩余非目标组件见 [源码恢复记录](docs/validation/SOURCE_RECOVERY.md)。

### 2. 网络失败不能归因于 AOTriton

GitHub 大传输出现 early EOF/TLS EOF；Python wheel 和 image 传输也曾截断。
使用了进程级现有代理、小批量精确 Git object fetch，以及认证 GitHub blob API。
Git 对象核对身份后才进入缓存；没有关闭 TLS 或更改 pin。

大文件续传必须核对 Content-Range、总长度和最终完整 SHA-256。每次请求前清除旧的
临时响应/头文件，避免零字节响应复用旧分块。本次一次分块恢复错误被完整 hash 检出，
错误 archive 未进入候选。最终仅接受与 CMake pin 和 GitHub release digest 一致的
231,642,586 字节 AOTriton archive，并再次验证七个解包 image 文件。

区分发布方给出的 digest 与本地计算的 digest。不要把自己算出的校验值写成官方签名。
SDK 已完成恢复后，应一直使用 `--no-install-rocm`，不要反复下载已验证文件。

### 3. 使用冻结官方构建流程并保留缓存

使用现有 `C:/b/be`、冻结 recipe、`--no-install-rocm`，仅构建 torch/gfx1151，
FLASH/EFFICIENT 保持 ON。Windows libuv 使用冻结 workflow 的 v1.47.0。
实际 269 个 HIP 编译 recipe 只有 `--offload-arch=gfx1151`。

`MAX_JOBS=8` 和 `CMAKE_BUILD_PARALLEL_LEVEL=8` 限制每级 Ninja；嵌套 AOTriton
构建与根构建会重叠，不代表整个系统只有八个编译进程。Ninja 各轮分母表示当轮剩余
动作，子构建也有自己的计数，不能当作统一的总进度百分比。

成功重试复用了主体编译产物。遇到缺头文件或安装输入时，先分类并补齐该 pin，
再增量继续。保存 stdout 和 stderr：本次 install 的 FAILED 行在 stdout，真正的
缺文件原因在 stderr。CMake 的负向探测也不能当作实际编译缺陷。

HIPIFY 会合法改变工作树。保持原始 HEAD，并记录生成文件/diff hash；不要为了
“看起来干净”而 reset 掉这些内容。fmt 的五处支持脚本变化只是可执行位元数据。

### 4. 基线必须冻结，候选身份要单独验证

原始 `repro_sdpa.py`、`reporting.py`、`collect_provenance.py` 字节未改。
候选 adapter 先检查复现器 hash，再只替换两个身份 AST 节点；fixture、seed、mask、
shape、reference、threshold、backend 和 HIP/downstream 逻辑均保持不变。

身份门禁不能只看 torch version 或 GPU DLL。CPU FP32 reference 也依赖 torch_cpu，
所以必须覆盖 torch_cpu、torch_python、C10 和 _C.pyd。当前门禁每个进程重新核对
全部 15 个 native wheel 成员、安装副本、加载路径/哈希和 HIP DLL；可选模块允许
未加载，但安装字节仍要核对。补充清单检查了全部 33 个已加载 torch/SDK 原生模块。

每个验收进程自行执行同一解释器的 `pip check`。旧文件中的 exit0 不能证明当前环境。
`--clear-error-nonacceptance` 及 argparse 缩写都被拒绝。候选不需要为满足历史基线门禁
而安装旧 torchvision/torchaudio。

本次诊断改动测量时尚未提交；每个进程记录的 producer hash 已逐一绑定到提交
`3e641d97…` 的 Git blob。后续任务应先提交并冻结已复核的诊断代码，再运行矩阵，
减少 checkout-context 与实际生产者字节的补充审计。

### 5. 正确性证据必须形成完整链条

能力标志、SDPA API 调用或 `synchronize()` 没抛异常，都不能证明 fused 成功。
原始基线就曾同步正常返回，但 HIP 错误状态为 1，函数句柄为空、block 为零。

验收链条是：精确 artifact → 完整加载身份 → 实际 FLASH/EFFICIENT aten 调度 →
模块加载和函数查找 → 有效 launch/PAL kernel → HIP 状态为零 → CPU FP32 数值比较
→ 下游 tensor 操作 → 三个独立进程重复 → 基线/候选 fixture 字节一致。

原始 BF16 规则为 `abs(actual-reference) <= 0.03 + 0.03*abs(reference)`。
候选 small FLASH 最大绝对误差约 0.00450444，small bias EFFICIENT 约 0.00648332，
vision EFFICIENT 约 0.00382197，所有 failing/nonfinite 元素为 0。
MATH 只作为显式控制，不能代替 fused 验收。

### 6. 独立复核和 Git blob 校验是交付的一部分

复核发现并通过 RED→GREEN 关闭了：pip-check 证据未绑定、错误清除选项缩写绕过、
native 身份覆盖不完整，以及 signed URL 的相对路径/引号边界脱敏遗漏。
当前 35 个候选门禁测试、30 个既有门禁测试通过；完整 CPU suite 为 121 passed、
两个 GPU opt-in marker skipped。实体 27 进程矩阵另行实际执行。

证据需要防止 Git 换行归一化和 `build/` 忽略规则漏提交。本仓库对
`eval/wheel-validation/**` 使用 `-text`；必要时显式强制加入证据目录，并检查
Git blob 的 SHA-256，不能只检查工作区文件。

本次暂存区独立复核通过了 710 个校验路径、709 条导出记录，以及八个 gzip 文件。
初始证据中 174 个文件保持原样，三个根元数据文件被最终结果替代，四份初始根元数据
另存于 `attempts/initial-blocked`。该目录内旧清单的路径属于冻结的 bc75 发布根目录，
不能当作当前目录的校验清单解释。

大日志使用 `gzip` 且固定 mtime=0，记录原始、脱敏后未压缩、发布压缩文件三个 hash。
脱敏包括用户目录，以及绝对 URL、GET 相对 URL、行首和引号内 signed query。
原始私密 URL/日志留在本机；wheel、SDK、DLL 载荷不提交 Git。

## 复现与恢复入口

现有结果已经完成，不需要重新构建或覆盖 `C:/b/candidate`。需要独立重测时，先核对
上述身份并使用新的输出目录：

```powershell
C:/b/new/Scripts/python.exe -m pip check
C:/b/qa/Scripts/python.exe C:/uocr/experiments/pytorch212_wheel/validate_wheel.py `
  --python C:/b/new/Scripts/python.exe `
  --identity-profile C:/b/resume/candidate-identity-profile.json `
  --output C:/b/candidate-independent-run
```

构建入口是已经记录环境、代理、源码缓存和 MSVC 设置的
`C:/b/resume/build-cached.cmd`。具体原始参数和每轮退出结果见 process JSON，
不要凭版本标签重新拉取 mutable branch。

## 证据索引与交付边界

- [最终决定](docs/validation/PYTORCH_212_CURRENT_WHEEL.md)
- [完整构建来源和失败分类](docs/validation/BUILD_PROVENANCE.md)
- [GPU 矩阵、数值和 fixture](docs/validation/GPU_TEST_MATRIX.md)
- [候选身份门禁](docs/validation/CANDIDATE_IDENTITY.md)
- [独立复核](docs/validation/REVIEW.md)
- [机器可读配对结果](eval/wheel-validation/windows-gfx1151/comparison.json)
- [候选原始报告](eval/wheel-validation/windows-gfx1151/candidate/)
- [构建/安装/身份/launch 证据](eval/wheel-validation/windows-gfx1151/continuation/)
- [完整 SHA256SUMS](eval/wheel-validation/windows-gfx1151/SHA256SUMS.txt)

交付目标始终是已有 PR #4。恢复任务时查询远程实际 head 和该 head 的检查：

```powershell
gh pr view 4 --repo AIwork4me/Unlimited-OCR-2-ROCm --json url,headRefOid,statusCheckRollup
git -C C:/uocr status --short
```

核对远程证据 blob 与 SHA256SUMS，并确认 Windows/Ubuntu CPU CI。CPU CI 不代表
实体 GPU 验收；实体结果来自上述固定硬件的原始报告。PR 最终 head 由远程查询得到，
不是本文件内自引用的 SHA。

任务边界保持不变：仅更新 PR #4，不创建 Issue/上游 PR，不修改生产 OCR 或 PR #3，
不自动 merge。上游建议是 TheRock 的 packaged-wheel 身份及实体 SDPA 回归门禁，
本次没有执行任何上游发布操作。
