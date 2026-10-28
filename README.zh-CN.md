# Unlimited-OCR-2-ROCm

在 AMD ROCm 上运行百度官方 Unlimited OCR 2：原生推理、长文档解析与可复现验证。
这是独立适配项目，并非百度或 AMD 官方产品。

第一阶段仅面向原生 Windows 11、Ryzen AI MAX+ 395、
Radeon 8060S（gfx1151）。保留官方 V2 权重、BF16、DASH、
`infer()` 和整份文档的原生 `infer_multi()`。

**实体 Windows 验证 A–E 全部通过。** 两个隔离环境均通过英文、中文、
混合文本、表格、简单算术和原生 2/3/5 页重复推理；规范化 CER 为0。
完整范围与证据见 [VALIDATION.md](docs/VALIDATION.md) 和
[最终报告](docs/FINAL_REPORT.md)。

指定版本的默认视觉 SDPA 在本机存在错误，因此仅对 SAM/CLIP 前向调用
使用有范围的 GPU MATH SDPA；解码器 DASH/Triton 和原生生成保持不变。
保留原始页标记，并单独记录可能出现的一个空尾部 PAGE 分隔符。

固定环境：CPython 3.10.20 x64；torch 2.12.0+rocm7.14.1；
torchvision 0.27.0+rocm7.14.1；torchaudio 2.11.0+rocm7.14.1；
ROCm 7.14.1（HIP 构建号 7.14.60850）；triton-windows 3.7.1.post27；
Transformers 4.52.1。不自动升级到 Transformers 5.x 或 Triton 3.8。

从仓库根目录运行，选择已安装的 Python 3.10。使用尚不存在且可写的短环境路径：
AMD wheel 的某个文件在完整路径达到 260 字符时会解压失败。
虚拟环境与缓存可放在深层仓库目录之外，无需修改注册表。若已安装 uv：

```powershell
$Python310 = (uv python find 3.10)
$VenvA = "C:\uocr\a"
$VenvB = "C:\uocr\b"
$CacheDir = "C:\uocr\cache"
.\scripts\reproduce_windows.ps1 -Python $Python310 -VenvPath $VenvA -CacheDir $CacheDir -OutputPath "eval/results/run-a"
.\scripts\reproduce_windows.ps1 -Python $Python310 -VenvPath $VenvB -CacheDir $CacheDir -OutputPath "eval/results/run-b" -BaselinePath "eval/results/run-a/reproduction.json"
```

没有 uv 时，把 `$Python310` 设置为 Python 3.10 的完整路径。
若 `C:\uocr` 不可写，选择其他可写的短目录；若输出目录已有结果，使用新目录。
脚本创建独立虚拟环境，安装指定 AMD 包及锁定依赖，执行真实 GPU 与
OCR 检查并保存 JSON 清单。不会修改全局 Python、驱动、安全功能或永久 PATH。
普通安装本项目包不会自动安装或替换 torch。

使用上面创建的第二个环境；显式指定其解释器，无需激活虚拟环境：

```powershell
$OcrPython = "C:\uocr\b\Scripts\python.exe"
& $OcrPython -m unlimited_ocr_2_rocm doctor
& $OcrPython -m unlimited_ocr_2_rocm single "C:\文档\测试 page.png" --cache-dir "C:\uocr\cache" --output "C:\OCR结果\单页"
& $OcrPython -m unlimited_ocr_2_rocm multi "第1页.png" "第2页.png" --cache-dir "C:\uocr\cache" --output "C:\OCR结果\多页"
```

可指定 `--cache-dir`、`--offline` 和 `--max-new-tokens`。
每次使用新的输出目录。保留原始输出、规范化文本、页标记、哈希、
耗时、GPU 分配/保留内存及进程/系统内存信息。
用户文档没有已知真值时，仅报告 INFERENCE_ONLY，不宣称识别正确。

模型与代码固定为
`9cf077c4b206e0e2a31879df937b0b0bcba3661c`；
执行前校验代码、配置、分词器和权重哈希。导入和构建包不会下载模型。
不执行上游使用 eval 的生成坐标可视化函数；模型原始输出仍应视为不可信内容。

五个门槛全部通过才能宣称第一阶段 PASS：

- A：环境身份与真实 BF16 GPU 运算。
- B：官方模型加载及必要原生内核。
- C：英文、中文、混合文本和表格的识别正确性。
- D：原生多页推理、内容归属及页序正确。
- E：同一实体机器上的第二个干净环境重跑。

APU 使用共享系统内存；PyTorch 可见 GPU 预算不等于独立显存。
未验证 Linux、WSL、Docker、W7900、gfx1100、gfx1200 或其他设备。
CPU CI 无需下载模型，也不能替代实体 GPU 验证。

完整说明见 [英文 README](README.md)、
[复现步骤](docs/REPRODUCIBILITY.md)、
[上游审计](docs/UPSTREAM_AUDIT.md)、
[Windows 兼容性](docs/WINDOWS_COMPATIBILITY.md)、
[安全说明](docs/SECURITY.md) 和 [限制](docs/LIMITATIONS.md)。
本阶段不发布 PyPI、GitHub Release 或稳定版本，不自动合并 PR。
封装代码使用 MIT；上游模型和第三方源码许可见 [NOTICE](NOTICE)。
