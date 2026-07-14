# FireRedLID 推理加速设计

日期：2026-07-14
状态：已批准，进入实施计划

## 1. 目标

在不重训、不更换 checkpoint、不改变标签体系和 beam search 语义的前提下，为官方 FireRedLID 增加更高效的推理路径。

第一版交付可嵌入现有服务的 Python API，在线低延迟是主要场景，同时支持离线批处理。目标部署 GPU 是 NVIDIA RTX PRO 5000 或 L20；当前开发环境是 macOS，因此 TensorRT engine 的构建、CUDA `torch.compile` 验证和真实性能测试留到 Linux NVIDIA 阶段完成。

## 2. 已确认约束

- 保留官方 `model.pth.tar`、`cmvn.ark`、`dict.txt`。
- 保留官方 Conformer Encoder、Transformer Decoder、`beam_size=3`、`decode_max_len=2` 和 confidence 计算。
- 不将 Decoder 替换为 Linear head，不重训或蒸馏。
- 默认最大音频时长为 60 秒；更长输入在 FBank 前截断。
- Python API 不维护跨请求队列，也不等待新请求来组成 micro-batch。
- Python API 接受任意大小的逻辑 batch；物理执行可根据后端限制拆分。
- 第一版不交付 Triton 服务，但 API 边界必须能被后续 Triton Python wrapper 直接调用。
- 第一版精度为 FP32/FP16，不做 INT8、FP8 或更低精度量化。

## 3. 非目标

第一版不包含：

- Triton HTTP/gRPC 服务和 model repository。
- TensorRT-LLM Decoder 或固定展开的两步 beam search。
- FBank 算法替换、CPU worker 池或特征缓存服务。
- 服务级排队、限流、优先级、超时和跨请求 dynamic batching。
- 对 ASR2-AED TensorRT runtime 的无关重构。
- 承诺未经目标 GPU 实测的固定加速倍数或最大 batch。

## 4. 方案选择

采用以下路线：

1. 官方 eager PyTorch 作为兼容基线。
2. 增加只编译 Encoder 的 `torch.compile` 后端。
3. 将官方 Encoder 导出为动态 ONNX，再在 Linux NVIDIA 环境构建 FP16 TensorRT engine。
4. TensorRT 模式继续使用官方 PyTorch Decoder 和 beam search。

不选择第一版直接 TensorRT 化 Decoder，因为 Decoder 最多运行两步，且官方 `batch_beam_search()` 包含 Python 循环、动态 cache、`.item()` 提前退出和嵌套结果组装。第一版先用 profiler 验证 Decoder 的实际占比，再决定是否进入后续专项优化。

## 5. 架构

公开调用保持兼容：

```python
lid = FireRedLid.from_pretrained(model_dir, config)
results = lid.process(uttids, wav_inputs)
```

整体数据流：

```text
音频路径或 waveform
  -> 原版加载逻辑
  -> 最长 60 秒截断
  -> 原版 FBank 和 CMVN
  -> batch planner：拆分、可选分桶、padding
  -> EncoderBackend
  -> 原版 PyTorch Decoder 和 beam search
  -> 原版 tokenizer 与结果组装
```

Feature Frontend 与 EncoderBackend 在代码职责上独立，通过 `FeatureBatch` 数据契约连接：

```text
Feature Frontend：音频 -> 每条音频独立的已做 CMVN 的 [T_i, 80] 特征
Batch Planner：List[[T_i, 80]] -> padded_features [B, T, 80] + feature_lengths [B]
EncoderBackend：FeatureBatch -> encoder_outputs + encoder_lengths + encoder_mask
```

当前 `FeatExtractor` 同时负责逐条 FBank/CMVN 和整个 batch 的 padding。实现时将其整理为“逐条特征计算”和“组 batch/padding”两个步骤，但保持 `kaldi_native_fbank`、CMVN 数学公式以及“先 CMVN、后 padding”的顺序不变。

`EncoderBackend` 的原生契约为：

```text
输入：features [B, T, 80]、feature_lengths [B]
输出：encoder_outputs、encoder_lengths、encoder_mask
```

三种后端都保持官方 `ConformerEncoder.forward()` 的原生三返回值接口。当前 FireRedLID Decoder 只消费 `encoder_outputs` 和 `encoder_mask`，但 `encoder_lengths` 仍作为后端契约的独立输出保留，便于与 eager 基线直接比较，也避免改变官方 Encoder 边界。

三个实现分别为：

- `EagerEncoderBackend`：调用官方 PyTorch Encoder。
- `CompileEncoderBackend`：仅对官方 Encoder 应用 `torch.compile`。
- `TensorRTEncoderBackend`：加载并执行 LID Encoder TensorRT engine。

`FireRedLidConfig.backend` 默认仍为 `"eager"`，因此现有调用方不设置新字段时保持原行为。

## 6. 配置与 API 语义

在现有 `FireRedLidConfig` 上增加：

```python
backend: str = "eager"                 # eager | compile | tensorrt
profile: str = "latency"               # latency | throughput
max_audio_seconds: float = 60.0
batch_strategy: str | None = None       # none | bucket | auto
engine_dir: str | None = None
max_sub_batch_size: int | None = None
fallback_backend: str | None = None
return_diagnostics: bool = False
```

规则如下：

- `profile="latency"` 时，`batch_strategy=None` 解析为 `none`。
- `profile="throughput"` 时，`batch_strategy=None` 解析为 `auto`。
- `max_sub_batch_size=None` 时，从 backend 或 engine manifest 读取运行上限。
- 显式选择 `compile` 或 `tensorrt` 时，初始化失败默认直接报错。
- 只有显式设置 `fallback_backend="eager"` 时才允许自动回退。
- 默认返回字段保持与现有实现兼容；`backend`、`truncated` 和分阶段耗时只在 `return_diagnostics=True` 时增加。

## 7. 逻辑 batch、物理 batch 与分桶

调用方可一次提交任意数量的音频。该数量是逻辑 batch，不要求等于 engine 的最大 batch。

### 7.1 不分桶

`batch_strategy="none"` 时：

1. 保持输入顺序。
2. 当逻辑 batch 超过 `max_sub_batch_size` 时按顺序拆成多个物理 batch。
3. 每个物理 batch padding 到其中最长特征。
4. 依次执行并拼接结果。

### 7.2 固定分桶

`batch_strategy="bucket"` 使用可配置边界，默认是 `5/15/30/60` 秒：

1. 按截断后的实际特征长度分组。
2. 每个组内再按 `max_sub_batch_size` 拆分。
3. 每个物理 batch 只 padding 到本组最长特征。
4. 推理结束后恢复调用方原始顺序。

### 7.3 自动分桶

`batch_strategy="auto"` 先按当前 `max_sub_batch_size` 分别生成“不分桶”和“固定分桶”两套候选物理 batch，然后计算：

```text
direct_padded_frames = sum(B_i * max(lengths_i))
bucket_padded_frames = sum(B_k * max(lengths_k))
saving_ratio = 1 - bucket_padded_frames / direct_padded_frames
```

其中 `i` 表示保持原顺序拆出的物理 batch，`k` 表示按长度分组后拆出的物理 batch。当预计 padding 节省至少 20% 时使用固定分桶，否则保持不分桶。20% 是第一版确定的保守阈值；Linux benchmark 会同时报告不同策略的结果，但不会在运行时隐式改写该阈值。

分桶只整理当前调用中已经收到的输入，不等待未来请求，因此不属于服务调度。未来使用 Triton 时，Triton 负责跨请求 dynamic batching，wrapper 仍可调用相同的 batch planner 处理可变长度音频。

## 8. TensorRT shape 上限

TensorRT engine 的硬上限来自构建时的 optimization profile：

```text
MIN <= [B, T, 80] <= MAX
```

第一版 profile 模板采用：

```text
min_batch = 1
opt_batch = 1
max_batch = 4
min_frames = 1
opt_frames = 1000
max_frames = 6000
feature_dim = 80
```

该模板用于建立最初的在线安全基线，不表示 RTX PRO 5000 或 L20 的最终容量。`max_batch`、`opt_batch` 和 `opt_frames` 全部由 `profiles.yaml` 控制；提高最大 batch 只需修改 profile 并重建 engine，不修改 Python API。

运行时还存在可选的策略上限 `max_sub_batch_size`。它可以小于 engine 的 `max_batch`，用于控制 P95 延迟或为共享 GPU 保留显存，但不能大于 engine 硬上限。

如果输入 batch 超过策略上限，runtime 自动拆分。如果特征长度超过 6000 帧，60 秒截断应已将其消除；若仍超出所有 profile，则抛出明确的 shape 错误。

## 9. Encoder 等价改写

只进行导出和编译所必需的局部等价修改：

- 将 `padding_position_is_0()` 中逐样本 Python 循环改成 `arange` 与广播比较。
- 删除 Encoder forward 中未被返回或消费的 `enc_outputs` Python 列表。
- 保持 mask 的形状、有效位语义和 `uint8` dtype 与现有 Decoder 契约一致。
- TensorRT 导出 wrapper 保持 `encoder_outputs`、`encoder_lengths` 和 `encoder_mask` 三个原生输出。
- 仅当导出测试提供具体失败证据时，才改写原地操作或不支持算子。
- 不重写 Attention 数学公式，不替换激活、归一化、卷积或位置编码。

上述修改必须先通过 eager 新旧 Encoder 的 FP32 等价测试，才能进入 ONNX 导出。

## 10. ONNX 与 TensorRT 产物链路

### 10.1 Mac 阶段

1. 从 `FireRedLID/model.pth.tar` 加载完整官方模型。
2. 提取 `model.encoder` 并切换到 eval 模式。
3. 使用 PyTorch ONNX exporter、opset 17 和动态 batch/time 维导出 FP32 ONNX。
4. 使用 ONNX Runtime CPU 与 eager FP32 对齐多个 `B/T` 组合。
5. 输出 ONNX、checkpoint 标识、输入输出名称和 profile 模板。

Mac 阶段不生成 TensorRT `.plan`，也不声称 CUDA `torch.compile` 已验证。

### 10.2 Linux NVIDIA 阶段

1. 读取同一 FP32 ONNX。
2. 使用 TensorRT builder 和 `profiles.yaml` 构建 FP16 engine。
3. 在 RTX PRO 5000 或 L20 上分别构建 engine，不跨 GPU SKU 宣称兼容。
4. 对齐 eager FP16、compile FP16 和 TensorRT FP16。
5. 生成 engine manifest 和 benchmark 报告。

engine 目录至少包含：

```text
encoder.plan
manifest.json
profiles.yaml
```

`manifest.json` 记录 checkpoint 标识、ONNX 标识、TensorRT/CUDA/PyTorch 版本、GPU 名称、精度、输入输出契约和 profile 范围。加载时发现契约或模型标识不匹配必须失败，不静默使用旧 engine。

## 11. 模型加载与显存所有权

- eager 后端保留完整 PyTorch Encoder 和 Decoder。
- compile 后端保留编译后的 PyTorch Encoder 和原版 Decoder。
- compile 后端的正式支持目标是 Linux CUDA；macOS 只允许执行不作为性能验收依据的兼容性 smoke test，失败时必须给出平台能力错误。
- TensorRT 后端加载 checkpoint 后，用无 PyTorch 参数的 TensorRT adapter 替换 `model.encoder`，再将 Decoder 移到 GPU。
- TensorRT 模式不在 GPU 上重复保留官方 PyTorch Encoder 权重。
- TensorRT 依赖采用延迟导入；仅使用 eager 的 Mac 环境不能因缺少 TensorRT 而导入失败。

## 12. 错误处理

需要提供清晰、可区分的错误：

- 后端在当前平台不可用。
- engine 文件不存在或无法反序列化。
- engine manifest 与 checkpoint 或输入输出契约不匹配。
- 输入 shape 不在任何 optimization profile 范围内。
- 显式后端初始化失败且未配置 fallback。

运行时规则：

- 超过 60 秒的音频截断，不因超长直接使整个 batch 失败。
- 超过最大物理 batch 的逻辑 batch 自动拆分。
- 不改变现有特征提取异常语义：特征阶段抛出异常时，仍由 `FireRedLid.process()` 按当前 batch 级空语言结果处理；诊断模式只补充失败阶段，不回显音频内容。
- 不捕获并吞掉后端初始化错误。

## 13. 正确性验证

### 13.1 Mac 验证

- 保存官方 eager FP32 的 Encoder 输出和最终 LID 结果。
- 比较等价改写前后的 eager FP32 Encoder。
- 比较 ONNX Runtime FP32 与 eager FP32。
- ONNX Runtime 的 `encoder_mask` 必须与 eager 基线逐元素一致。
- ONNX Runtime 的 `encoder_lengths` 必须与 eager Encoder 返回的 lengths 逐元素一致。
- FP32 `encoder_outputs` 使用 `rtol=1e-3, atol=1e-4`。
- 覆盖 1、5、15、30、60 秒以及至少两个 batch size。
- 覆盖逻辑 batch 拆分、顺序恢复、截断和三个 batch strategy。

### 13.2 Linux NVIDIA 验证

- 对照 eager FP32、eager FP16、compile FP16、TensorRT FP16。
- 最终语言标签必须与 eager FP16 一致。
- confidence 绝对误差不超过 `5e-3`。
- TensorRT 的 `encoder_mask` 必须与 eager FP16 基线逐元素一致。
- TensorRT 的 `encoder_lengths` 必须与 eager FP16 Encoder 返回的 lengths 逐元素一致。
- FP16 `encoder_outputs` 使用 `rtol=2e-2, atol=2e-2`。
- 任何标签不一致都视为失败并单独分析，不用整体准确率掩盖。

真实回归集通过通用 manifest 读取；在尚无代表性数据集时，先用仓库示例音频和合成长度输入验证工具链。合成特征只用于 shape、稳定性和 Encoder benchmark，不用于声明 LID 精度。

## 14. Benchmark 设计

同一 benchmark 程序支持：

- 后端：eager FP32、eager FP16、compile FP16、TensorRT FP16。
- 范围：Encoder-only、模型推理、端到端含 FBank。
- 模式：latency、throughput。
- 长度：1、5、15、30、60 秒。
- batch：从 1 开始，逐步增加到当前 profile 和显存允许的上限。
- 策略：`none`、`bucket`、`auto`。

报告必须包含：

- P50/P95/P99 延迟。
- utterances/s。
- audio-seconds/s。
- RTF。
- 峰值 GPU 显存。
- FBank、H2D、Encoder、Decoder 和结果整理的分阶段耗时。
- 首次编译、engine warm-up 和稳定态分别统计。
- 实际输入 shape、物理 batch 数和 padding 比例。

性能验收不预设固定倍数。Linux 阶段的完成标准是：所有正确性要求通过，并生成能够比较各后端、长度、batch 和策略的可复现实测报告。

## 15. 代码与产物布局

Python package：

```text
fireredasr2s/fireredlid/
├── lid.py
└── runtime/
    ├── __init__.py
    ├── encoder_backend.py
    ├── pytorch_backend.py
    ├── tensorrt_backend.py
    └── batch_planner.py
```

导出、构建和验证工具：

```text
runtime/fireredlid/
├── README.md
├── pyproject.toml
├── profiles.yaml
├── export_encoder_onnx.py
├── build_engine.py
├── verify.py
└── benchmark.py
```

测试统一放入新建的 `tests/fireredlid/`，不迁移其他模块测试。

## 16. 分阶段交付

### 阶段 A：macOS

- 建立 eager 基线。
- 完成后端接口、配置和 batch planner。
- 完成 Encoder 等价改写。
- 完成动态 ONNX 导出和 ONNX Runtime FP32 对齐。
- 完成 TensorRT 构建脚本、runtime 接口、验证与 benchmark 工具的静态准备。
- 完成 Mac 可执行测试和 Linux 操作文档。

阶段 A 不将 TensorRT 或 CUDA `torch.compile` 标记为已跑通。

### 阶段 B：Linux + RTX PRO 5000/L20

- 安装与目标 CUDA/TensorRT 匹配的独立运行环境。
- 构建 FP16 engine。
- 跑通 compile 和 TensorRT 后端。
- 完成正确性验证。
- 搜索并提高安全的 `max_batch`。
- 生成 latency/throughput benchmark 报告和推荐 profile。

### 后续候选

只有 profiler 证明有必要时，再单独设计：

- Triton Python wrapper 与 dynamic batching 配置。
- Decoder step 编译或 TensorRT-LLM。
- FBank 并行化、预取、pinned memory 和异步 H2D。
- INT8/FP8 及精度校准。

## 17. 第一版完成标准

第一版设计对应的实现只有在以下条件全部满足后才算完成：

- 现有 eager 默认调用保持兼容。
- Mac 上能够加载真实 FireRedLID checkpoint 并生成基线。
- Encoder 等价改写通过 FP32 回归。
- 动态 ONNX 能覆盖约定的 batch/time shape 并通过 ONNX Runtime 对齐。
- Python API 能处理任意逻辑 batch、60 秒截断、物理 batch 拆分和顺序恢复。
- 三种后端具有一致的 Encoder 输入输出契约。
- TensorRT 不可用时错误清晰，eager 不受可选依赖影响。
- Linux 构建、验证和 benchmark 命令有完整文档。
- 目标 GPU 阶段完成后，标签一致性和数值容差通过，并产出实测报告。
