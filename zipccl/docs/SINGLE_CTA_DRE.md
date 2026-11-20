# 单 CTA 的 DRE、Encode 与 DRC

在 RTX 4070 上，当前 DRE 用一个 512 线程 CTA 处理 1 MiB BF16 张量，复测耗时 **94.339 µs**，原始 BF16 等效吞吐为 **11.115 GB/s**。这里的 1 MiB 指一个张量的原始大小，即 524,288 个 BF16 元素。下文的性能数据均来自 DRE，Encode 和 DRC 部分给出实现建议。

## 调用 DRE

DRE 读取已压缩的 A 和原始 BF16 的 B，执行 BF16 加法，直接生成压缩的和。调用方传入 A 实际使用的指数窗口起点，以及为输出选定的窗口起点。单 CTA 模式显式传入 `max_blocks = 1`。

以下代码假定设备指针和 stream 已就绪。A 和 B 在这个例子中独立、零均值，标准差均为 0.02，且 A 已用 `input_base` 压缩。实际应用应按数据分布估计输出窗口。

```cpp
constexpr std::size_t n = 524288;
const auto input_base = zipccl::ComputeExponentBase(0.02);
const auto output_base = zipccl::ComputeExponentBase(std::sqrt(2.0) * 0.02);
const auto capacity = zipccl::MaxCompressedBytes(n);
constexpr auto workspace_bytes = zipccl::kMinDreWorkspaceBytes;

// d_output 至少 capacity 字节，d_output_count 为一个设备端 uint32_t。
// d_workspace 是调用方预分配的独立设备缓冲区，至少 workspace_bytes 字节。
cudaError_t status = zipccl::DecodeReduceEncodeAsync(
    d_compressed_a, input_base, d_b, n, output_base,
    d_output, capacity, d_output_count, d_workspace, workspace_bytes,
    stream, 1);
```

检查返回值，并在消费结果的 stream 上建立顺序。同一 stream 可以连续复用暂存区；不同 stream 并发执行时应各用一块，或用事件串行化复用。暂存区下限为 1 MiB。更大的输入会分批复用它，压缩输出仍只需 `MaxCompressedBytes(n)`。若需要紧凑结果的实际字节数，在 DRE 完成后取回 outlier 计数，再调用 `CompressedBytes(n, count)`。

## 1 MiB 吞吐与复测方法

从仓库根目录运行：

```sh
make -C fused dre_benchmark CUDA_ARCH=sm_89
./fused/build/dre_benchmark 524288 1000 0.02 500 1 1048576
```

RTX 4070，预热 500 ms，每次计时前用 72 MiB 缓冲区清理 L2。取 7 组、每组 1000 次的组均值中位数。输入压缩和结果校验在计时区间外。本次输入、输出 outlier 比例分别为 2.142% 和 4.024%，逐位 BF16 校验和链式 DRE 校验均通过。

| 吞吐指标 | 分子 | 实测 |
|---|---|---:|
| 原始 BF16 等效吞吐 | 元素数 × 2 字节 | 11.115 GB/s |
| 实际压缩输出吞吐 | 压缩输出字节数 | 7.898 GB/s |
| 输入输出有效载荷吞吐 | 压缩 A + 原始 B + 压缩输出 | 26.806 GB/s |

三项均以 94.339 µs 为分母，GB/s 按十进制计算。有效载荷按张量大小推算，实际 DRAM 流量还受缓存、暂存区读写及内存事务粒度影响。

## DRE 的性能特点

- **按 warp 编码。** 单 CTA 和多 CTA 使用相同的工作单位。每个 warp 独立处理连续的 1024 个 BF16，独占自己的共享内存切片，按段号读取相邻的输入 outlier offset，并将输出异常指数写入自己的固定槽。16 个 warp 每轮共处理 32 KiB 原始 BF16。完整段使用 16 字节对齐搬运，最后不足 1024 个元素的段按有效掩码处理。
- **按逻辑块压紧。** 压缩格式仍以 4096 个元素为一个逻辑块。每个压紧 warp 接手一个逻辑块，四组各 8 个 lane 分别搬运四个段，并写出块起点、四个段起点和段尾。尾块中空段的起止 offset 相等。单 CTA 一次最多并行压紧 16 个逻辑块，用前缀和生成输出偏移；多 CTA 由每个压紧 warp 用一次 `atomicAdd` 预留本逻辑块的连续输出范围。
- **同步与暂存区复用。** 每轮需 16 KiB 最坏情况暂存空间。单 CTA 处理 1 MiB 输入共 32 轮、512 KiB 槽位，在默认 1 MiB 暂存区内只需一批。批内编码时各 warp 独立推进，压紧前执行一次 CTA 屏障。如果该 CTA 还有下一批，压紧后再同步一次，以便安全覆写其他 warp 刚读过的槽位。最后一批省去复用屏障。
- **实测范围。** 相同配置下，单 CTA 对 256 KiB、512 KiB 和 1 MiB 的原始等效吞吐分别为 9.246、10.397 和 11.115 GB/s。1 MiB 输入使用两个 CTA 时测得 19.459 GB/s。逐位 BF16 和链式 DRE 校验覆盖了段尾、单 CTA 与多 CTA 跨批复用，以及 0% 和 100% 输出 outlier；表中吞吐对应约 4% 输出 outlier。

## 单 CTA Encode 的实现参考

现有入口是 `CompressAsync(..., stream, 1)`。当前 `CompressKernel` 让每个 warp 独立负责一个逻辑块：第一遍写 sign/mantissa 和三个位图、统计各段异常数，随后用 `atomicAdd` 取得块起点，第二遍重新读取原始 BF16 并输出异常指数。这个版本适合作为单 CTA 的计时和格式兼容性基线。

建议先比较一个**只读一次原始 BF16** 的版本。沿用 DRE 的每 warp 独立处理 1024 元素的分工，读取 BF16 后直接分离指数与 sign/mantissa，生成三个位图，同时把窗口外的原始指数写入固定槽。批末复用 DRE 的前缀和与压紧逻辑，在单 CTA 内维护累计输出起点。需要保留 7 个连续指数的编码、每块起点、四个分段起点及段尾，以及段内元素顺序，使结果既能独立解压，也能直接作为 DRE 的 A 输入。

这项改动用暂存区读写换掉第二遍 BF16 读取和逐块原子分配。先在相同窗口和 outlier 比例下比较端到端耗时，再分别测试零异常和高异常比例；后者会显著增加暂存流量。新增暂存需求时宜像 DRE 一样由调用方预分配，并在计时外分配。

## 单 CTA DRC 的实现参考

DRC 的目标是压缩 A 加原始 BF16 B，写出连续 BF16 和。当前 `fused/zipccl.cuh` 已声明 `DecodeReduceCopyAsync`，当前 `fused/zipccl.cu` 中尚无对应定义。

建议从 DRE 的**解码与加法部分**抽取共用设备函数。保持相同的 warp 分段，按 A 的相邻 `warp_offsets` 读取本段异常流；通过三个位图恢复指数，用 warp 内前缀定位异常，复用 `ReduceFullFourCodes` 的 BF16 成对加法。在得到打包 BF16 和后直接写入稠密输出。输出位置由元素索引确定，因此可以省去输出指数窗口、位图、异常槽、前缀压紧以及相应的 CTA 屏障。

优先验证完整轮上相邻线程的写入合并情况：DRE 当前每个 lane 处理多个连续元素，直接从寄存器写出时应检查实际 store sectors。若写事务分散，再与「先写入重排的共享内存，再由 warp 连续写回」对照。尾部用有效元素掩码，奇数长度最后一个 BF16 单独写回。正确性参考为解压 A 后逐元素执行相同 BF16 加法；吞吐按稠密输出的 `2 × n / 时间` 报告，沿用 DRE 的单 CTA、冷缓存和预热设置。

接口、kernel 实现和计时逻辑分别见 [zipccl.cuh](../zipccl.cuh)、[zipccl.cu](../zipccl.cu) 和 [dre_benchmark.cu](../dre_benchmark.cu)。
