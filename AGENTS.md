# AGENTS.md

This AGENTS.md file defines the project's immutable architecture, algorithms, and design constraints.

Rules:
- NEVER modify, rewrite, delete, or regenerate AGENTS.md.
- Treat its contents as authoritative project constraints.
- Do not alter established algorithms or design decisions without explicit user approval.
- If a task conflicts with this specification, report the conflict instead of modifying the specification.
- Only the user may authorize changes to AGENTS.md.

## 仓库

- 此为NCCL的源码仓库，用于测试ZipCCL算法与ReduceScatter的集成

## ZipCCL的优化版本

- 位于/home/zlc/develop/cpp/nccl/zipccl，其中有面向单CTA高度优化的ZipCCL DRE算子，在docs下可找到描述文档

## 与ReduceScatter集成的思路

建立新的 ZipCCLReduceScatter 集合通信接口，接管 NCCL ReduceScatter 的 BFloat16 Sum Simple协议 Ring 算法的路径，其他路径报错（测试阶段）或回退（实际使用阶段）。

在Ring算法中，集成 ZipCCL 的 Encode，DecodeReduceEncode 和 DecodeReduceCopy 算子。其中Encode和DRE都是同时将发送进行融合，将output直接写到目标rank缓冲区。Encoded帧要包含标准差，元素数量等信息，DRE可以通过这些信息更新估算标准差而无需重新统计。另外Slice和Chunk的大小要按原始BFloat16的等效大小分配，比如原始1MiB的BFloat16,可能按ZipCCL的最坏情况就需要分配1.2MiB,不过同样是常量。


## ZipCCL: BF16 Exponent Window Lossless Compression

**Compression Algorithm:**
- Split each BF16 value into an 8-bit exponent and 8-bit sign-and-mantissa.
- Select 7 consecutive exponents and encode them using 3-bit codes `001`–`111`.
- Encode out-of-window exponents as `000` and compact their original 8-bit exponents into a separate stream.
- Preserve sign-and-mantissa bits for lossless reconstruction.
- With outlier ratio \(p\), the theoretical average size is \(11+8p\) bits/value.

**Exponent Window Estimation:**

Assume the input follows a zero-mean Gaussian distribution \(w\sim N(0,\sigma^2)\). Choose the starting exponent \(x\) to maximize the probability covered by 7 consecutive exponents:

\[
P(x)=\operatorname{erf}\left(\frac{2^{x+7}}{\sigma\sqrt2}\right)-\operatorname{erf}\left(\frac{2^x}{\sigma\sqrt2}\right)
\]

Maximizing \(P(x)\) yields:

\[
\boxed{x_{\mathrm{opt}}\approx\log_2(\sigma)-5.35}
\]

Estimate the standard deviation \(\sigma\), compute \(x_{\mathrm{opt}}\), and select the best nearby integer window. This avoids exponent frequency counting and sorting, assuming an approximately Gaussian distribution.