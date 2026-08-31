# MetaX C500 D2RS contract demo

这个目录是沐曦 C500 D2RS 的“可运行接口原型”，不是伪装成真机直通的 benchmark。

- 默认 `sim` 后端可在 macOS/Linux 上编译运行，用来验证：设备区注册语义、请求分片、并发完成、存储侧 staging、完整读和 CRC/字节正确性。
- `direct_data_path=false` 是刻意的：模拟后端不声称绕过 Host DRAM。
- `adapters/rdma_dmabuf_registration.cpp` 是真实的 `ibv_reg_dmabuf_mr` 注册边界。
- `adapters/urma_registration.cpp` 是真实的 `urma_register_seg` 边界，但公共 UMDK API 没有 dma-buf fd 参数；它能否注册 C500 VA 取决于 UDMA/UMMU Provider。
- `adapters/metax_provider_template.cpp` 列出需要沐曦 SDK/驱动落实的四个接口，默认返回 `ENOSYS`，避免编造厂商 API。

## 构建与自测

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/d2rs_demo --self-test --json
```

没有 CMake 时，可以直接编译核心 Demo：

```bash
c++ -std=c++17 -O2 -pthread -Iinclude \
  src/d2rs.cpp src/main.cpp -o d2rs_demo
./d2rs_demo --self-test
```

读取指定文件：

```bash
./build/d2rs_demo \
  --input /mnt/nvme/d2rs-test.bin \
  --offset 4K \
  --length 1G \
  --chunk 4M \
  --iodepth 8 \
  --json
```

## 可选适配器

构建 RDMA dma-buf 注册适配器：

```bash
cmake -S . -B build-rdma \
  -DD2RS_BUILD_RDMA_ADAPTER=ON
cmake --build build-rdma -j
```

构建 URMA VA 注册适配器：

```bash
cmake -S . -B build-urma \
  -DD2RS_BUILD_URMA_ADAPTER=ON \
  -DUMDK_ROOT=/path/to/umdk-or-install-root
cmake --build build-urma -j
```

这两个库只负责最容易被误写的注册边界，不负责伪造 MetaX 导出接口、建 QP/Jetty 或存储 Agent。真机接入顺序详见实验手册。

## 目录映射

```text
include/d2rs/d2rs.hpp             传输无关 API、请求、Region、完成语义
src/d2rs.cpp                      注册缓存、Flow/Slice、模拟存储 Agent
src/main.cpp                      正确性/性能入口
include/d2rs/adapter_c.h          RDMA/URMA 最小 C ABI
adapters/rdma_dmabuf_registration.cpp
                                    dma-buf -> ibv_mr
adapters/urma_registration.cpp    VA -> urma_target_seg
adapters/metax_provider_template.cpp
                                    需要厂商补齐的内存导出/同步边界
tools/check_env.sh                C500/RNIC/UB/IOMMU/拓扑只读探测
```

设计文档见 `../../docs/06-metax-d2rs-design.md`，逐步实验见 `../../docs/07-metax-d2rs-experiment-guide.md`。
