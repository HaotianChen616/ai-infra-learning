#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace d2rs {

enum class MemoryType : uint8_t {
    kHostSimulation = 0,
    kMetaXDevice = 1,
};

enum class PathKind : uint8_t {
    kSimulation = 0,
    kCompat = 1,
    kRdma = 2,
    kUrma = 3,
};

struct DeviceAllocation {
    void *address = nullptr;
    uint64_t length = 0;
    MemoryType memory_type = MemoryType::kHostSimulation;
    int32_t device_id = 0;

    // A dma-buf fd is meaningful only in the process where it is open. It is
    // consumed locally by an RDMA/UB provider and must never be sent on wire.
    int32_t dmabuf_fd = -1;
    uint64_t dmabuf_offset = 0;
    uint64_t generation = 0;
};

struct BufferHandle {
    uint64_t id = 0;
    uint64_t length = 0;
};

struct RegionDescriptor {
    uint64_t region_id = 0;
    uint64_t length = 0;
    uint64_t generation = 0;
    MemoryType memory_type = MemoryType::kHostSimulation;
    PathKind path = PathKind::kSimulation;

    // Opaque only to the transport implementation. A production control
    // protocol serializes an RDMA window or URMA segment, not these fields.
    uint64_t transport_address = 0;
    uint64_t transport_key = 0;
};

struct ChunkRequest {
    uint64_t request_id = 0;
    std::string source_uri;
    uint64_t source_offset = 0;
    uint64_t length = 0;
    uint64_t destination_region_id = 0;
    uint64_t destination_offset = 0;
    bool checksum = true;
};

struct Completion {
    uint64_t request_id = 0;
    uint64_t bytes = 0;
    uint32_t crc32 = 0;
    int32_t status = 0;
    std::string message;
};

struct EngineOptions {
    uint64_t chunk_size = 4ULL * 1024 * 1024;
    uint32_t io_depth = 4;
    bool checksum = true;
};

struct RunStats {
    PathKind path = PathKind::kSimulation;
    bool direct_data_path = false;
    uint64_t bytes = 0;
    uint64_t chunks = 0;
    uint64_t registration_hits = 0;
    uint64_t registration_misses = 0;
    double elapsed_seconds = 0.0;
    double throughput_mib_s = 0.0;
};

class DeviceMemoryProvider {
public:
    virtual ~DeviceMemoryProvider() = default;
    virtual const char *name() const noexcept = 0;
    virtual DeviceAllocation allocate(uint64_t length, int32_t device_id) = 0;
    virtual void release(DeviceAllocation &allocation) noexcept = 0;

    // Called after inbound DMA completion and before a device kernel consumes
    // the data. The MetaX implementation must provide the vendor-defined
    // visibility/cache operation; an RNIC CQE alone is not this contract.
    virtual void make_visible_to_device(const DeviceAllocation &allocation,
                                        uint64_t offset,
                                        uint64_t length) = 0;
};

class TransportBackend {
public:
    virtual ~TransportBackend() = default;
    virtual const char *name() const noexcept = 0;
    virtual PathKind path_kind() const noexcept = 0;
    virtual bool is_direct_data_path() const noexcept = 0;
    virtual RegionDescriptor register_region(const DeviceAllocation &allocation) = 0;
    virtual void unregister_region(const RegionDescriptor &region) noexcept = 0;
    virtual void submit(const ChunkRequest &request) = 0;
    virtual std::vector<Completion> poll(size_t max_count,
                                         std::chrono::milliseconds timeout) = 0;
};

class Engine {
public:
    Engine(std::unique_ptr<DeviceMemoryProvider> provider,
           std::unique_ptr<TransportBackend> transport,
           EngineOptions options = {});
    ~Engine();

    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;

    BufferHandle allocate(uint64_t length, int32_t device_id = 0);
    BufferHandle register_external(DeviceAllocation allocation);
    void release(BufferHandle handle);

    RunStats read(const std::string &source_uri,
                  uint64_t source_offset,
                  uint64_t length,
                  BufferHandle destination,
                  uint64_t destination_offset = 0);

    // Intended for the simulation and for provider-side verification helpers.
    // Production applications should consume the device pointer through their
    // runtime instead of dereferencing it on the CPU.
    const DeviceAllocation &allocation(BufferHandle handle) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<DeviceMemoryProvider> make_sim_device_provider();
std::unique_ptr<TransportBackend> make_sim_transport(uint32_t worker_count);
uint32_t crc32(const void *data, size_t length);
const char *to_string(PathKind path) noexcept;

} // namespace d2rs
