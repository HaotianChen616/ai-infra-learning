#include "d2rs/d2rs.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unistd.h>

namespace d2rs {
namespace {

std::runtime_error system_error(const std::string &operation)
{
    return std::runtime_error(operation + ": " + std::strerror(errno));
}

class SimDeviceMemoryProvider final : public DeviceMemoryProvider {
public:
    const char *name() const noexcept override
    {
        return "host-simulation";
    }

    DeviceAllocation allocate(uint64_t length, int32_t device_id) override
    {
        if (length == 0 || length > static_cast<uint64_t>(SIZE_MAX)) {
            throw std::invalid_argument("invalid simulated device allocation length");
        }

        auto *memory = new uint8_t[static_cast<size_t>(length)];
        std::memset(memory, 0, static_cast<size_t>(length));
        return DeviceAllocation{
            memory,
            length,
            MemoryType::kHostSimulation,
            device_id,
            -1,
            0,
            next_generation_.fetch_add(1),
        };
    }

    void release(DeviceAllocation &allocation) noexcept override
    {
        delete[] static_cast<uint8_t *>(allocation.address);
        allocation.address = nullptr;
        allocation.length = 0;
    }

    void make_visible_to_device(const DeviceAllocation &allocation,
                                uint64_t offset,
                                uint64_t length) override
    {
        if (allocation.address == nullptr || offset > allocation.length ||
            length > allocation.length - offset) {
            throw std::out_of_range("simulated device visibility range is invalid");
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

private:
    std::atomic<uint64_t> next_generation_{1};
};

class SimTransportBackend final : public TransportBackend {
public:
    explicit SimTransportBackend(uint32_t worker_count)
    {
        if (worker_count == 0) {
            throw std::invalid_argument("worker_count must be greater than zero");
        }
        workers_.reserve(worker_count);
        for (uint32_t i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this]() { worker_loop(); });
        }
    }

    ~SimTransportBackend() override
    {
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            stopping_ = true;
        }
        work_cv_.notify_all();
        for (auto &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    const char *name() const noexcept override
    {
        return "simulated-storage-agent";
    }

    PathKind path_kind() const noexcept override
    {
        return PathKind::kSimulation;
    }

    bool is_direct_data_path() const noexcept override
    {
        return false;
    }

    RegionDescriptor register_region(const DeviceAllocation &allocation) override
    {
        if (allocation.address == nullptr || allocation.length == 0) {
            throw std::invalid_argument("cannot register an empty device allocation");
        }

        const uint64_t id = next_region_id_.fetch_add(1);
        {
            std::lock_guard<std::mutex> lock(region_mutex_);
            regions_.emplace(id, allocation);
        }
        return RegionDescriptor{
            id,
            allocation.length,
            allocation.generation,
            allocation.memory_type,
            PathKind::kSimulation,
            reinterpret_cast<uint64_t>(allocation.address),
            id,
        };
    }

    void unregister_region(const RegionDescriptor &region) noexcept override
    {
        std::lock_guard<std::mutex> lock(region_mutex_);
        regions_.erase(region.region_id);
    }

    void submit(const ChunkRequest &request) override
    {
        if (request.length == 0) {
            throw std::invalid_argument("cannot submit a zero-length request");
        }
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            if (stopping_) {
                throw std::runtime_error("transport is stopping");
            }
            work_queue_.push_back(request);
        }
        work_cv_.notify_one();
    }

    std::vector<Completion> poll(size_t max_count,
                                 std::chrono::milliseconds timeout) override
    {
        if (max_count == 0) {
            return {};
        }

        std::unique_lock<std::mutex> lock(completion_mutex_);
        completion_cv_.wait_for(lock, timeout, [this]() {
            return !completion_queue_.empty() || stopping_;
        });

        std::vector<Completion> result;
        const size_t count = std::min(max_count, completion_queue_.size());
        result.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            result.push_back(std::move(completion_queue_.front()));
            completion_queue_.pop_front();
        }
        return result;
    }

private:
    static std::string uri_to_path(const std::string &uri)
    {
        constexpr const char *prefix = "file://";
        if (uri.compare(0, std::strlen(prefix), prefix) == 0) {
            return uri.substr(std::strlen(prefix));
        }
        return uri;
    }

    DeviceAllocation find_region(uint64_t id)
    {
        std::lock_guard<std::mutex> lock(region_mutex_);
        const auto it = regions_.find(id);
        if (it == regions_.end()) {
            throw std::runtime_error("destination region is not registered");
        }
        return it->second;
    }

    static void read_exact(int fd, uint8_t *destination, uint64_t length, uint64_t offset)
    {
        uint64_t completed = 0;
        while (completed < length) {
            const size_t remaining = static_cast<size_t>(length - completed);
            const ssize_t rc = ::pread(fd,
                                       destination + completed,
                                       remaining,
                                       static_cast<off_t>(offset + completed));
            if (rc < 0 && errno == EINTR) {
                continue;
            }
            if (rc < 0) {
                throw system_error("pread");
            }
            if (rc == 0) {
                throw std::runtime_error("unexpected EOF while reading source object");
            }
            completed += static_cast<uint64_t>(rc);
        }
    }

    Completion execute(const ChunkRequest &request)
    {
        Completion completion;
        completion.request_id = request.request_id;
        try {
            const DeviceAllocation region = find_region(request.destination_region_id);
            if (request.destination_offset > region.length ||
                request.length > region.length - request.destination_offset) {
                throw std::out_of_range("destination range exceeds registered region");
            }

            const std::string path = uri_to_path(request.source_uri);
            const int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) {
                throw system_error("open " + path);
            }

            std::vector<uint8_t> storage_staging(static_cast<size_t>(request.length));
            try {
                read_exact(fd, storage_staging.data(), request.length, request.source_offset);
            } catch (...) {
                ::close(fd);
                throw;
            }
            ::close(fd);

            auto *destination = static_cast<uint8_t *>(region.address) + request.destination_offset;
            std::memcpy(destination, storage_staging.data(), static_cast<size_t>(request.length));
            completion.bytes = request.length;
            completion.crc32 = request.checksum ? d2rs::crc32(storage_staging.data(), storage_staging.size()) : 0;
            completion.status = 0;
        } catch (const std::exception &error) {
            completion.status = -1;
            completion.message = error.what();
        }
        return completion;
    }

    void worker_loop()
    {
        for (;;) {
            ChunkRequest request;
            {
                std::unique_lock<std::mutex> lock(work_mutex_);
                work_cv_.wait(lock, [this]() { return stopping_ || !work_queue_.empty(); });
                if (stopping_ && work_queue_.empty()) {
                    return;
                }
                request = std::move(work_queue_.front());
                work_queue_.pop_front();
            }

            Completion completion = execute(request);
            {
                std::lock_guard<std::mutex> lock(completion_mutex_);
                completion_queue_.push_back(std::move(completion));
            }
            completion_cv_.notify_one();
        }
    }

    std::atomic<uint64_t> next_region_id_{1};
    std::mutex region_mutex_;
    std::unordered_map<uint64_t, DeviceAllocation> regions_;

    std::mutex work_mutex_;
    std::condition_variable work_cv_;
    std::deque<ChunkRequest> work_queue_;
    std::atomic<bool> stopping_{false};
    std::vector<std::thread> workers_;

    std::mutex completion_mutex_;
    std::condition_variable completion_cv_;
    std::deque<Completion> completion_queue_;
};

} // namespace

struct Engine::Impl {
    struct Entry {
        DeviceAllocation allocation;
        RegionDescriptor region;
        uint64_t references = 1;
        bool owned = false;
    };

    std::unique_ptr<DeviceMemoryProvider> provider;
    std::unique_ptr<TransportBackend> transport;
    EngineOptions options;
    std::unordered_map<uint64_t, Entry> entries;
    uint64_t registration_hits = 0;
    uint64_t registration_misses = 0;

    Entry &find(BufferHandle handle)
    {
        const auto it = entries.find(handle.id);
        if (it == entries.end()) {
            throw std::invalid_argument("unknown buffer handle");
        }
        return it->second;
    }

    const Entry &find(BufferHandle handle) const
    {
        const auto it = entries.find(handle.id);
        if (it == entries.end()) {
            throw std::invalid_argument("unknown buffer handle");
        }
        return it->second;
    }
};

Engine::Engine(std::unique_ptr<DeviceMemoryProvider> provider,
               std::unique_ptr<TransportBackend> transport,
               EngineOptions options)
    : impl_(std::make_unique<Impl>())
{
    if (!provider || !transport) {
        throw std::invalid_argument("provider and transport are required");
    }
    if (options.chunk_size == 0 || options.io_depth == 0) {
        throw std::invalid_argument("chunk_size and io_depth must be greater than zero");
    }
    impl_->provider = std::move(provider);
    impl_->transport = std::move(transport);
    impl_->options = options;
}

Engine::~Engine()
{
    if (!impl_) {
        return;
    }
    for (auto &item : impl_->entries) {
        impl_->transport->unregister_region(item.second.region);
        if (item.second.owned) {
            impl_->provider->release(item.second.allocation);
        }
    }
}

BufferHandle Engine::allocate(uint64_t length, int32_t device_id)
{
    DeviceAllocation allocation = impl_->provider->allocate(length, device_id);
    RegionDescriptor region;
    try {
        region = impl_->transport->register_region(allocation);
    } catch (...) {
        impl_->provider->release(allocation);
        throw;
    }
    impl_->registration_misses++;
    impl_->entries.emplace(region.region_id, Impl::Entry{allocation, region, 1, true});
    return BufferHandle{region.region_id, allocation.length};
}

BufferHandle Engine::register_external(DeviceAllocation allocation)
{
    if (allocation.address == nullptr || allocation.length == 0) {
        throw std::invalid_argument("external allocation is empty");
    }

    for (auto &item : impl_->entries) {
        auto &entry = item.second;
        if (entry.allocation.address == allocation.address &&
            entry.allocation.length == allocation.length &&
            entry.allocation.generation == allocation.generation) {
            entry.references++;
            impl_->registration_hits++;
            return BufferHandle{entry.region.region_id, entry.allocation.length};
        }
    }

    RegionDescriptor region = impl_->transport->register_region(allocation);
    impl_->registration_misses++;
    impl_->entries.emplace(region.region_id, Impl::Entry{allocation, region, 1, false});
    return BufferHandle{region.region_id, allocation.length};
}

void Engine::release(BufferHandle handle)
{
    auto it = impl_->entries.find(handle.id);
    if (it == impl_->entries.end()) {
        throw std::invalid_argument("unknown buffer handle");
    }
    if (--it->second.references != 0) {
        return;
    }

    impl_->transport->unregister_region(it->second.region);
    if (it->second.owned) {
        impl_->provider->release(it->second.allocation);
    }
    impl_->entries.erase(it);
}

RunStats Engine::read(const std::string &source_uri,
                      uint64_t source_offset,
                      uint64_t length,
                      BufferHandle destination,
                      uint64_t destination_offset)
{
    if (length == 0) {
        throw std::invalid_argument("read length must be greater than zero");
    }
    auto &entry = impl_->find(destination);
    if (destination_offset > entry.allocation.length ||
        length > entry.allocation.length - destination_offset) {
        throw std::out_of_range("read exceeds destination buffer");
    }

    const uint64_t chunks = (length + impl_->options.chunk_size - 1) / impl_->options.chunk_size;
    const auto started = std::chrono::steady_clock::now();
    uint64_t submitted = 0;
    std::exception_ptr submission_error;
    for (uint64_t i = 0; i < chunks; ++i) {
        const uint64_t chunk_offset = i * impl_->options.chunk_size;
        const uint64_t chunk_length = std::min(impl_->options.chunk_size, length - chunk_offset);
        try {
            impl_->transport->submit(ChunkRequest{
                i + 1,
                source_uri,
                source_offset + chunk_offset,
                chunk_length,
                entry.region.region_id,
                destination_offset + chunk_offset,
                impl_->options.checksum,
            });
            submitted++;
        } catch (...) {
            submission_error = std::current_exception();
            break;
        }
    }

    uint64_t completed = 0;
    uint64_t completed_bytes = 0;
    std::string first_error;
    auto last_progress = std::chrono::steady_clock::now();
    while (completed < submitted) {
        auto completions = impl_->transport->poll(
            static_cast<size_t>(submitted - completed), std::chrono::milliseconds(1000));
        if (completions.empty()) {
            if (std::chrono::steady_clock::now() - last_progress > std::chrono::seconds(30)) {
                throw std::runtime_error("no transport completion for 30 seconds");
            }
            continue;
        }
        last_progress = std::chrono::steady_clock::now();
        for (const auto &completion : completions) {
            completed++;
            completed_bytes += completion.bytes;
            if (completion.status != 0 && first_error.empty()) {
                first_error = completion.message.empty() ? "transport request failed" : completion.message;
            }
        }
    }

    if (!first_error.empty()) {
        throw std::runtime_error(first_error);
    }
    if (submission_error) {
        std::rethrow_exception(submission_error);
    }
    if (completed_bytes != length) {
        throw std::runtime_error("transport completed with a short byte count");
    }

    impl_->provider->make_visible_to_device(entry.allocation, destination_offset, length);
    const auto stopped = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(stopped - started).count();

    RunStats stats;
    stats.path = impl_->transport->path_kind();
    stats.direct_data_path = impl_->transport->is_direct_data_path();
    stats.bytes = completed_bytes;
    stats.chunks = chunks;
    stats.registration_hits = impl_->registration_hits;
    stats.registration_misses = impl_->registration_misses;
    stats.elapsed_seconds = elapsed;
    stats.throughput_mib_s = elapsed > 0.0 ? (static_cast<double>(completed_bytes) / (1024.0 * 1024.0)) / elapsed : 0.0;
    return stats;
}

const DeviceAllocation &Engine::allocation(BufferHandle handle) const
{
    return impl_->find(handle).allocation;
}

std::unique_ptr<DeviceMemoryProvider> make_sim_device_provider()
{
    return std::make_unique<SimDeviceMemoryProvider>();
}

std::unique_ptr<TransportBackend> make_sim_transport(uint32_t worker_count)
{
    return std::make_unique<SimTransportBackend>(worker_count);
}

uint32_t crc32(const void *data, size_t length)
{
    const auto *bytes = static_cast<const uint8_t *>(data);
    uint32_t value = 0xFFFFFFFFU;
    for (size_t i = 0; i < length; ++i) {
        value ^= bytes[i];
        for (uint32_t bit = 0; bit < 8; ++bit) {
            const uint32_t mask = 0U - (value & 1U);
            value = (value >> 1U) ^ (0xEDB88320U & mask);
        }
    }
    return ~value;
}

const char *to_string(PathKind path) noexcept
{
    switch (path) {
        case PathKind::kSimulation:
            return "simulation";
        case PathKind::kCompat:
            return "compat";
        case PathKind::kRdma:
            return "rdma";
        case PathKind::kUrma:
            return "urma";
    }
    return "unknown";
}

} // namespace d2rs
