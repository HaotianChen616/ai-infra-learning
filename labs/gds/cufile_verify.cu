#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda_runtime.h>
#include <cufile.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string file;
  std::size_t bytes = 64ULL << 20;
  off_t offset = 0;
  int gpu = 0;
  int iterations = 5;
  bool register_buffer = true;
  bool verify = false;
};

void usage(const char* program) {
  std::cerr
      << "Usage: " << program << " --file PATH [options]\n"
      << "  --bytes SIZE       4K-aligned read size; default 64M\n"
      << "  --offset SIZE      4K-aligned file offset; default 0\n"
      << "  --gpu INDEX        CUDA device; default 0\n"
      << "  --iterations N     Timed synchronous reads; default 5\n"
      << "  --no-register      Do not call cuFileBufRegister\n"
      << "  --verify           Compare a GPU checksum with a POSIX checksum\n";
}

std::size_t parse_size(std::string value) {
  value.erase(
      std::remove_if(
          value.begin(), value.end(),
          [](unsigned char ch) { return std::isspace(ch); }),
      value.end());
  if (value.empty()) throw std::invalid_argument("empty size");
  std::size_t multiplier = 1;
  char suffix = static_cast<char>(std::toupper(static_cast<unsigned char>(value.back())));
  if (suffix == 'K' || suffix == 'M' || suffix == 'G') {
    value.pop_back();
    multiplier = suffix == 'K' ? 1ULL << 10 : suffix == 'M' ? 1ULL << 20 : 1ULL << 30;
  }
  std::size_t consumed = 0;
  unsigned long long amount = std::stoull(value, &consumed);
  if (consumed != value.size() || amount == 0) throw std::invalid_argument("invalid size");
  return static_cast<std::size_t>(amount * multiplier);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];
    auto next = [&]() -> std::string {
      if (++i >= argc) throw std::invalid_argument("missing value for " + argument);
      return argv[i];
    };
    if (argument == "--file") {
      options.file = next();
    } else if (argument == "--bytes") {
      options.bytes = parse_size(next());
    } else if (argument == "--offset") {
      options.offset = static_cast<off_t>(parse_size(next()));
    } else if (argument == "--gpu") {
      options.gpu = std::stoi(next());
    } else if (argument == "--iterations") {
      options.iterations = std::stoi(next());
    } else if (argument == "--no-register") {
      options.register_buffer = false;
    } else if (argument == "--verify") {
      options.verify = true;
    } else if (argument == "-h" || argument == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (options.file.empty()) throw std::invalid_argument("--file is required");
  if (options.iterations <= 0) throw std::invalid_argument("--iterations must be positive");
  constexpr std::size_t alignment = 4096;
  if (options.bytes % alignment != 0 || options.offset % alignment != 0) {
    throw std::invalid_argument("--bytes and --offset must be 4K aligned");
  }
  return options;
}

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorString(result));
  }
}

void check_cufile(CUfileError_t result, const char* operation) {
  if (result.err != CU_FILE_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + " failed: cuFile error=" + std::to_string(result.err) +
        ", CUDA error=" + std::to_string(result.cu_err));
  }
}

__global__ void checksum_kernel(
    const unsigned char* data, std::size_t bytes, unsigned long long* result) {
  unsigned long long local = 0;
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < bytes;
       index += blockDim.x * gridDim.x) {
    local += data[index];
  }
  atomicAdd(result, local);
}

std::uint64_t gpu_checksum(void* device_buffer, std::size_t bytes) {
  unsigned long long* device_sum = nullptr;
  check_cuda(cudaMalloc(&device_sum, sizeof(*device_sum)), "cudaMalloc(checksum)");
  try {
    check_cuda(cudaMemset(device_sum, 0, sizeof(*device_sum)), "cudaMemset(checksum)");
    checksum_kernel<<<256, 256>>>(
        static_cast<const unsigned char*>(device_buffer), bytes, device_sum);
    check_cuda(cudaGetLastError(), "checksum kernel launch");
    check_cuda(cudaDeviceSynchronize(), "checksum kernel synchronize");
    unsigned long long host_sum = 0;
    check_cuda(
        cudaMemcpy(&host_sum, device_sum, sizeof(host_sum), cudaMemcpyDeviceToHost),
        "cudaMemcpy(checksum)");
    cudaFree(device_sum);
    return host_sum;
  } catch (...) {
    cudaFree(device_sum);
    throw;
  }
}

std::uint64_t cpu_checksum(const std::string& path, off_t offset, std::size_t bytes) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error(
        "POSIX verification open failed: " + std::string(std::strerror(errno)));
  }
  std::vector<unsigned char> buffer(8ULL << 20);
  std::uint64_t sum = 0;
  std::size_t completed = 0;
  try {
    while (completed < bytes) {
      std::size_t requested = std::min(buffer.size(), bytes - completed);
      ssize_t result =
          pread(fd, buffer.data(), requested, offset + static_cast<off_t>(completed));
      if (result < 0) {
        throw std::runtime_error(
            "POSIX verification read failed: " + std::string(std::strerror(errno)));
      }
      if (result == 0) throw std::runtime_error("unexpected EOF during POSIX verification");
      for (ssize_t i = 0; i < result; ++i) sum += buffer[static_cast<std::size_t>(i)];
      completed += static_cast<std::size_t>(result);
    }
    close(fd);
    return sum;
  } catch (...) {
    close(fd);
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  int fd = -1;
  void* device_buffer = nullptr;
  CUfileHandle_t handle{};
  bool driver_open = false;
  bool handle_registered = false;
  bool buffer_registered = false;
  try {
    Options options = parse_options(argc, argv);
    struct stat file_stat {};
    if (stat(options.file.c_str(), &file_stat) != 0) {
      throw std::runtime_error("stat failed: " + std::string(std::strerror(errno)));
    }
    if (options.offset + static_cast<off_t>(options.bytes) > file_stat.st_size) {
      throw std::runtime_error("requested range extends past EOF");
    }

    check_cuda(cudaSetDevice(options.gpu), "cudaSetDevice");
    check_cuda(cudaFree(nullptr), "CUDA context initialization");
    check_cufile(cuFileDriverOpen(), "cuFileDriverOpen");
    driver_open = true;

    fd = open(options.file.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
      throw std::runtime_error(
          "O_DIRECT open failed: " + std::string(std::strerror(errno)));
    }

    CUfileDescr_t descriptor{};
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descriptor.handle.fd = fd;
    check_cufile(cuFileHandleRegister(&handle, &descriptor), "cuFileHandleRegister");
    handle_registered = true;

    check_cuda(cudaMalloc(&device_buffer, options.bytes), "cudaMalloc(data)");
    if (options.register_buffer) {
      check_cufile(cuFileBufRegister(device_buffer, options.bytes, 0), "cuFileBufRegister");
      buffer_registered = true;
    }

    std::vector<double> durations_ms;
    durations_ms.reserve(options.iterations);
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      auto start = std::chrono::steady_clock::now();
      ssize_t completed = cuFileRead(
          handle, device_buffer, options.bytes, options.offset, 0);
      auto finish = std::chrono::steady_clock::now();
      if (completed < 0) {
        throw std::runtime_error("cuFileRead failed: return=" + std::to_string(completed));
      }
      if (static_cast<std::size_t>(completed) != options.bytes) {
        throw std::runtime_error(
            "short cuFileRead: requested=" + std::to_string(options.bytes) +
            ", completed=" + std::to_string(completed));
      }
      durations_ms.push_back(
          std::chrono::duration<double, std::milli>(finish - start).count());
    }

    std::sort(durations_ms.begin(), durations_ms.end());
    double median_ms = durations_ms[durations_ms.size() / 2];
    if (durations_ms.size() % 2 == 0) {
      median_ms = (durations_ms[durations_ms.size() / 2 - 1] + median_ms) / 2.0;
    }
    double total_ms = 0;
    for (double value : durations_ms) total_ms += value;
    double aggregate_gib_s =
        (static_cast<double>(options.bytes) * options.iterations / (1ULL << 30)) /
        (total_ms / 1000.0);

    std::cout << "file=" << options.file << "\n"
              << "gpu=" << options.gpu << "\n"
              << "bytes=" << options.bytes << "\n"
              << "offset=" << options.offset << "\n"
              << "iterations=" << options.iterations << "\n"
              << "buffer_registered=" << (options.register_buffer ? "yes" : "no") << "\n"
              << std::fixed << std::setprecision(6)
              << "read_ms_median=" << median_ms << "\n"
              << "throughput_gib_s=" << aggregate_gib_s << "\n";

    if (options.verify) {
      std::uint64_t device_sum = gpu_checksum(device_buffer, options.bytes);
      std::uint64_t host_sum = cpu_checksum(options.file, options.offset, options.bytes);
      std::cout << "gpu_checksum=" << device_sum << "\n"
                << "cpu_checksum=" << host_sum << "\n"
                << "verification=" << (device_sum == host_sum ? "PASS" : "FAIL") << "\n";
      if (device_sum != host_sum) throw std::runtime_error("checksum mismatch");
    }

    if (buffer_registered) cuFileBufDeregister(device_buffer);
    cudaFree(device_buffer);
    cuFileHandleDeregister(handle);
    close(fd);
    cuFileDriverClose();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    if (buffer_registered && device_buffer != nullptr) cuFileBufDeregister(device_buffer);
    if (device_buffer != nullptr) cudaFree(device_buffer);
    if (handle_registered) cuFileHandleDeregister(handle);
    if (fd >= 0) close(fd);
    if (driver_open) cuFileDriverClose();
    return 1;
  }
}
