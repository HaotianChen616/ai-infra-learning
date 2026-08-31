#include "d2rs/d2rs.hpp"

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct Options {
    std::string input;
    uint64_t offset = 0;
    uint64_t length = 0;
    uint64_t chunk = 4ULL * 1024 * 1024;
    uint32_t io_depth = 4;
    int32_t device_id = 0;
    bool verify = true;
    bool self_test = false;
    bool json = false;
};

uint64_t parse_size(const std::string &text)
{
    if (text.empty()) {
        throw std::invalid_argument("empty size");
    }
    size_t consumed = 0;
    const uint64_t base = std::stoull(text, &consumed, 0);
    uint64_t multiplier = 1;
    if (consumed < text.size()) {
        if (consumed + 1 != text.size()) {
            throw std::invalid_argument("invalid size suffix: " + text);
        }
        switch (static_cast<char>(std::toupper(static_cast<unsigned char>(text[consumed])))) {
            case 'K':
                multiplier = 1024ULL;
                break;
            case 'M':
                multiplier = 1024ULL * 1024;
                break;
            case 'G':
                multiplier = 1024ULL * 1024 * 1024;
                break;
            default:
                throw std::invalid_argument("invalid size suffix: " + text);
        }
    }
    if (base > std::numeric_limits<uint64_t>::max() / multiplier) {
        throw std::overflow_error("size is too large");
    }
    return base * multiplier;
}

void usage(const char *program)
{
    std::cout
        << "Usage:\n"
        << "  " << program << " --self-test [--json]\n"
        << "  " << program << " --input FILE [--offset N] [--length N]\n"
        << "      [--chunk 4M] [--iodepth 4] [--device 0] [--no-verify] [--json]\n\n"
        << "This executable is the transport-neutral contract demo. Its sim backend\n"
        << "uses host memory and a local file, so direct_data_path is always false.\n";
}

Options parse_options(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto require_value = [&](const char *name) -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + name);
            }
            return argv[i];
        };

        if (argument == "--self-test") {
            options.self_test = true;
        } else if (argument == "--input") {
            options.input = require_value("--input");
        } else if (argument == "--offset") {
            options.offset = parse_size(require_value("--offset"));
        } else if (argument == "--length") {
            options.length = parse_size(require_value("--length"));
        } else if (argument == "--chunk") {
            options.chunk = parse_size(require_value("--chunk"));
        } else if (argument == "--iodepth") {
            options.io_depth = static_cast<uint32_t>(std::stoul(require_value("--iodepth")));
        } else if (argument == "--device") {
            options.device_id = static_cast<int32_t>(std::stoi(require_value("--device")));
        } else if (argument == "--no-verify") {
            options.verify = false;
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }

    if (!options.self_test && options.input.empty()) {
        throw std::invalid_argument("--input or --self-test is required");
    }
    if (options.chunk == 0 || options.io_depth == 0) {
        throw std::invalid_argument("--chunk and --iodepth must be greater than zero");
    }
    return options;
}

void write_exact(int fd, const uint8_t *source, size_t length)
{
    size_t completed = 0;
    while (completed < length) {
        const ssize_t rc = ::write(fd, source + completed, length - completed);
        if (rc < 0 && errno == EINTR) {
            continue;
        }
        if (rc < 0) {
            throw std::runtime_error(std::string("write: ") + std::strerror(errno));
        }
        completed += static_cast<size_t>(rc);
    }
}

std::vector<uint8_t> read_expected(const std::string &path, uint64_t offset, uint64_t length)
{
    if (length > static_cast<uint64_t>(SIZE_MAX)) {
        throw std::overflow_error("verification range is too large");
    }
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("open verification source: " + std::string(std::strerror(errno)));
    }

    std::vector<uint8_t> result(static_cast<size_t>(length));
    uint64_t completed = 0;
    while (completed < length) {
        const ssize_t rc = ::pread(fd,
                                   result.data() + completed,
                                   static_cast<size_t>(length - completed),
                                   static_cast<off_t>(offset + completed));
        if (rc < 0 && errno == EINTR) {
            continue;
        }
        if (rc <= 0) {
            ::close(fd);
            throw std::runtime_error(rc == 0 ? "unexpected EOF during verification"
                                             : "pread verification source: " + std::string(std::strerror(errno)));
        }
        completed += static_cast<uint64_t>(rc);
    }
    ::close(fd);
    return result;
}

int run(Options options)
{
    std::string temporary_path;
    std::vector<uint8_t> self_test_data;
    if (options.self_test) {
        char path[] = "/tmp/d2rs-self-test-XXXXXX";
        const int fd = ::mkstemp(path);
        if (fd < 0) {
            throw std::runtime_error("mkstemp failed");
        }
        temporary_path = path;
        self_test_data.resize(9ULL * 1024 * 1024 + 777);
        for (size_t i = 0; i < self_test_data.size(); ++i) {
            self_test_data[i] = static_cast<uint8_t>((i * 131U + 17U) & 0xFFU);
        }
        try {
            write_exact(fd, self_test_data.data(), self_test_data.size());
        } catch (...) {
            ::close(fd);
            ::unlink(path);
            throw;
        }
        ::close(fd);
        options.input = temporary_path;
        options.offset = 123;
        options.length = self_test_data.size() - 321;
        options.chunk = 1024 * 1024;
        options.io_depth = 3;
        options.verify = true;
    }

    try {
        const uint64_t file_size = std::filesystem::file_size(options.input);
        if (options.offset > file_size) {
            throw std::out_of_range("offset is beyond end of file");
        }
        if (options.length == 0) {
            options.length = file_size - options.offset;
        }
        if (options.length == 0 || options.length > file_size - options.offset) {
            throw std::out_of_range("requested range is outside the input file");
        }

        d2rs::EngineOptions engine_options;
        engine_options.chunk_size = options.chunk;
        engine_options.io_depth = options.io_depth;
        engine_options.checksum = true;
        d2rs::Engine engine(d2rs::make_sim_device_provider(),
                           d2rs::make_sim_transport(options.io_depth),
                           engine_options);

        const d2rs::BufferHandle buffer = engine.allocate(options.length, options.device_id);
        const d2rs::RunStats stats = engine.read(
            "file://" + std::filesystem::absolute(options.input).string(),
            options.offset,
            options.length,
            buffer);

        bool verified = true;
        uint32_t output_crc = 0;
        if (options.verify) {
            const auto expected = read_expected(options.input, options.offset, options.length);
            const auto &allocation = engine.allocation(buffer);
            verified = std::memcmp(expected.data(), allocation.address, expected.size()) == 0;
            output_crc = d2rs::crc32(allocation.address, static_cast<size_t>(options.length));
        }

        if (options.json) {
            std::cout << "{"
                      << "\"backend\":\"sim\","
                      << "\"path\":\"" << d2rs::to_string(stats.path) << "\","
                      << "\"direct_data_path\":" << (stats.direct_data_path ? "true" : "false") << ","
                      << "\"bytes\":" << stats.bytes << ","
                      << "\"chunks\":" << stats.chunks << ","
                      << "\"registration_hits\":" << stats.registration_hits << ","
                      << "\"registration_misses\":" << stats.registration_misses << ","
                      << "\"elapsed_seconds\":" << std::fixed << std::setprecision(6) << stats.elapsed_seconds << ","
                      << "\"throughput_mib_s\":" << std::fixed << std::setprecision(3) << stats.throughput_mib_s << ","
                      << "\"verified\":" << (verified ? "true" : "false") << ","
                      << "\"crc32\":\"0x" << std::hex << std::setw(8) << std::setfill('0') << output_crc << "\""
                      << "}\n";
        } else {
            std::cout << "backend              : sim\n"
                      << "path                 : " << d2rs::to_string(stats.path) << "\n"
                      << "direct_data_path     : " << (stats.direct_data_path ? "yes" : "no") << "\n"
                      << "bytes                : " << stats.bytes << "\n"
                      << "chunks               : " << stats.chunks << "\n"
                      << "registration hits    : " << stats.registration_hits << "\n"
                      << "registration misses  : " << stats.registration_misses << "\n"
                      << "elapsed seconds      : " << std::fixed << std::setprecision(6) << stats.elapsed_seconds << "\n"
                      << "throughput MiB/s     : " << std::fixed << std::setprecision(3) << stats.throughput_mib_s << "\n"
                      << "verified             : " << (verified ? "yes" : "no") << "\n"
                      << "crc32                : 0x" << std::hex << std::setw(8) << std::setfill('0') << output_crc << "\n";
        }

        engine.release(buffer);
        if (!temporary_path.empty()) {
            ::unlink(temporary_path.c_str());
        }
        return verified ? 0 : 2;
    } catch (...) {
        if (!temporary_path.empty()) {
            ::unlink(temporary_path.c_str());
        }
        throw;
    }
}

} // namespace

int main(int argc, char **argv)
{
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception &error) {
        std::cerr << "d2rs_demo: " << error.what() << "\n";
        return 1;
    }
}
