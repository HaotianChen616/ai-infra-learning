#include "d2rs/adapter_c.h"

#include <cerrno>
#include <cstring>
#include <urma_api.h>

extern "C" int d2rs_urma_register_va(void *context,
                                      uint64_t va,
                                      uint64_t length,
                                      uint64_t iova,
                                      d2rs_urma_seg_info *out)
{
    if (context == nullptr || va == 0 || length == 0 || out == nullptr) {
        return EINVAL;
    }

    urma_seg_cfg_t config{};
    config.va = va;
    config.len = length;
    config.flag.bs.token_policy = URMA_TOKEN_NONE;
    config.flag.bs.cacheable = URMA_NON_CACHEABLE;
    config.flag.bs.access = URMA_ACCESS_READ | URMA_ACCESS_WRITE;
    config.flag.bs.user_iova = iova == 0 ? 0 : 1;
    config.iova = iova;

    auto *segment = urma_register_seg(static_cast<urma_context_t *>(context), &config);
    if (segment == nullptr) {
        return EIO;
    }

    urma_seg_t *serialized = nullptr;
    uint32_t serialized_length = 0;
    const urma_status_t status = urma_get_seg_ctx(segment, &serialized, &serialized_length);
    if (status != URMA_SUCCESS) {
        urma_unregister_seg(segment);
        return EIO;
    }

    std::memset(out, 0, sizeof(*out));
    out->opaque_target_seg = segment;
    out->serialized_segment = serialized;
    out->serialized_segment_length = serialized_length;
    return 0;
}

extern "C" int d2rs_urma_unregister_va(d2rs_urma_seg_info *segment)
{
    if (segment == nullptr || segment->opaque_target_seg == nullptr) {
        return EINVAL;
    }
    if (segment->serialized_segment != nullptr) {
        urma_put_seg_ctx(static_cast<urma_seg_t *>(segment->serialized_segment));
    }
    const urma_status_t status = urma_unregister_seg(
        static_cast<urma_target_seg_t *>(segment->opaque_target_seg));
    if (status == URMA_SUCCESS) {
        std::memset(segment, 0, sizeof(*segment));
        return 0;
    }
    return EIO;
}
