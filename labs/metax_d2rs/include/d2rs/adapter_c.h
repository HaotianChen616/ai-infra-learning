#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct d2rs_rdma_mr_info {
    void *opaque_mr;
    uint64_t iova;
    uint64_t length;
    uint32_t lkey;
    uint32_t rkey;
} d2rs_rdma_mr_info;

// pd is struct ibv_pd*. dmabuf_fd must be valid in the current process.
int d2rs_rdma_register_dmabuf(void *pd,
                              int dmabuf_fd,
                              uint64_t dmabuf_offset,
                              uint64_t length,
                              uint64_t iova,
                              d2rs_rdma_mr_info *out);
int d2rs_rdma_unregister_dmabuf(d2rs_rdma_mr_info *mr);

typedef struct d2rs_urma_seg_info {
    void *opaque_target_seg;
    void *serialized_segment;
    uint32_t serialized_segment_length;
} d2rs_urma_seg_info;

// context is urma_context_t*. Current public UMDK accepts VA/len/iova but no
// dma-buf fd; a C500-capable provider/UMMU bridge is therefore a prerequisite.
int d2rs_urma_register_va(void *context,
                          uint64_t va,
                          uint64_t length,
                          uint64_t iova,
                          d2rs_urma_seg_info *out);
int d2rs_urma_unregister_va(d2rs_urma_seg_info *segment);

#ifdef __cplusplus
}
#endif
