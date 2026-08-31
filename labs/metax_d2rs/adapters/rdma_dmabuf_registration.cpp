#include "d2rs/adapter_c.h"

#include <cerrno>
#include <cstring>
#include <infiniband/verbs.h>
#include <unistd.h>

extern "C" int d2rs_rdma_register_dmabuf(void *pd,
                                          int dmabuf_fd,
                                          uint64_t dmabuf_offset,
                                          uint64_t length,
                                          uint64_t iova,
                                          d2rs_rdma_mr_info *out)
{
    if (pd == nullptr || dmabuf_fd < 0 || length == 0 || out == nullptr) {
        return EINVAL;
    }

    // libibverbs requires iova and dma-buf offset to have the same page
    // offset. The exporter and RNIC provider still decide whether this
    // particular device memory can actually be mapped.
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || (iova % static_cast<uint64_t>(page_size)) !=
                              (dmabuf_offset % static_cast<uint64_t>(page_size))) {
        return EINVAL;
    }

    constexpr int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    ibv_mr *mr = ibv_reg_dmabuf_mr(static_cast<ibv_pd *>(pd),
                                   dmabuf_offset,
                                   static_cast<size_t>(length),
                                   iova,
                                   dmabuf_fd,
                                   access);
    if (mr == nullptr) {
        return errno == 0 ? EIO : errno;
    }

    std::memset(out, 0, sizeof(*out));
    out->opaque_mr = mr;
    out->iova = iova;
    out->length = length;
    out->lkey = mr->lkey;
    out->rkey = mr->rkey;
    return 0;
}

extern "C" int d2rs_rdma_unregister_dmabuf(d2rs_rdma_mr_info *mr)
{
    if (mr == nullptr || mr->opaque_mr == nullptr) {
        return EINVAL;
    }
    const int result = ibv_dereg_mr(static_cast<ibv_mr *>(mr->opaque_mr));
    if (result == 0) {
        std::memset(mr, 0, sizeof(*mr));
    }
    return result;
}
