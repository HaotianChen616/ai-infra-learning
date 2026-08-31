// This file is an integration template, not a working MetaX implementation.
// Copy it into a vendor-enabled build and replace the ENOSYS paths with the
// supported MACA/MAS/GDR APIs. Do not invent or depend on private ioctl numbers.

#include <cerrno>
#include <cstddef>
#include <cstdint>

extern "C" {

struct d2rs_metax_exported_buffer {
    void *device_pointer;
    uint64_t length;
    int dmabuf_fd;
    uint64_t dmabuf_offset;
    uint64_t generation;
};

int d2rs_metax_initialize(int device_id)
{
    (void)device_id;
    // TODO(vendor): initialize MACA runtime and select device.
    return ENOSYS;
}

int d2rs_metax_allocate_and_export(uint64_t length, d2rs_metax_exported_buffer *out)
{
    (void)length;
    (void)out;
    // TODO(vendor):
    // 1. allocate C500 memory;
    // 2. export the allocation as dma-buf (or return the documented GDR
    //    registration handle if dma-buf is not the vendor ABI);
    // 3. bind the fd lifetime to the device allocation;
    // 4. increment generation whenever the allocation identity changes.
    return ENOSYS;
}

int d2rs_metax_make_visible_to_device(const d2rs_metax_exported_buffer *buffer,
                                      uint64_t offset,
                                      uint64_t length)
{
    (void)buffer;
    (void)offset;
    (void)length;
    // TODO(vendor): issue the documented cache maintenance / stream event /
    // device fence after an inbound NIC or UB write. A transport CQE is not a
    // substitute for this device-runtime contract.
    return ENOSYS;
}

int d2rs_metax_release(d2rs_metax_exported_buffer *buffer)
{
    (void)buffer;
    // TODO(vendor): close export handle, then free device memory after all
    // in-flight requests and registrations have completed.
    return ENOSYS;
}

} // extern "C"
