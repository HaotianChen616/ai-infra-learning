cmake_minimum_required(VERSION 3.16)
project(metax_d2rs_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

option(D2RS_BUILD_RDMA_ADAPTER "Build the optional libibverbs dma-buf registration adapter" OFF)
option(D2RS_BUILD_URMA_ADAPTER "Build the optional UMDK/URMA VA registration adapter" OFF)

add_library(d2rs_core
    src/d2rs.cpp
)
target_include_directories(d2rs_core PUBLIC include)
target_compile_options(d2rs_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(d2rs_core PRIVATE Threads::Threads)

add_executable(d2rs_demo src/main.cpp)
target_link_libraries(d2rs_demo PRIVATE d2rs_core)

if(D2RS_BUILD_RDMA_ADAPTER)
    find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h REQUIRED)
    find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
    add_library(d2rs_rdma_adapter adapters/rdma_dmabuf_registration.cpp)
    target_include_directories(d2rs_rdma_adapter PUBLIC include PRIVATE ${IBVERBS_INCLUDE_DIR})
    target_link_libraries(d2rs_rdma_adapter PRIVATE ${IBVERBS_LIBRARY})
endif()

if(D2RS_BUILD_URMA_ADAPTER)
    set(UMDK_ROOT "" CACHE PATH "Path to the UMDK source or installation root")
    find_path(URMA_INCLUDE_DIR urma_api.h
        PATHS
            ${UMDK_ROOT}/src/urma/lib/urma/core/include
            ${UMDK_ROOT}/include
        REQUIRED)
    find_library(URMA_LIBRARY urma PATHS ${UMDK_ROOT}/lib ${UMDK_ROOT}/lib64 REQUIRED)
    add_library(d2rs_urma_adapter adapters/urma_registration.cpp)
    target_include_directories(d2rs_urma_adapter PUBLIC include PRIVATE ${URMA_INCLUDE_DIR})
    target_link_libraries(d2rs_urma_adapter PRIVATE ${URMA_LIBRARY})
endif()

enable_testing()
add_test(NAME d2rs_self_test COMMAND d2rs_demo --self-test)
