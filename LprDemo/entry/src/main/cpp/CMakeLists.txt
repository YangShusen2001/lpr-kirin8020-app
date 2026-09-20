cmake_minimum_required(VERSION 3.5.0)
project(lpr_native)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

# MindSpore Lite NDK 运行时（在 OHOS sysroot 的 usr/lib/aarch64-linux-ohos/ 下）
find_library(mindspore_lite_lib libmindspore_lite_ndk.so)

# ---- ncnn（A 阶段第二步：Vulkan compute 的载体）
# 库是用 OHOS NDK 交叉编译出来的 AArch64 .so，放在 entry/libs/<arch>/。
# 头文件来自 ncnn 源码树 + 它的构建目录（platform.h / layer_registry.h 等是生成的）。
set(NCNN_SRC "C:/Users/26671/lpr-harmony/third_party/ncnn/src")
set(NCNN_GEN "C:/Users/26671/lpr-harmony/third_party/ncnn_build_ohos_vk/src")
set(NCNN_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/../../../libs/${OHOS_ARCH}")
add_library(ncnn SHARED IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION "${NCNN_LIBS}/libncnn.so")

add_library(entry SHARED
    napi_init.cpp
    ms_engine.cpp
    lpr_pipeline.cpp
    vulkan_probe.cpp
    ncnn_engine.cpp
    nnrt_probe.cpp
    yolov8_detect.cpp
)

target_include_directories(entry PRIVATE ${NCNN_SRC} ${NCNN_GEN})

# libdl：NNRt / HiAI 探针用 dlopen 按 soname 逐级回退加载，失败要能拿到 dlerror。
target_link_libraries(entry PUBLIC
    libace_napi.z.so
    libhilog_ndk.z.so
    ${mindspore_lite_lib}
    ncnn
    dl
)
