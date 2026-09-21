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
#
# ⚠️ 2026-09-22 修：这两行原先写死成字面量 `<PRIOR_WORK>`，是开源脱敏时把本机路径
#    替换成占位符、却**忘了接回变量**留下的。后果不是报错指向路径，而是
#    `fatal error: 'gpu.h' file not found` —— 报错指向 include 语句，看不出是路径问题。
#    现在从环境变量 `LPR_PRIOR_WORK` 取（与 tools/paths.py、build.sh 同一套变量名），
#    取不到就 FATAL_ERROR 立即停 —— 宁可构建失败，也不要静默产出一个找不到头文件的路径。
if(NOT DEFINED LPR_PRIOR_WORK OR LPR_PRIOR_WORK STREQUAL "")
    if(DEFINED ENV{LPR_PRIOR_WORK})
        set(LPR_PRIOR_WORK "$ENV{LPR_PRIOR_WORK}")
    else()
        message(FATAL_ERROR
            "LPR_PRIOR_WORK 未设置。ncnn 头文件在前期工作仓库里，需要它的根目录。\n"
            "  设一次即可：export LPR_PRIOR_WORK=<前期工作根目录>\n"
            "  或让 build.sh 导出（它会读同一个变量名）。")
    endif()
endif()
set(NCNN_SRC "${LPR_PRIOR_WORK}/lpr-harmony/third_party/ncnn/src")
set(NCNN_GEN "${LPR_PRIOR_WORK}/lpr-harmony/third_party/ncnn_build_ohos_vk/src")
foreach(_d "${NCNN_SRC}" "${NCNN_GEN}")
    if(NOT EXISTS "${_d}")
        message(FATAL_ERROR "ncnn 头文件目录不存在：${_d}\n  检查 LPR_PRIOR_WORK=${LPR_PRIOR_WORK}")
    endif()
endforeach()
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
