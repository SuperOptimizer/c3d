cmake_minimum_required(VERSION 3.20)
project(c3d C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)   # for __attribute__((optimize(...))) on the ref path

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Common flags for all targets.
add_compile_options(
    -Wall -Wextra -Wshadow -Wpedantic
    -Wno-implicit-fallthrough   # MurmurHash3 switch uses documented fallthrough
    -Wno-parentheses
)

# Release defaults: -O3 + fast-math per PLAN §0.  Debug keeps fast-math off so
# the debugger sees the canonical evaluation order.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE
    "-O3 -ffast-math -funsafe-math-optimizations -mcpu=native -flto=auto -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG   "-O0 -g -fno-fast-math")

# Library targets: static + shared.  Shared is used by Python ctypes bindings
# (e.g. c3d_zarr_to_c3d.py).
add_library(c3d STATIC c3d.c)
target_include_directories(c3d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_library(c3d_shared SHARED c3d.c)
set_target_properties(c3d_shared PROPERTIES OUTPUT_NAME c3d)
target_include_directories(c3d_shared PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(c3d_shared PRIVATE m)
# Early stages: several static helpers are implemented but not yet wired to
# public entry points.  Silence -Wunused-function on the library target until
# later stages consume them.
target_compile_options(c3d        PRIVATE -Wno-unused-function)
target_compile_options(c3d_shared PRIVATE -Wno-unused-function -fPIC)

# Test target: includes c3d.c directly to reach static internals; defines
# C3D_BUILD_REF so the reference decoder path compiles in (later stages).
add_executable(c3d_test c3d_test.c)
target_compile_definitions(c3d_test PRIVATE C3D_BUILD_REF=1)
target_link_libraries(c3d_test PRIVATE m)
target_include_directories(c3d_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# CLI tools — each is a small wrapper around libc3d.
add_executable(c3d_inspect c3d_inspect.c)
target_link_libraries(c3d_inspect PRIVATE c3d m)

add_executable(c3d_compact c3d_compact.c)
target_link_libraries(c3d_compact PRIVATE c3d m)

add_executable(c3d_train c3d_train.c)
target_link_libraries(c3d_train PRIVATE c3d m)

add_executable(c3d_perf c3d_perf.c)
# c3d_perf #includes c3d.c directly; don't link libc3d (avoid duplicate symbols).
target_link_libraries(c3d_perf PRIVATE m)
target_include_directories(c3d_perf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(c3d_perf PRIVATE -Wno-unused-function)

add_executable(c3d_bench_par c3d_bench_par.c)
target_link_libraries(c3d_bench_par PRIVATE c3d m pthread)

# Benchmark — optionally builds against openh264 for the H.264 baseline.
# Skipped if openh264 isn't installed.
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(OPENH264 QUIET openh264)
endif()
if(OPENH264_FOUND)
    add_executable(c3d_bench c3d_bench.c)
    target_link_libraries(c3d_bench PRIVATE c3d m ${OPENH264_LIBRARIES})
    target_include_directories(c3d_bench PRIVATE ${OPENH264_INCLUDE_DIRS})
    target_compile_options(c3d_bench PRIVATE ${OPENH264_CFLAGS_OTHER})
    message(STATUS "c3d_bench: found openh264 ${OPENH264_VERSION}")
else()
    message(STATUS "c3d_bench: openh264 not found — skipping")
endif()

enable_testing()
add_test(NAME c3d_test COMMAND c3d_test)
