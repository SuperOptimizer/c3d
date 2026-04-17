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

# PGO: configure with -DPGO=GENERATE (instrumented), run c3d_perf / c3d_bench_par
# once on a representative corpus (writes .gcda files under ${PGO_DIR}), then
# reconfigure with -DPGO=USE to rebuild with the profile applied.
set(PGO "OFF" CACHE STRING "PGO mode: OFF | GENERATE | USE")
set(PGO_DIR "${CMAKE_BINARY_DIR}/pgo-data" CACHE PATH "PGO profile directory")
if(PGO STREQUAL "GENERATE")
    file(MAKE_DIRECTORY "${PGO_DIR}")
    add_compile_options(-fprofile-generate=${PGO_DIR})
    add_link_options(-fprofile-generate=${PGO_DIR})
    message(STATUS "PGO=GENERATE → profile will be written to ${PGO_DIR}")
elseif(PGO STREQUAL "USE")
    add_compile_options(-fprofile-use=${PGO_DIR} -fprofile-correction)
    add_link_options(-fprofile-use=${PGO_DIR} -fprofile-correction)
    message(STATUS "PGO=USE → applying profile from ${PGO_DIR}")
elseif(NOT PGO STREQUAL "OFF")
    message(FATAL_ERROR "PGO must be one of OFF | GENERATE | USE (got '${PGO}')")
endif()

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

# Benchmark — builds against openh264 (H.264) + x265 + libde265 (H.265) + libaom (AV1).
# All four must be installed; the bench is skipped otherwise.
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(OPENH264 QUIET openh264)
    pkg_check_modules(X265     QUIET x265)
    pkg_check_modules(DE265    QUIET libde265)
    pkg_check_modules(AOM      QUIET aom)
endif()
if(OPENH264_FOUND AND X265_FOUND AND DE265_FOUND AND AOM_FOUND)
    add_executable(c3d_bench c3d_bench.c)
    target_link_libraries(c3d_bench PRIVATE c3d m pthread
        ${OPENH264_LIBRARIES} ${X265_LIBRARIES} ${DE265_LIBRARIES} ${AOM_LIBRARIES})
    target_include_directories(c3d_bench PRIVATE
        ${OPENH264_INCLUDE_DIRS} ${X265_INCLUDE_DIRS}
        ${DE265_INCLUDE_DIRS}    ${AOM_INCLUDE_DIRS})
    target_compile_options(c3d_bench PRIVATE
        ${OPENH264_CFLAGS_OTHER} ${X265_CFLAGS_OTHER}
        ${DE265_CFLAGS_OTHER}    ${AOM_CFLAGS_OTHER})
    message(STATUS "c3d_bench: openh264 ${OPENH264_VERSION}, x265 ${X265_VERSION}, "
                   "libde265 ${DE265_VERSION}, libaom ${AOM_VERSION}")
else()
    set(_c3d_bench_missing "")
    if(NOT OPENH264_FOUND)
        list(APPEND _c3d_bench_missing openh264)
    endif()
    if(NOT X265_FOUND)
        list(APPEND _c3d_bench_missing x265)
    endif()
    if(NOT DE265_FOUND)
        list(APPEND _c3d_bench_missing libde265)
    endif()
    if(NOT AOM_FOUND)
        list(APPEND _c3d_bench_missing aom)
    endif()
    message(STATUS "c3d_bench: skipping — missing ${_c3d_bench_missing}")
endif()

enable_testing()
add_test(NAME c3d_test COMMAND c3d_test)

# --- PGO helper: `cmake --build build --target pgo_train` ------------------
# Runs c3d_bench_par against an 8-chunk mini corpus at two ratios.  Takes
# ~60-90 s under instrumentation vs ~15 min for the full 64-chunk × 5-ratio
# sweep.  Good enough to capture the hot paths for development rebuilds.
# Full calibration still uses a broader manual sweep.
set(C3D_PGO_CORPUS "${CMAKE_SOURCE_DIR}/corpus" CACHE PATH "Corpus for pgo_train")
set(C3D_PGO_MINI   "${CMAKE_BINARY_DIR}/pgo-mini")
add_custom_target(pgo_train
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${C3D_PGO_MINI}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${C3D_PGO_MINI}"
    COMMAND bash -c "cd '${C3D_PGO_CORPUS}' && ls | head -n 8 | xargs -I{} ln -sf '${C3D_PGO_CORPUS}/{}' '${C3D_PGO_MINI}/{}'"
    COMMAND $<TARGET_FILE:c3d_bench_par> "${C3D_PGO_MINI}" 2 10
    COMMAND $<TARGET_FILE:c3d_bench_par> "${C3D_PGO_MINI}" 2 50
    COMMENT "PGO training on 8-chunk mini corpus (r=10, r=50)"
    DEPENDS c3d_bench_par
    VERBATIM)
