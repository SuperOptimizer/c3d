cmake_minimum_required(VERSION 3.20)
project(c3d VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)   # for __attribute__((optimize(...))) on the ref path

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Make the static lib PIC by default so downstream shared consumers (e.g.
# Volume-Cartographer's codec registry built as a plugin) can link against
# the static libc3d.a without hitting R_X86_64_PC32 relocation errors.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Common warning flags for all targets.
#
# GCC: explicit curated list.  Clang: `-Weverything` minus the known-noise
# categories (docs, padding, C++-interop checks, pre-C23 pedantry, etc.).
if(CMAKE_C_COMPILER_ID STREQUAL "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "AppleClang")
    add_compile_options(
        -Weverything
        # Not applicable to this project:
        -Wno-c++98-compat -Wno-c++98-compat-pedantic
        -Wno-pre-c11-compat -Wno-pre-c23-compat -Wno-pre-c23-compat-pedantic
        -Wno-c11-extensions -Wno-c2x-extensions -Wno-c23-extensions
        -Wno-gnu-zero-variadic-macro-arguments
        # Noise:
        -Wno-documentation -Wno-documentation-unknown-command
        -Wno-padded
        -Wno-covered-switch-default
        -Wno-declaration-after-statement
        -Wno-unsafe-buffer-usage -Wno-unsafe-buffer-usage-in-libc-call
        -Wno-disabled-macro-expansion
        -Wno-switch-enum -Wno-switch-default
        -Wno-reserved-identifier -Wno-reserved-id-macro
        -Wno-unknown-warning-option
        -Wno-unused-macros
        -Wno-bad-function-cast
        -Wno-implicit-void-ptr-cast
        -Wno-implicit-int-enum-cast
        # Project-wide carve-outs (match the GCC list):
        -Wno-implicit-fallthrough   # MurmurHash3 uses documented fallthrough
        -Wno-parentheses
        # Float equality is load-bearing for sentinel comparisons in the codec:
        -Wno-float-equal
        # We use designated-init zero-fills as a C23 idiom:
        -Wno-missing-field-initializers
    )
else()
    add_compile_options(
        -Wall -Wextra -Wpedantic -Wshadow
        -Wconversion -Wdouble-promotion
        -Wcast-align -Wcast-qual
        -Wmissing-prototypes -Wstrict-prototypes -Wold-style-definition
        -Wnull-dereference -Wstringop-overflow
        -Wformat=2 -Wformat-security
        -Wundef -Wwrite-strings -Wvla
        -Wno-implicit-fallthrough   # MurmurHash3 uses documented fallthrough
        -Wno-parentheses
        -Wno-missing-field-initializers
    )
endif()

# OpenMP — used by §E DWT per-level passes to parallelise the outer z/y loops
# within a single chunk encode/decode.  Each pragma is a no-op when OpenMP is
# unavailable (compilers gracefully fall through), but libgomp gives us real
# thread-level scaling on the hot DWT kernels.
find_package(OpenMP)
if(OpenMP_C_FOUND)
    add_compile_options(${OpenMP_C_FLAGS})
    link_libraries(${OpenMP_C_LIBRARIES})
    message(STATUS "OpenMP: ${OpenMP_C_FLAGS} (libs: ${OpenMP_C_LIBRARIES})")
else()
    message(STATUS "OpenMP not found — DWT passes stay single-threaded")
endif()

# Release defaults: -O3 + fast-math per PLAN §0.  Debug keeps fast-math off so
# the debugger sees the canonical evaluation order.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
# CPU-arch defaults + compile-time SIMD tier selection.
#
# C3D_SIMD = AUTO | SCALAR | NEON | AVX2 | AVX512
#   AUTO    — auto-detect from compiler predefines (-march=native path).
#   SCALAR  — disable all SIMD, portable C only (used for cross-compile sanity).
#   NEON    — require aarch64 + NEON (always on for aarch64).
#   AVX2    — require x86_64 + AVX2 (Haswell and later).
#   AVX512  — require x86_64 + AVX512F/DQ/BW/VL + Zen-4-class ISA bundle.
#
# The selection maps to a -m flag and a C3D_FORCE_* macro; c3d.c picks the
# matching kernel path via its `#if C3D_HAVE_*` chain.
set(C3D_SIMD "AUTO" CACHE STRING "SIMD tier: AUTO | SCALAR | NEON | AVX2 | AVX512")
set_property(CACHE C3D_SIMD PROPERTY STRINGS AUTO SCALAR NEON AVX2 AVX512)

if(C3D_SIMD STREQUAL "AUTO")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm)")
        set(C3D_ARCH_FLAGS "-mcpu=native")
    else()
        set(C3D_ARCH_FLAGS "-march=native")
    endif()
elseif(C3D_SIMD STREQUAL "SCALAR")
    # Keep the normal -mcpu / -march so FP codegen still works, but set
    # C3D_FORCE_SCALAR so the hot kernels bypass their NEON / AVX paths.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm)")
        set(C3D_ARCH_FLAGS "-mcpu=native")
    else()
        set(C3D_ARCH_FLAGS "-march=x86-64")
    endif()
    add_compile_definitions(C3D_FORCE_SCALAR=1)
elseif(C3D_SIMD STREQUAL "NEON")
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm)")
        message(FATAL_ERROR "C3D_SIMD=NEON requires an aarch64/arm target")
    endif()
    set(C3D_ARCH_FLAGS "-mcpu=native")
    add_compile_definitions(C3D_FORCE_NEON=1)
elseif(C3D_SIMD STREQUAL "AVX2")
    set(C3D_ARCH_FLAGS "-march=x86-64-v3 -mavx2 -mfma -mbmi -mbmi2")
    add_compile_definitions(C3D_FORCE_AVX2=1)
elseif(C3D_SIMD STREQUAL "AVX512")
    # Zen 4 / Sapphire Rapids tier.  Covers F + DQ + BW + VL (the Skylake-X
    # baseline) plus VBMI/VBMI2/BITALG/VNNI/VPOPCNTDQ/IFMA.  -march=znver4 is
    # the closest turnkey toolchain bundle.
    set(C3D_ARCH_FLAGS "-march=znver4")
    add_compile_definitions(C3D_FORCE_AVX2=1 C3D_FORCE_AVX512=1)
else()
    message(FATAL_ERROR "C3D_SIMD must be AUTO | SCALAR | NEON | AVX2 | AVX512 (got '${C3D_SIMD}')")
endif()
message(STATUS "C3D_SIMD=${C3D_SIMD} → ${C3D_ARCH_FLAGS}")

set(CMAKE_C_FLAGS_RELEASE
    "-O3 -ffast-math -funsafe-math-optimizations ${C3D_ARCH_FLAGS} -flto=auto -DNDEBUG \
     -fstack-protector-strong")
set(CMAKE_C_FLAGS_DEBUG   "-O0 -g -fno-fast-math -fstack-protector-strong")

# Release-only hardening (FORTIFY_SOURCE needs optimisation to work; it also
# conflicts with some sanitizers, so we add it via compile definitions on
# the non-sanitizer path only).  Checked at generation time.
if(SANITIZE STREQUAL "OFF")
    add_compile_definitions($<$<CONFIG:Release>:_FORTIFY_SOURCE=2>)
endif()

# Sanitizer presets.  Choose with -DSANITIZE=<mode>, override build dir to keep
# it separate from the Release build (e.g. cmake -S . -B build-asan -DSANITIZE=address).
# Modes: OFF | address | undefined | address+undefined | thread | memory | leak.
# Sanitizer builds drop LTO + fast-math and reduce opt level so diagnostics stay
# readable; keep them off the Release default.
set(SANITIZE "OFF" CACHE STRING
    "Sanitizer: OFF | address | undefined | address+undefined | thread | memory | leak")
if(NOT SANITIZE STREQUAL "OFF")
    set(_san_flags "")
    if(SANITIZE STREQUAL "address")
        set(_san_flags "-fsanitize=address")
    elseif(SANITIZE STREQUAL "undefined")
        set(_san_flags "-fsanitize=undefined")
    elseif(SANITIZE STREQUAL "address+undefined")
        set(_san_flags "-fsanitize=address,undefined")
    elseif(SANITIZE STREQUAL "thread")
        set(_san_flags "-fsanitize=thread")
    elseif(SANITIZE STREQUAL "memory")
        set(_san_flags "-fsanitize=memory -fsanitize-memory-track-origins=2")
    elseif(SANITIZE STREQUAL "leak")
        set(_san_flags "-fsanitize=leak")
    else()
        message(FATAL_ERROR "SANITIZE must be OFF | address | undefined | address+undefined | thread | memory | leak (got '${SANITIZE}')")
    endif()
    # Split string → list for add_compile_options / add_link_options.
    separate_arguments(_san_flags_list UNIX_COMMAND "${_san_flags}")
    add_compile_options(${_san_flags_list} -fno-omit-frame-pointer -g -O1)
    add_link_options(${_san_flags_list})
    # Sanitizers + fast-math + LTO are a trap for UBSan diagnostics; drop them.
    set(CMAKE_C_FLAGS_RELEASE "-O1 -g -fno-omit-frame-pointer ${C3D_ARCH_FLAGS} -DNDEBUG")
    message(STATUS "SANITIZE=${SANITIZE} → ${_san_flags}")
endif()

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

# Library targets: static + shared.  Both are first-class install targets
# (see the install() block near the bottom of this file).  The static lib
# is what most C++ embedders link (Volume-Cartographer, etc.); the shared
# lib is what Python ctypes bindings + dynamically-loaded plugins pick up.
#
# Include-interface uses generator expressions so install EXPORT writes the
# correct include path downstream (source tree during local builds, the
# installed header directory after `make install`).
add_library(c3d STATIC c3d.c)
target_include_directories(c3d PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(c3d PUBLIC m)

add_library(c3d_shared SHARED c3d.c)
set_target_properties(c3d_shared PROPERTIES
    OUTPUT_NAME c3d
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
target_include_directories(c3d_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(c3d_shared PUBLIC m)
# -fno-semantic-interposition so internal calls bind direct instead of going
# through the PLT (small perf win on the hot path).
target_compile_options(c3d_shared PRIVATE -fno-semantic-interposition)

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

add_executable(c3d_train_dz c3d_train_dz.c)
target_link_libraries(c3d_train_dz PRIVATE c3d m)

add_executable(c3d_progressive c3d_progressive.c)
target_link_libraries(c3d_progressive PRIVATE c3d m)

add_executable(c3d_perf c3d_perf.c)
# c3d_perf #includes c3d.c directly; don't link libc3d (avoid duplicate symbols).
target_link_libraries(c3d_perf PRIVATE m)
target_include_directories(c3d_perf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(c3d_perf PRIVATE -Wno-unused-function)

add_executable(c3d_bench_par c3d_bench_par.c)
target_link_libraries(c3d_bench_par PRIVATE c3d m pthread)

# Benchmark — builds against openh264 (H.264) + x265 + libde265 (H.265) +
# libaom (AV1) + libzfp (ZFP).  All must be installed; bench is skipped otherwise.
# ZFP in Debian doesn't ship a .pc, so we look it up via find_library.
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(OPENH264 QUIET openh264)
    pkg_check_modules(X265     QUIET x265)
    pkg_check_modules(DE265    QUIET libde265)
    pkg_check_modules(AOM      QUIET aom)
endif()
find_path(ZFP_INCLUDE_DIR  zfp.h)
find_library(ZFP_LIBRARY   zfp)
if(ZFP_INCLUDE_DIR AND ZFP_LIBRARY)
    set(ZFP_FOUND TRUE)
endif()
# SZ3: vendored under third_party/SZ3 (static libSZ3c.a + zstd).  Detect via
# the built artifact path; user runs `cmake --build third_party/SZ3/build`
# once before configuring c3d_bench.
set(SZ3_DIR "${CMAKE_SOURCE_DIR}/third_party/SZ3")
set(SZ3_LIB "${SZ3_DIR}/build/tools/sz3c/libSZ3c.a")
set(SZ3_INC "${SZ3_DIR}/tools/sz3c/include")
if(EXISTS "${SZ3_LIB}")
    set(SZ3_FOUND TRUE)
    pkg_check_modules(ZSTD QUIET libzstd)
endif()
# TTHRESH: invoked as CLI, just need the binary present.
set(TTHRESH_BIN "${CMAKE_SOURCE_DIR}/third_party/tthresh/build/tthresh")
if(EXISTS "${TTHRESH_BIN}")
    set(TTHRESH_FOUND TRUE)
endif()
if(OPENH264_FOUND AND X265_FOUND AND DE265_FOUND AND AOM_FOUND
   AND ZFP_FOUND AND SZ3_FOUND AND ZSTD_FOUND AND TTHRESH_FOUND)
    add_executable(c3d_bench c3d_bench.c)
    target_link_libraries(c3d_bench PRIVATE c3d m pthread
        ${OPENH264_LIBRARIES} ${X265_LIBRARIES} ${DE265_LIBRARIES} ${AOM_LIBRARIES}
        ${ZFP_LIBRARY} ${SZ3_LIB} ${ZSTD_LIBRARIES} stdc++ gomp)
    target_include_directories(c3d_bench PRIVATE
        ${OPENH264_INCLUDE_DIRS} ${X265_INCLUDE_DIRS}
        ${DE265_INCLUDE_DIRS}    ${AOM_INCLUDE_DIRS}
        ${ZFP_INCLUDE_DIR}       ${SZ3_INC})
    target_compile_options(c3d_bench PRIVATE
        ${OPENH264_CFLAGS_OTHER} ${X265_CFLAGS_OTHER}
        ${DE265_CFLAGS_OTHER}    ${AOM_CFLAGS_OTHER})
    target_compile_definitions(c3d_bench PRIVATE TTHRESH_BIN="${TTHRESH_BIN}")
    message(STATUS "c3d_bench: openh264 ${OPENH264_VERSION}, x265 ${X265_VERSION}, "
                   "libde265 ${DE265_VERSION}, libaom ${AOM_VERSION}, "
                   "zfp ${ZFP_LIBRARY}, sz3 ${SZ3_LIB}, tthresh ${TTHRESH_BIN}")
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
    if(NOT ZFP_FOUND)
        list(APPEND _c3d_bench_missing zfp)
    endif()
    if(NOT SZ3_FOUND)
        list(APPEND _c3d_bench_missing sz3)
    endif()
    if(NOT ZSTD_FOUND)
        list(APPEND _c3d_bench_missing libzstd)
    endif()
    if(NOT TTHRESH_FOUND)
        list(APPEND _c3d_bench_missing tthresh)
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

# ---------------------------------------------------------------------------
# Install rules + CMake package config + pkg-config — so downstream projects
# (e.g. Volume-Cartographer) can `find_package(c3d)` or `pkg_check_modules(c3d)`
# and link `c3d::c3d` (static) or `c3d::c3d_shared` (shared).
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS c3d c3d_shared
    EXPORT c3dTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES c3d.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT c3dTargets
    FILE c3dTargets.cmake
    NAMESPACE c3d::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/c3d)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/c3d-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/c3d-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/c3d)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/c3d-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/c3d-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/c3d-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/c3d)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/c3d.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/c3d.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/c3d.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
