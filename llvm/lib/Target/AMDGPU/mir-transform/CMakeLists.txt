# Standalone build for the mir-transform prototype driver.
#
# Built as a separate CMake project that consumes a prebuilt in-tree LLVM + MLIR
# via find_package, sidestepping the monorepo's llvm-before-mlir configure order.
#
# Configure (point LLVM_DIR at the in-tree build's cmake dir):
#   cmake -S . -B build -G Ninja \
#     -DLLVM_DIR=/home/ichernia/workspace/llvm-project/build/lib/cmake/llvm
#   cmake --build build          ->  build/mir-transform

cmake_minimum_required(VERSION 3.20)
project(mir-transform CXX C)

find_package(LLVM REQUIRED CONFIG)

# MLIR lives beside LLVM in the same in-tree build. Default MLIR_DIR next to
# LLVM_DIR; a single -DLLVM_DIR=... at configure time is then enough.
if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "${LLVM_LIBRARY_DIR}/cmake/mlir")
endif()
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# Match this LLVM's assertion setting before HandleLLVMOptions. A mismatch in
# ABI-sensitive type sizes stack-smashes at teardown rather than failing to
# compile.
if(NOT DEFINED LLVM_ENABLE_ASSERTIONS)
  set(LLVM_ENABLE_ASSERTIONS ON)
endif()

include(TableGen)
include(HandleLLVMOptions)
include(AddLLVM)
include(AddMLIR)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# These TUs use LLVM/MLIR types with vtables; match their -fno-rtti or typeid
# references fail to link.
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

add_executable(mir-transform mir-transform.cpp)

# Match LLVM's RTTI setting or undefined-typeinfo link errors result.
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(mir-transform PRIVATE -fno-rtti)
endif()

llvm_map_components_to_libnames(mir_transform_libs
  Core
  Support
  MC
  Target
  CodeGen
  CodeGenTypes
  MIRParser
  AMDGPUCodeGen
  AMDGPUAsmParser
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  )

target_link_libraries(mir-transform PRIVATE MirImport ${mir_transform_libs})

#===----------------------------------------------------------------------===#
# The `mir` mirror dialect (MLIR side).
#===----------------------------------------------------------------------===#

# Tablegen the dialect from MirOps.td into the build dir.
set(LLVM_TARGET_DEFINITIONS MirOps.td)
mlir_tablegen(MirOps.h.inc -gen-op-decls)
mlir_tablegen(MirOps.cpp.inc -gen-op-defs)
mlir_tablegen(MirOpsDialect.h.inc -gen-dialect-decls -dialect=mir)
mlir_tablegen(MirOpsDialect.cpp.inc -gen-dialect-defs -dialect=mir)
mlir_tablegen(MirOpsTypes.h.inc -gen-typedef-decls --typedefs-dialect=mir)
mlir_tablegen(MirOpsTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=mir)
mlir_tablegen(MirOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(MirOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MirOpsIncGen)

# The dialect library (registration + generated code).
add_library(MirDialect STATIC MirDialect.cpp)
add_dependencies(MirDialect MirOpsIncGen)
target_include_directories(MirDialect PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(MirDialect PUBLIC MLIRIR MLIRSupport)

# The importer: MachineFunction -> mir mirror module (structural).
#
# It reaches AMDGPU-internal headers, which #include their tablegen'd companions
# by bare name; both the source and generated dirs go on the include path.
set(AMDGPU_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(AMDGPU_GEN_DIR "${LLVM_BINARY_DIR}/lib/Target/AMDGPU")
add_library(MirImport STATIC MirImport.cpp)
add_dependencies(MirImport MirOpsIncGen)
target_include_directories(MirImport PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(MirImport PRIVATE
  ${AMDGPU_SRC_DIR}
  ${AMDGPU_GEN_DIR})
target_link_libraries(MirImport PUBLIC MirDialect MLIRIR MLIRSupport)

#===----------------------------------------------------------------------===#
# Transform-dialect extension ops (control-plane scripting).
#===----------------------------------------------------------------------===#

# Tablegen the transform.mir.* ops. Extension ops on the existing `transform`
# dialect: only op decls/defs are generated, no dialect defs.
set(LLVM_TARGET_DEFINITIONS MirTransformOps.td)
mlir_tablegen(MirTransformOps.h.inc -gen-op-decls)
mlir_tablegen(MirTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MirTransformOpsIncGen)

# The !transform.mir.mbb handle type, tablegen'd from its own .td to keep the
# typedef generator from re-emitting the transform dialect's built-in types. No
# --typedefs-dialect: the type is injected onto the existing `transform` dialect;
# only the type class decls/defs are generated.
set(LLVM_TARGET_DEFINITIONS MirTransformOpsTypes.td)
mlir_tablegen(MirTransformOpsTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MirTransformOpsTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MirTransformOpsTypesIncGen)

# The transform-ops library: op apply()/getEffects(), the MMI transform-state
# extension, and the dialect-extension registration.
add_library(MirTransformOps STATIC MirTransformOps.cpp)
add_dependencies(MirTransformOps
  MirTransformOpsIncGen MirTransformOpsTypesIncGen MirOpsIncGen)
target_include_directories(MirTransformOps PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})
# transform.mir.schedule reaches AMDGPU-internal scheduler headers; same include
# arrangement as MirImport.
target_include_directories(MirTransformOps PRIVATE
  ${AMDGPU_SRC_DIR}
  ${AMDGPU_GEN_DIR})
target_link_libraries(MirTransformOps PUBLIC
  MirDialect
  MLIRIR
  MLIRSupport
  MLIRTransformDialect
  MLIRTransformDialectInterfaces)

# The driver additionally links the transform interpreter utils (script parse +
# entry-point lookup) and the transform-ops library.
target_link_libraries(mir-transform PRIVATE
  MirTransformOps
  MLIRParser
  MLIRTransformDialect
  MLIRTransformDialectInterfaces
  MLIRTransformDialectTransforms)

# mlir-opt-style driver: parse/print/round-trip the mirror dialect.
add_executable(mir-opt mir-opt.cpp)
target_link_libraries(mir-opt PRIVATE
  MirDialect
  MLIROptLib
  MLIRIR
  MLIRParser
  MLIRSupport)
