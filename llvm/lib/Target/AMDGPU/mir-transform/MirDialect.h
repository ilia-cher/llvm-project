//===- MirDialect.h - AMDGPU MIR mirror dialect ---------------------------===//
//
// C++ surface for the `mir` mirror dialect declared in MirOps.td. Pulls in the
// tablegen-generated dialect, enum, type, and op declarations.
//
//===----------------------------------------------------------------------===//

#ifndef MIR_DIALECT_H
#define MIR_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect declaration.
#include "MirOpsDialect.h.inc"

// Domain enum (scalar/vector) declaration.
#include "MirOpsEnums.h.inc"

// !mir.reg type declaration.
#define GET_TYPEDEF_CLASSES
#include "MirOpsTypes.h.inc"

// Op declarations.
#define GET_OP_CLASSES
#include "MirOps.h.inc"

#endif // MIR_DIALECT_H
