//===- MirImport.h - MachineFunction -> mir mirror module -----------------===//
//
// The importer: walk a real AMDGPU MachineFunction (the data plane) and build
// its in-memory `mir` mirror module (the control plane). Structural surface only:
// locations, defs/uses, and CFG edges; no effects, ties, MMOs, or aliasing.
//
//===----------------------------------------------------------------------===//

#ifndef MIR_IMPORT_H
#define MIR_IMPORT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace llvm {
class Module;
class MachineModuleInfo;
} // namespace llvm

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mir {

// Build a mirror module holding one `mir.func` per MachineFunction reachable
// through MMI for the functions in M. Loads the `mir` dialect into ctx.
mlir::OwningOpRef<mlir::ModuleOp>
importMirModule(llvm::Module &M, llvm::MachineModuleInfo &MMI,
                mlir::MLIRContext &ctx);

} // namespace mir

#endif // MIR_IMPORT_H
