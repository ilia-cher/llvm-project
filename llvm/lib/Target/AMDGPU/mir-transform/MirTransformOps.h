//===- MirTransformOps.h - Transform-dialect ops for the mir mirror -------===//
//
// C++ surface for the `transform.mir.*` extension ops plus the transform-state
// extension that hands the data plane to the op `apply()` methods.
//
//===----------------------------------------------------------------------===//

#ifndef MIR_TRANSFORM_OPS_H
#define MIR_TRANSFORM_OPS_H

#include "MirDialect.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace llvm {
class MachineModuleInfo;
class Module;
class raw_ostream;
} // namespace llvm

namespace mir {

// Transform-state extension carrying the data plane into the interpreter: the
// MachineModuleInfo, the Module (resolves a mir.func symbol to its
// llvm::Function; MMI.getModule() is null in the direct-parse path), and the
// console stream for the transform's output.
class MirMMIExtension : public mlir::transform::TransformState::Extension {
public:
  MirMMIExtension(mlir::transform::TransformState &state,
                  llvm::MachineModuleInfo &mmi, llvm::Module &mod,
                  llvm::raw_ostream &os)
      : Extension(state), mmi(mmi), mod(mod), os(os) {}

  llvm::MachineModuleInfo &getMMI() { return mmi; }
  llvm::Module &getModule() { return mod; }
  llvm::raw_ostream &getOS() { return os; }

private:
  llvm::MachineModuleInfo &mmi;
  llvm::Module &mod;
  llvm::raw_ostream &os;
};

// Registers the `transform.mir.*` extension ops onto the transform dialect.
// Call on the DialectRegistry before creating the MLIRContext that will parse
// the transform script.
void registerMirTransformExtension(mlir::DialectRegistry &registry);

} // namespace mir

// Generated custom transform handle type (!transform.mir.mbb).
#define GET_TYPEDEF_CLASSES
#include "MirTransformOpsTypes.h.inc"

// Generated op declarations (added to the `transform` dialect).
#define GET_OP_CLASSES
#include "MirTransformOps.h.inc"

MLIR_DECLARE_EXPLICIT_TYPE_ID(mir::MirMMIExtension)

#endif // MIR_TRANSFORM_OPS_H
