//===- MirTransformOpsTypes.td - Custom transform handle types ------------===//
//
// The `!transform.mir.mbb` handle type.
//
// Separate from MirTransformOps.td: the typedef generator emits code for every
// TypeDef reachable through the includes, and pulling in TransformTypes.td would
// re-emit the transform dialect's own types and collide with them. Only what a
// bare TypeDef needs is included here.
//
//===----------------------------------------------------------------------===//

#ifndef MIR_TRANSFORM_OPS_TYPES_TD
#define MIR_TRANSFORM_OPS_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"

// A handle whose payload elements each represent one MachineBasicBlock. The
// representative is that MBB's `mir.term` (the primary block's terminator),
// unique per MBB and carrying the `mbb` grouping attr. `mbb_ops` widens a
// representative to the MBB's instruction ops; `parent_mbb` narrows a group back
// to its owning MBB.
def Mir_MbbHandle
  : TypeDef<Transform_Dialect, "MbbHandle",
      [DeclareTypeInterfaceMethods<TransformHandleTypeInterface>]> {
  let summary = "handle to MachineBasicBlocks (one mir.term representative each)";
  let mnemonic = "mir.mbb";
  let assemblyFormat = "";
}

#endif // MIR_TRANSFORM_OPS_TYPES_TD
