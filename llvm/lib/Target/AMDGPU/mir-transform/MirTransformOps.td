//===- MirTransformOps.td - Transform-dialect ops for the mir mirror ------===//
//
// Custom `transform.mir.*` ops that drive the `mir` mirror (MirOps.td) via the
// transform-dialect interpreter. Extension ops on the existing `transform`
// dialect, not a new dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MIR_TRANSFORM_OPS_TD
#define MIR_TRANSFORM_OPS_TD

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

// The !transform.mir.mbb handle type is defined in MirTransformOpsTypes.td. Ops
// declare TransformHandleTypeInterface operands/results; apply()/checkPayload
// enforce the concrete !mbb typing.

//===----------------------------------------------------------------------===//
// MBB enumeration / navigation over the !transform.mir.mbb type.
//===----------------------------------------------------------------------===//

// Enumerate every MachineBasicBlock of a `mir.func` as an `!transform.mir.mbb`
// handle, one representative per MBB in layout order. `$target` is one
// `mir.func`. A func with no blocks yields an empty handle.
def Mir_GetMbbsOp : Op<Transform_Dialect, "mir.get_mbbs", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Handle over every MBB of a mir.func (one representative each)";
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat =
      "$target `:` functional-type($target, $result) attr-dict";
}

// Widen an `!transform.mir.mbb` handle to a handle over the instruction ops
// (asm+term) of the represented MBB(s). Inverse of `parent_mbb`.
def Mir_MbbOpsOp : Op<Transform_Dialect, "mir.mbb_ops", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Handle over all instruction ops (asm+term) of an MBB handle";
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat =
      "$target `:` functional-type($target, $result) attr-dict";
}

// Keep the MBBs of an `!transform.mir.mbb` handle that a named matcher accepts,
// running `@matcher` once per representative in `$target`. The matcher takes one
// readonly `!transform.mir.mbb` and yields one `!transform.mir.mbb`; a
// silenceable failure means "does not match" and skips the MBB, a definite
// failure aborts. The result concatenates the accepted MBBs.
def Mir_CollectMbbsOp : Op<Transform_Dialect, "mir.collect_mbbs", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Keep the MBBs of an !mbb handle a named matcher accepts";
  let arguments = (ins TransformHandleTypeInterface:$target,
                       SymbolRefAttr:$matcher);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat =
      "$matcher `in` $target `:` functional-type($target, $result) attr-dict";
}

// Run the body once per MachineBasicBlock of a `mir.func`, in layout order. Each
// iteration binds the body's block argument to a size-one `!transform.mir.mbb`
// handle. Fuses `get_mbbs` + `transform.foreach`; filter with `collect_mbbs`
// between them when the blocks must be narrowed first.
def Mir_ForeachMbbOp : Op<Transform_Dialect, "mir.foreach_mbb", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SingleBlockImplicitTerminator<"::mlir::transform::YieldOp">]> {
  let summary = "Run the body once per MachineBasicBlock of a mir.func";
  let arguments = (ins TransformHandleTypeInterface:$target);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$target `:` type($target) $body attr-dict";
  let extraClassDeclaration = [{
    /// Allow the `transform.` dialect prefix to be omitted inside the body.
    static ::llvm::StringRef getDefaultDialect() { return "transform"; }
  }];
}

// Print each MachineBasicBlock an `!transform.mir.mbb` handle represents,
// dumping the real MachineInstrs from the data plane.
def Mir_PrintMbbOp : Op<Transform_Dialect, "mir.print_mbb", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Print one MachineBasicBlock (dumped from the data plane)";
  let arguments = (ins TransformHandleTypeInterface:$target);
  let assemblyFormat = "$target `:` type($target) attr-dict";
  let extraClassDeclaration = [{
    static ::llvm::StringRef getDefaultDialect() { return "transform"; }
  }];
}

//===----------------------------------------------------------------------===//
// Match / navigate / split.
//===----------------------------------------------------------------------===//

// Filter a handle's payload to the `mir.asm` ops whose mnemonic contains a
// substring. `$target` may hold a `mir.func` (searched recursively) or a set of
// mir ops (tested directly); non-asm ops are ignored.
def Mir_MatchAsmOp : Op<Transform_Dialect, "mir.match_asm", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    MatchOpInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Filter to mir.asm ops whose mnemonic contains a substring";
  let arguments = (ins TransformHandleTypeInterface:$target, StrAttr:$contains);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat =
      "$target `contains` $contains `:` functional-type($target, $result)"
      " attr-dict";
}

// Given a handle over a group of mir instruction ops, produce the owning MBB's
// `!transform.mir.mbb` handle. Inverse of `mbb_ops`. Errors (silenceable) if the
// group is empty or spans more than one MBB.
def Mir_ParentMbbOp : Op<Transform_Dialect, "mir.parent_mbb", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Handle over the owning MBB's instruction ops of a group";
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat =
      "$target `:` functional-type($target, $result) attr-dict";
}

// Hoist `$group` plus its in-MBB dependency cone into a new fallthrough
// predecessor block, on both planes. The cone is the transitive predecessor
// closure over all dep kinds (buildSchedGraph), covering EXEC/SCC/memory
// ordering. The owning MBB is derived from `$group`.
//
//   * `$group`     consumed (its ops are moved, not erased).
//   * `$pred`      the new predecessor MBB, holding the hoisted group + cone.
//   * `$remainder` the original MBB, holding the leftovers.
//
// `$group` must be non-empty and hold `mir.asm` ops of a single MBB. Predecessors
// reaching that MBB by an explicit branch are not supported.
def Mir_SplitOp : Op<Transform_Dialect, "mir.split", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Hoist a group + its cone into a new predecessor MBB";
  let arguments = (ins TransformHandleTypeInterface:$group);
  let results = (outs TransformHandleTypeInterface:$pred,
                      TransformHandleTypeInterface:$remainder);
  let assemblyFormat =
      "$group `:` functional-type($group, results) attr-dict";
}

//===----------------------------------------------------------------------===//
// Schedule one MBB with the coexec machine scheduler.
//===----------------------------------------------------------------------===//

// Schedule the MBB that `$target` refers to, in place, with the AMDGPU coexec
// scheduler. `$target` is a single `!transform.mir.mbb`. Errors (silenceable) if
// the handle is empty or holds more than one MBB.
//
// A `ScheduleDAGMILive` is built around `AMDGPUCoExecSchedStrategy` plus the two
// coexec mutations (IGroupLP Initial + BarrierLatency) and driven over this
// block's single region; the GCN stage machinery is bypassed and the schedule is
// kept, never reverted. Analyses are recomputed per call. The mirror is then
// reconciled move-only: the primary block's `mir.asm` ops are permuted to the new
// order. No handle is consumed or produced.
//
// `scheduler` is optional, defaults to "coexec", and only "coexec" is accepted.
def Mir_ScheduleOp : Op<Transform_Dialect, "mir.schedule", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Schedule one MBB in place with the coexec scheduler";
  let arguments = (ins TransformHandleTypeInterface:$target,
                       OptionalAttr<StrAttr>:$scheduler);
  // $target is a single !transform.mir.mbb, no results: a plain type suffices.
  let assemblyFormat = "$target `:` type($target) attr-dict";
  let hasVerifier = 1;
}

#endif // MIR_TRANSFORM_OPS_TD
