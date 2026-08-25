//===- MirOps.td - AMDGPU MIR mirror dialect ----------------*- tablegen -*-===//
//
// The mirror dialect: an in-memory MLIR handle space that shadows an AMDGPU
// MachineFunction (the data plane). Structural surface only: ops carry defs/uses
// locations and CFG edges, but no effects, no MemoryEffectOpInterface, and no
// custom aliasing.
//
//===----------------------------------------------------------------------===//

#ifndef MIR_OPS_TD
#define MIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Mir_Dialect : Dialect {
  let name = "mir";
  let cppNamespace = "::mir";
  let summary = "In-memory mirror of an AMDGPU MachineFunction as a handle space";
  // Generates the !mir.reg type parser/printer dispatch.
  let useDefaultTypePrinterParser = 1;
}

class Mir_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mir_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// The one value type: a register-storage location handle. Produced by mir.regmem
// or a mir.subreg* view; consumed as a defs/uses operand by mir.asm / mir.term.
def Mir_RegType : TypeDef<Mir_Dialect, "Reg"> {
  let mnemonic = "reg";
  let summary = "A register-storage location handle (regmem or subreg view)";
}

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

// Register bank domain (gfx1250 has no AGPR domain). `$domain` prints the bare
// keyword `scalar` / `vector`.
def Mir_DomainScalar : I32EnumAttrCase<"Scalar", 0, "scalar">;
def Mir_DomainVector : I32EnumAttrCase<"Vector", 1, "vector">;
def Mir_DomainEnum
    : I32EnumAttr<"Domain", "AMDGPU register bank domain",
                  [Mir_DomainScalar, Mir_DomainVector]> {
  let cppNamespace = "::mir";
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

// One MachineFunction: a symbol plus a CFG region (one MLIR block per MBB).
// `properties` mirrors MachineFunctionProperties as a list of enabled flag names.
// Regmem objects live in the entry block and dominate every use
// (IsolatedFromAbove).
def Mir_FuncOp : Mir_Op<"func", [IsolatedFromAbove, Symbol,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockNames"]>]> {
  let summary = "One MachineFunction: a symbol plus a CFG region";
  // The ODS arg is `props`, not `properties`, which would collide with the op's
  // built-in getProperties() accessor. The printed keyword is still `properties`.
  let arguments = (ins SymbolNameAttr:$sym_name, StrArrayAttr:$props);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name `properties` $props $body attr-dict";
}

// A fresh, disjoint register-storage object, sized in 32-bit dwords. Pre-RA every
// vreg becomes its own regmem; shared physical files (VGPR/SGPR) are one regmem
// each. `align` is the optional dword alignment (absent means 1-dword); `regClass`
// optionally tags the source class (e.g. "vgpr_16"). Not Pure: two regmem objects
// are distinct storage even if identically shaped.
def Mir_RegmemOp : Mir_Op<"regmem"> {
  let summary = "A fresh disjoint register-memory object, sized in dwords";
  let arguments = (ins Mir_DomainEnum:$domain, I64Attr:$dwords,
                       OptionalAttr<I64Attr>:$align,
                       OptionalAttr<StrAttr>:$regClass);
  let results = (outs Mir_RegType:$result);
  let assemblyFormat = "$domain `,` $dwords (`align` $align^)? attr-dict";
}

// A static 32-bit-granular view into a base handle; offset and size in dwords.
// Pure: identical views may be CSE'd.
def Mir_SubregOp : Mir_Op<"subreg", [Pure]> {
  let summary = "A 32-bit-granular static view: offset/size in dwords";
  let arguments = (ins Mir_RegType:$base, I64Attr:$offset, I64Attr:$size);
  let results = (outs Mir_RegType:$result);
  let assemblyFormat = "$base `,` $offset `,` $size attr-dict";
}

// The only sub-dword views: the low / high 16 bits of one dword (true16).
def Mir_SubregLo16Op : Mir_Op<"subreg_lo16", [Pure]> {
  let summary = "Low 16 bits of one dword";
  let arguments = (ins Mir_RegType:$base);
  let results = (outs Mir_RegType:$result);
  let assemblyFormat = "$base attr-dict";
}

def Mir_SubregHi16Op : Mir_Op<"subreg_hi16", [Pure]> {
  let summary = "High 16 bits of one dword";
  let arguments = (ins Mir_RegType:$base);
  let results = (outs Mir_RegType:$result);
  let assemblyFormat = "$base attr-dict";
}

// One non-terminator machine instruction. The mnemonic is an attribute (all MIs
// share the op name `mir.asm`); the op has no SSA result -- produced registers
// are `defs` entries. The two variadic operand groups (defs, uses) require
// AttrSizedOperandSegments. Register dependences are not SSA edges; they are
// recomputed from defs/uses interval geometry. `effects`/`ties` are not modeled.
def Mir_AsmOp : Mir_Op<"asm", [AttrSizedOperandSegments]> {
  let summary = "One non-terminator machine instruction (mnemonic opaque)";
  let arguments = (ins StrAttr:$mnemonic,
                       Variadic<Mir_RegType>:$defs,
                       Variadic<Mir_RegType>:$uses);
  let assemblyFormat =
      "$mnemonic `defs` `(` $defs `)` `uses` `(` $uses `)` attr-dict";
}

// One terminator MI. A single generic op covers every terminator, with a variadic
// successor list naming real successor blocks (2 for s_cbranch, 1 for s_branch, 0
// for s_endpgm). Opaque mnemonic + explicit edges; no per-mnemonic CF modeling.
//
// Bookkeeping attrs ride on the block's sole terminator. `mbb` is the owning
// MachineBasicBlock number, present on every term. An MBB with several terminator
// MIs is split into a chain (the primary block plus synthetic forwarders, all
// sharing the same `mbb`); `split` is the 0-based chain index, present only on the
// forwarders. An empty mnemonic denotes a synthesized layout-fallthrough for a
// block with no terminator MI.
def Mir_TermOp : Mir_Op<"term", [Terminator]> {
  let summary = "One terminator MI: opaque mnemonic + explicit successors";
  let arguments = (ins StrAttr:$mnemonic, Variadic<Mir_RegType>:$uses,
                       OptionalAttr<I64Attr>:$mbb,
                       OptionalAttr<I64Attr>:$split);
  let successors = (successor VariadicSuccessor<AnySuccessor>:$dests);
  let assemblyFormat =
      "$mnemonic (`uses` `(` $uses^ `)`)? `[` $dests `]` attr-dict";
}

#endif // MIR_OPS_TD
