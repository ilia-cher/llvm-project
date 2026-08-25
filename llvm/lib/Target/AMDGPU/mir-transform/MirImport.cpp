//===- MirImport.cpp - MachineFunction -> mir mirror module ---------------===//
//
// Structural importer. For each MachineFunction:
//   * one mir.func with a CFG region (one MLIR block per MBB);
//   * each register storage object is a mir.regmem in the entry block -- one per
//     vreg, one per shared physical file (VGPR/SGPR), one per special register;
//   * numbered physregs are mir.subreg views into their file regmem;
//   * sub-register operands become mir.subreg / mir.subreg_lo16 / _hi16 views;
//   * each non-terminator MI -> mir.asm; the trailing terminator -> mir.term
//     carrying the MBB's successor edges.
//
// Handle producers (regmem + subreg views) are emitted into the entry block in
// dependency order.
//
//===----------------------------------------------------------------------===//

#include "MirImport.h"
#include "MirDialect.h"

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

// AMDGPU-internal header: bank sizes from the subtarget model (GCNSubtarget),
// physreg encodings rebased against AMDGPU::VGPR0/SGPR0.
#include "GCNSubtarget.h"

#include "llvm/Support/MathExtras.h"

// Only mlir is pulled in wholesale; llvm types are named explicitly to avoid
// Value/DenseMap/Attribute ambiguity.
using namespace mlir;

using llvm::Function;
using llvm::MachineBasicBlock;
using llvm::MachineFunction;
using llvm::MachineInstr;
using llvm::MachineModuleInfo;
using llvm::MachineOperand;
using llvm::MachineRegisterInfo;
using llvm::Module;
using llvm::raw_string_ostream;
using llvm::Register;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;
using llvm::TargetInstrInfo;
using llvm::TargetRegisterClass;
using llvm::TargetRegisterInfo;

namespace {

// Domain of a register class, keyed off its (AMDGPU) class name.
mir::Domain domainForClassName(StringRef n) {
  if (n.contains("VGPR") || n.contains("VReg") || n.contains("AGPR") ||
      n.contains("AReg") || n.starts_with("AV") || n.starts_with("VS"))
    return mir::Domain::Vector;
  return mir::Domain::Scalar;
}

// A numbered file register: VGPR<n> / SGPR<n> / AGPR<n>, including tuples like
// SGPR0_SGPR1. Anything else (EXEC/VCC/SCC/M0/...) is a special register.
bool isNumberedFileReg(StringRef name, bool &isVector) {
  StringRef rest;
  if (name.starts_with("VGPR") || name.starts_with("AGPR")) {
    rest = name.drop_front(4);
    isVector = true;
  } else if (name.starts_with("SGPR")) {
    rest = name.drop_front(4);
    isVector = false;
  } else {
    return false;
  }
  return !rest.empty() && isdigit((unsigned char)rest[0]);
}

// Per-MachineFunction importer. One instance builds one mir.func.
struct FuncImporter {
  MLIRContext &ctx;
  MachineFunction &MF;
  const TargetRegisterInfo &TRI;
  const TargetInstrInfo &TII;
  MachineRegisterInfo &MRI;

  mir::RegType regTy;
  Location loc;

  // Insertion point for handle producers, at the top of the entry block.
  OpBuilder prologB;
  Operation *lastHandle = nullptr;
  Block *entryBlock = nullptr;

  // Insertion point for the instruction stream (per block).
  OpBuilder bodyB;

  DenseMap<const MachineBasicBlock *, Block *> mbbMap;
  DenseMap<Register, Value> baseHandle;              // reg -> base location
  DenseMap<std::pair<unsigned, unsigned>, Value> leafHandle; // (reg,sub) -> view
  Value vgprFile, sgprFile;                          // lazy file regmem

  // File model, from computeFileModel(). Bank size is the per-wave register
  // budget; enc base is the file origin (VGPR0/SGPR0). A physreg's in-file dword
  // offset is getEncodingValue(R) - encBase.
  uint64_t vgprBankSize = 1, sgprBankSize = 1;
  uint64_t vgprEncBase = 0, sgprEncBase = 0;

  FuncImporter(MLIRContext &ctx, MachineFunction &MF)
      : ctx(ctx), MF(MF), TRI(*MF.getSubtarget().getRegisterInfo()),
        TII(*MF.getSubtarget().getInstrInfo()), MRI(MF.getRegInfo()),
        regTy(mir::RegType::get(&ctx)), loc(UnknownLoc::get(&ctx)),
        prologB(&ctx), bodyB(&ctx) {}

  //--- handle producers (all emitted into the entry-block prologue) ---------//

  void positionProlog() {
    if (lastHandle)
      prologB.setInsertionPointAfter(lastHandle);
    else
      prologB.setInsertionPointToStart(entryBlock);
  }

  Value emitRegmem(mir::Domain d, uint64_t dwords, std::optional<uint64_t> align,
                  StringRef regClass) {
    positionProlog();
    OperationState st(loc, mir::RegmemOp::getOperationName());
    st.addTypes(regTy);
    st.addAttribute("domain", prologB.getI32IntegerAttr((int32_t)d));
    st.addAttribute("dwords", prologB.getI64IntegerAttr(dwords));
    if (align)
      st.addAttribute("align", prologB.getI64IntegerAttr(*align));
    if (!regClass.empty())
      st.addAttribute("regClass", prologB.getStringAttr(regClass));
    Operation *op = prologB.create(st);
    lastHandle = op;
    return op->getResult(0);
  }

  Value emitSubreg(Value base, uint64_t offset, uint64_t size) {
    positionProlog();
    OperationState st(loc, mir::SubregOp::getOperationName());
    st.addTypes(regTy);
    st.addOperands(base);
    st.addAttribute("offset", prologB.getI64IntegerAttr(offset));
    st.addAttribute("size", prologB.getI64IntegerAttr(size));
    Operation *op = prologB.create(st);
    lastHandle = op;
    return op->getResult(0);
  }

  Value emitSub16(Value base, bool hi) {
    positionProlog();
    OperationState st(loc, hi ? mir::SubregHi16Op::getOperationName()
                              : mir::SubregLo16Op::getOperationName());
    st.addTypes(regTy);
    st.addOperands(base);
    Operation *op = prologB.create(st);
    lastHandle = op;
    return op->getResult(0);
  }

  //--- register -> location handle ------------------------------------------//

  static uint64_t bitsToDwords(unsigned bits) {
    unsigned dw = (bits + 31) / 32;
    return dw ? dw : 1;
  }

  Value vgprFileRegmem() {
    if (!vgprFile)
      vgprFile = emitRegmem(mir::Domain::Vector, vgprBankSize, std::nullopt,
                           "VGPR_FILE");
    return vgprFile;
  }
  Value sgprFileRegmem() {
    if (!sgprFile)
      sgprFile = emitRegmem(mir::Domain::Scalar, sgprBankSize, std::nullopt,
                           "SGPR_FILE");
    return sgprFile;
  }

  // Compute bank sizes and encoding bases (see the file-model fields above).
  void computeFileModel() {
    const llvm::GCNSubtarget &ST = MF.getSubtarget<llvm::GCNSubtarget>();
    const llvm::Function &F = MF.getFunction();

    // Wave count from launch geometry: workgroup work-items / wavefront size.
    unsigned wgSize = ST.getFlatWorkGroupSizes(F).second;
    unsigned waveSize = ST.getWavefrontSize();
    unsigned wavesPerWG = std::max(1u, (unsigned)llvm::divideCeil(wgSize, waveSize));

    // Per-SIMD wave count = wavesPerWG / SIMDsPerCU.
    // TODO: source SIMDsPerCU from the subtarget; 4 is the gfx1250 value.
    const unsigned SIMDsPerCU = 4;
    unsigned wavesPerSIMD =
        std::max(1u, (unsigned)llvm::divideCeil(wavesPerWG, SIMDsPerCU));

    // Arch VGPR/SGPR budget at this wave count. Dynamic-VGPR block size is
    // threaded through getMaxNumVGPRs.
    unsigned dynBlock = llvm::AMDGPU::getDynamicVGPRBlockSize(F);
    if (dynBlock == 0 && ST.isDynamicVGPREnabled())
      dynBlock = ST.getDynamicVGPRBlockSize();
    vgprBankSize = ST.getMaxNumVGPRs(wavesPerSIMD, dynBlock);
    sgprBankSize = ST.getMaxNumSGPRs(wavesPerSIMD, /*Addressable=*/false);

    vgprEncBase = TRI.getEncodingValue(llvm::AMDGPU::VGPR0);
    sgprEncBase = TRI.getEncodingValue(llvm::AMDGPU::SGPR0);
    // Rebasing by a single base assumes contiguous encodings from register 0.
    assert(TRI.getEncodingValue(llvm::AMDGPU::VGPR1) - vgprEncBase == 1 &&
           "VGPR encoding must be contiguous from VGPR0");
  }

  Value makeVregHandle(Register R) {
    const TargetRegisterClass *RC = MRI.getRegClassOrNull(R);
    if (!RC)
      return emitRegmem(mir::Domain::Scalar, 1, std::nullopt, "");
    StringRef cn = TRI.getRegClassName(RC);
    uint64_t dwords = bitsToDwords(TRI.getRegSizeInBits(*RC));
    // align left unpopulated (TODO: aligned-tuple-class query).
    return emitRegmem(domainForClassName(cn), dwords, std::nullopt, cn);
  }

  Value makePhysHandle(Register R) {
    StringRef name = TRI.getName(R);
    bool isVector = false;
    if (isNumberedFileReg(name, isVector)) {
      Value file = isVector ? vgprFileRegmem() : sgprFileRegmem();
      uint64_t encBase = isVector ? vgprEncBase : sgprEncBase;
      uint64_t enc = TRI.getEncodingValue(R);
      const TargetRegisterClass *RC = TRI.getMinimalPhysRegClass(R);
      uint64_t dwords = RC ? bitsToDwords(TRI.getRegSizeInBits(*RC)) : 1;
      return emitSubreg(file, enc - encBase, dwords);
    }
    // Special register (exec/vcc/scc/m0/...): its own scalar-domain regmem.
    return emitRegmem(mir::Domain::Scalar, 1, std::nullopt, name);
  }

  Value getBase(Register R) {
    auto it = baseHandle.find(R);
    if (it != baseHandle.end())
      return it->second;
    Value v = R.isVirtual() ? makeVregHandle(R) : makePhysHandle(R);
    baseHandle[R] = v;
    return v;
  }

  uint64_t baseDwords(Register R) {
    const TargetRegisterClass *RC =
        R.isVirtual() ? MRI.getRegClassOrNull(R) : TRI.getMinimalPhysRegClass(R);
    return RC ? bitsToDwords(TRI.getRegSizeInBits(*RC)) : 1;
  }

  // Leaf location handle for a (register, subreg-index) operand pair.
  Value getLeaf(Register R, unsigned sub) {
    if (sub == 0)
      return getBase(R);
    auto key = std::make_pair(R.id(), sub);
    auto it = leafHandle.find(key);
    if (it != leafHandle.end())
      return it->second;

    Value base = getBase(R);
    unsigned offBits = TRI.getSubRegIdxOffset(sub);
    unsigned szBits = TRI.getSubRegIdxSize(sub);
    Value v;
    if (szBits == 16) {
      unsigned dwordOff = offBits / 32;
      bool hi = (offBits % 32) != 0;
      Value lane = (dwordOff == 0 && baseDwords(R) == 1)
                       ? base
                       : emitSubreg(base, dwordOff, 1);
      v = emitSub16(lane, hi);
    } else {
      v = emitSubreg(base, offBits / 32, bitsToDwords(szBits));
    }
    leafHandle[key] = v;
    return v;
  }

  //--- instructions ---------------------------------------------------------//

  void collectRegs(const MachineInstr &MI, SmallVectorImpl<Value> &defs,
                   SmallVectorImpl<Value> &uses) {
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.getReg())
        continue;
      Value h = getLeaf(MO.getReg(), MO.getSubReg());
      // Ties ignored: a def+use operand lands in defs only (TODO).
      (MO.isDef() ? defs : uses).push_back(h);
    }
  }

  void emitAsm(const MachineInstr &MI, Block *blk) {
    SmallVector<Value> defs, uses;
    collectRegs(MI, defs, uses);
    bodyB.setInsertionPointToEnd(blk);
    bodyB.create<mir::AsmOp>(loc, StringRef(TII.getName(MI.getOpcode())),
                             ValueRange(defs), ValueRange(uses));
  }

  // `mbb` is the owning MachineBasicBlock number; `split` is the chain index, set
  // only on synthetic forwarders.
  void emitTerm(StringRef mnem, ArrayRef<Value> uses, ArrayRef<Block *> succ,
                Block *blk, int64_t mbb, std::optional<int64_t> split) {
    bodyB.setInsertionPointToEnd(blk);
    OperationState st(loc, mir::TermOp::getOperationName());
    st.addAttribute("mnemonic", bodyB.getStringAttr(mnem));
    st.addOperands(uses);
    for (Block *s : succ)
      st.addSuccessors(s);
    st.addAttribute("mbb", bodyB.getI64IntegerAttr(mbb));
    if (split)
      st.addAttribute("split", bodyB.getI64IntegerAttr(*split));
    bodyB.create(st);
  }

  //--- driver ---------------------------------------------------------------//

  Operation *run(OpBuilder &moduleB) {
    computeFileModel();

    // Enabled MachineFunctionProperties flag names.
    std::string propBuf;
    raw_string_ostream os(propBuf);
    MF.getProperties().print(os);
    os.flush();
    SmallVector<Attribute> props;
    for (StringRef tok : llvm::split(StringRef(propBuf), ',')) {
      tok = tok.trim();
      if (!tok.empty())
        props.push_back(moduleB.getStringAttr(tok));
    }

    OperationState fst(loc, mir::FuncOp::getOperationName());
    fst.addAttribute("sym_name", moduleB.getStringAttr(MF.getName()));
    fst.addAttribute("props", moduleB.getArrayAttr(props));
    fst.addRegion();
    Operation *funcOp = moduleB.create(fst);

    // One MLIR block per MBB, created up front for successor wiring.
    Region &body = funcOp->getRegion(0);
    for (MachineBasicBlock &MBB : MF) {
      Block *b = new Block();
      body.push_back(b);
      mbbMap[&MBB] = b;
    }
    entryBlock = mbbMap[&MF.front()];

    for (MachineBasicBlock &MBB : MF) {
      Block *primary = mbbMap[&MBB];
      int64_t mbbNum = MBB.getNumber();

      // Non-terminators -> asm into the primary block; collect the trailing
      // terminator suffix. MLIR allows one terminator per block; a multi-terminator
      // suffix is split into a chain below.
      SmallVector<const MachineInstr *> terms;
      for (MachineInstr &MI : MBB) {
        if (MI.isTerminator())
          terms.push_back(&MI);
        else
          emitAsm(MI, primary);
      }

      // The successor not named by any explicit branch target is the layout
      // fallthrough (the last terminator's implicit not-taken edge).
      SmallPtrSet<MachineBasicBlock *, 4> explicitSet;
      for (const MachineInstr *T : terms)
        for (const MachineOperand &MO : T->operands())
          if (MO.isMBB())
            explicitSet.insert(MO.getMBB());
      Block *fallthrough = nullptr;
      for (MachineBasicBlock *S : MBB.successors())
        if (!explicitSet.contains(S)) {
          fallthrough = mbbMap[S];
          break; // at most one layout fallthrough
        }

      // No terminator MI: synthesize a fallthrough term to the layout successor.
      if (terms.empty()) {
        SmallVector<Block *> succ;
        if (fallthrough)
          succ.push_back(fallthrough);
        emitTerm(/*mnem=*/"", /*uses=*/{}, succ, primary, mbbNum, std::nullopt);
        continue;
      }

      // Chain the suffix: terms[0] stays in the primary block, each later terms[i]
      // goes in a fresh forwarder. A non-last terminator's not-taken edge points at
      // the next block in the chain; the last one's is the layout fallthrough.
      Block *cur = primary;
      for (size_t i = 0; i < terms.size(); ++i) {
        const MachineInstr &T = *terms[i];
        SmallVector<Value> uses, defsIgnored;
        collectRegs(T, defsIgnored, uses); // terminator defs dropped (TODO)

        SmallVector<Block *> succ;
        for (const MachineOperand &MO : T.operands())
          if (MO.isMBB())
            succ.push_back(mbbMap[MO.getMBB()]);

        std::optional<int64_t> split =
            i == 0 ? std::nullopt : std::optional<int64_t>((int64_t)(i - 1));

        if (i + 1 < terms.size()) {
          Block *next = new Block();
          body.getBlocks().insert(std::next(cur->getIterator()), next);
          succ.push_back(next);
          emitTerm(TII.getName(T.getOpcode()), uses, succ, cur, mbbNum, split);
          cur = next;
        } else {
          if (fallthrough)
            succ.push_back(fallthrough);
          emitTerm(TII.getName(T.getOpcode()), uses, succ, cur, mbbNum, split);
        }
      }
    }
    return funcOp;
  }
};

} // namespace

OwningOpRef<ModuleOp> mir::importMirModule(Module &M, MachineModuleInfo &MMI,
                                           MLIRContext &ctx) {
  ctx.getOrLoadDialect<mir::MirDialect>();

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder moduleB(module->getBodyRegion());
  moduleB.setInsertionPointToEnd(module->getBody());

  for (Function &F : M) {
    MachineFunction *MF = MMI.getMachineFunction(F);
    if (!MF)
      continue;
    FuncImporter FI(ctx, *MF);
    FI.run(moduleB);
  }
  return module;
}
