//===- MirTransformOps.cpp - Transform-dialect ops for the mir mirror -----===//
//
// Implementations of the `transform.mir.*` extension ops.
//
//===----------------------------------------------------------------------===//

#include "MirTransformOps.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Support/raw_ostream.h"

// AMDGPU-internal scheduler components for transform.mir.schedule.
#include "AMDGPUBarrierLatency.h"
#include "AMDGPUCoExecSchedStrategy.h"
#include "AMDGPUIGroupLP.h"
#include "GCNRegPressure.h"

using namespace mlir;

// Generated custom transform handle type (!transform.mir.mbb).
#define GET_TYPEDEF_CLASSES
#include "MirTransformOpsTypes.cpp.inc"

// Generated op definitions (added to the `transform` dialect).
#define GET_OP_CLASSES
#include "MirTransformOps.cpp.inc"

MLIR_DEFINE_EXPLICIT_TYPE_ID(mir::MirMMIExtension)

//===----------------------------------------------------------------------===//
// !transform.mir.mbb type
//===----------------------------------------------------------------------===//

// Every `!mbb` representative must be a `mir.term`. The silenceable failure reads
// as "does not match" rather than aborting the script.
DiagnosedSilenceableFailure transform::MbbHandleType::checkPayload(
    Location loc, ArrayRef<Operation *> payload) const {
  for (Operation *op : payload) {
    if (isa<mir::TermOp>(op))
      continue;
    DiagnosedSilenceableFailure diag =
        emitSilenceableError(loc)
        << "expected an !transform.mir.mbb payload op to be a mir.term "
           "representative, got '"
        << op->getName() << "'";
    diag.attachNote(op->getLoc()) << "offending operation";
    return diag;
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// transform.mir.foreach_mbb
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ForeachMbbOp::apply(transform::TransformRewriter &rewriter,
                         transform::TransformResults &results,
                         transform::TransformState &state) {
  // The target handle must point at exactly one mir.func.
  SmallVector<Operation *> funcs(
      llvm::to_vector(state.getPayloadOps(getTarget())));
  if (funcs.size() != 1)
    return emitSilenceableError()
           << "expected exactly one mir.func payload, got " << funcs.size();
  auto func = dyn_cast<mir::FuncOp>(funcs[0]);
  if (!func)
    return emitSilenceableError()
           << "expected the payload to be a mir.func, got '"
           << funcs[0]->getName() << "'";

  // One representative per MBB: the primary block's `mir.term`, i.e. the
  // first-seen terminator for each `mbb` number in layout order.
  BlockArgument bodyArg = getBody().front().getArgument(0);
  llvm::SmallDenseSet<int64_t> seen;
  SmallVector<Operation *> reps;
  for (Block &blk : func.getBody()) {
    if (blk.empty())
      continue;
    auto term = dyn_cast<mir::TermOp>(blk.back());
    if (!term || !term.getMbb())
      continue;
    if (seen.insert((int64_t)*term.getMbb()).second)
      reps.push_back(term);
  }

  // Run the body once per MBB, binding its single block argument to the
  // representative term (a size-one `!transform.mir.mbb` handle).
  for (Operation *rep : reps) {
    auto scope = state.make_region_scope(getBody());
    SmallVector<transform::MappedValue> mapped;
    mapped.push_back(rep);
    if (failed(state.mapBlockArgument(bodyArg, mapped)))
      return DiagnosedSilenceableFailure::definiteFailure();

    for (Operation &op : getBody().front().without_terminator()) {
      DiagnosedSilenceableFailure result =
          state.applyTransform(cast<transform::TransformOpInterface>(op));
      if (!result.succeeded())
        return result;
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::ForeachMbbOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Handle and payload are read, not consumed.
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::onlyReadsPayload(effects);
}

//===----------------------------------------------------------------------===//
// transform.mir.print_mbb
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::PrintMbbOp::apply(transform::TransformRewriter &rewriter,
                       transform::TransformResults &results,
                       transform::TransformState &state) {
  mir::MirMMIExtension *ext = state.getExtension<mir::MirMMIExtension>();
  if (!ext)
    return emitDefiniteFailure()
           << "no MirMMIExtension installed on the transform state "
              "(the driver must add it via applyTransforms stateInitializer)";
  llvm::MachineModuleInfo &MMI = ext->getMMI();
  llvm::raw_ostream &os = ext->getOS();

  // The handle holds one `mir.term` representative per MBB. Print each; an empty
  // handle prints nothing and succeeds.
  SmallVector<Operation *> reps(
      llvm::to_vector(state.getPayloadOps(getTarget())));
  for (Operation *rep : reps) {
    auto term = dyn_cast<mir::TermOp>(rep);
    if (!term || !term.getMbb())
      return emitSilenceableError()
             << "print_mbb handle op is not a mir.term representative";
    int64_t mbbNum = (int64_t)*term.getMbb();

    auto func = rep->getParentOfType<mir::FuncOp>();
    if (!func)
      return emitSilenceableError() << "handle op is not nested under a mir.func";
    StringRef funcSym = func.getSymName();

    // Count mirror instruction ops across the MBB's split-chain of blocks.
    size_t opCount = 0;
    llvm::SmallPtrSet<Block *, 4> blocks;
    for (Block &blk : func.getBody()) {
      if (blk.empty())
        continue;
      auto t = dyn_cast<mir::TermOp>(blk.back());
      if (!t || !t.getMbb() || (int64_t)*t.getMbb() != mbbNum)
        continue;
      blocks.insert(&blk);
      for (Operation &op : blk)
        if (isa<mir::AsmOp, mir::TermOp>(op))
          ++opCount;
    }

    // Dump the numbered MBB from the data plane. MMI.getModule() is null in the
    // direct-parse path; the extension carries the Module.
    const llvm::Function *F = ext->getModule().getFunction(funcSym);
    llvm::MachineFunction *MF = F ? MMI.getMachineFunction(*F) : nullptr;
    if (!MF)
      return emitSilenceableError()
             << "no MachineFunction for symbol '" << funcSym << "'";
    llvm::MachineBasicBlock *MBB = MF->getBlockNumbered((unsigned)mbbNum);
    if (!MBB)
      return emitSilenceableError()
             << "no MBB numbered " << mbbNum << " in '" << funcSym << "'";

    os << "==== MBB " << mbbNum << " of @" << funcSym << " ====\n";
    os << "   " << opCount << " instruction op(s) across " << blocks.size()
       << " MLIR block(s)\n";
    os << "   " << std::distance(MBB->begin(), MBB->end())
       << " MachineInstr(s)\n";
    for (llvm::MachineInstr &MI : *MBB) {
      std::string line;
      llvm::raw_string_ostream ss(line);
      MI.print(ss, /*IsStandalone=*/false, /*SkipOpers=*/false,
               /*SkipDebugLoc=*/true, /*AddNewLine=*/false);
      os << "     " << line << "\n";
    }
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::PrintMbbOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::onlyReadsPayload(effects);
}

//===----------------------------------------------------------------------===//
// transform.mir.get_mbbs
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GetMbbsOp::apply(transform::TransformRewriter &rewriter,
                            transform::TransformResults &results,
                            transform::TransformState &state) {
  SmallVector<Operation *> funcs(
      llvm::to_vector(state.getPayloadOps(getTarget())));
  if (funcs.size() != 1)
    return emitSilenceableError()
           << "expected exactly one mir.func payload, got " << funcs.size();
  auto func = dyn_cast<mir::FuncOp>(funcs[0]);
  if (!func)
    return emitSilenceableError()
           << "expected the payload to be a mir.func, got '"
           << funcs[0]->getName() << "'";

  // One representative per MBB: the primary block's terminator, in layout order.
  llvm::SmallDenseSet<int64_t> seen;
  SmallVector<Operation *> reps;
  for (Block &blk : func.getBody()) {
    if (blk.empty())
      continue;
    auto term = dyn_cast<mir::TermOp>(blk.back());
    if (!term || !term.getMbb())
      continue;
    if (seen.insert((int64_t)*term.getMbb()).second)
      reps.push_back(term);
  }
  results.set(cast<OpResult>(getResult()), reps);
  return DiagnosedSilenceableFailure::success();
}

void transform::GetMbbsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::onlyReadsPayload(effects);
}

//===----------------------------------------------------------------------===//
// transform.mir.collect_mbbs
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::CollectMbbsOp::apply(transform::TransformRewriter &rewriter,
                                transform::TransformResults &results,
                                transform::TransformState &state) {
  auto matcher = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
      getOperation(), getMatcher());
  if (!matcher || matcher.isExternal())
    return emitDefiniteFailure()
           << "unresolved matcher symbol " << getMatcher();

  Block &matcherBlock = matcher.getFunctionBody().front();

  // Run the matcher once per representative in the input handle.
  SmallVector<transform::MappedValue> collected;
  for (Operation *rep : state.getPayloadOps(getTarget())) {
    // Fresh scope: discard the matcher's result mappings before the next rep.
    auto matchScope = state.make_region_scope(*matcherBlock.getParent());
    if (failed(state.mapBlockArguments(matcherBlock.getArgument(0),
                                       ArrayRef<Operation *>(rep))))
      return emitDefiniteFailure() << "failed to bind the matcher argument";

    // Apply each matcher op in order. A silenceable failure means "does not
    // match" and skips this MBB; a definite failure aborts.
    DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
    for (Operation &op : matcherBlock.without_terminator()) {
      diag = state.applyTransform(cast<transform::TransformOpInterface>(op));
      if (!diag.succeeded())
        break;
    }
    if (diag.isDefiniteFailure())
      return diag;
    if (diag.isSilenceableFailure())
      continue;

    // Accepted: forward the matcher terminator's yield, which must be exactly
    // one !mbb mapping to one payload op.
    SmallVector<SmallVector<transform::MappedValue>> yielded;
    transform::detail::prepareValueMappings(
        yielded, matcherBlock.getTerminator()->getOperands(), state);
    if (yielded.size() != 1 || yielded.front().size() != 1)
      return emitSilenceableError()
             << "matcher must yield exactly one MBB representative";
    collected.push_back(yielded.front().front());
  }

  results.setMappedValues(cast<OpResult>(getResult()), collected);
  return DiagnosedSilenceableFailure::success();
}

void transform::CollectMbbsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::onlyReadsPayload(effects);
}

LogicalResult transform::CollectMbbsOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  auto matcherSymbol = dyn_cast_or_null<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(getOperation(), getMatcher()));
  if (!matcherSymbol ||
      !isa<transform::TransformOpInterface>(matcherSymbol.getOperation()))
    return emitError() << "unresolved matcher symbol " << getMatcher();

  ArrayRef<Type> argumentTypes = matcherSymbol.getArgumentTypes();
  if (argumentTypes.size() != 1 ||
      !isa<transform::TransformHandleTypeInterface>(argumentTypes[0]))
    return emitError()
           << "expected the matcher to take one handle argument";
  if (!matcherSymbol.getArgAttr(
          0, transform::TransformDialect::kArgReadOnlyAttrName))
    return emitError() << "expected the matcher argument to be marked readonly";

  ArrayRef<Type> resultTypes = matcherSymbol.getResultTypes();
  if (resultTypes.size() != 1 ||
      !isa<transform::TransformHandleTypeInterface>(resultTypes[0]))
    return emitError() << "expected the matcher to yield one handle result";

  return success();
}

//===----------------------------------------------------------------------===//
// transform.mir.mbb_ops
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MbbOpsOp::apply(transform::TransformRewriter &rewriter,
                           transform::TransformResults &results,
                           transform::TransformState &state) {
  SmallVector<Operation *> reps(
      llvm::to_vector(state.getPayloadOps(getTarget())));
  // Widen each representative to its MBB's instruction ops (asm+term, across the
  // split-chain); dedup guards against duplicate representatives.
  SmallVector<Operation *> out;
  llvm::SmallPtrSet<Operation *, 32> added;
  for (Operation *rep : reps) {
    auto term = dyn_cast<mir::TermOp>(rep);
    if (!term || !term.getMbb())
      return emitSilenceableError()
             << "mbb handle op is not a mir.term representative";
    int64_t mbbNum = (int64_t)*term.getMbb();
    auto func = rep->getParentOfType<mir::FuncOp>();
    if (!func)
      return emitSilenceableError()
             << "mbb representative is not nested under a mir.func";
    for (Block &blk : func.getBody()) {
      if (blk.empty())
        continue;
      auto t = dyn_cast<mir::TermOp>(blk.back());
      if (!t || !t.getMbb() || (int64_t)*t.getMbb() != mbbNum)
        continue;
      for (Operation &op : blk)
        if (isa<mir::AsmOp, mir::TermOp>(op))
          if (added.insert(&op).second)
            out.push_back(&op);
    }
  }
  results.set(cast<OpResult>(getResult()), out);
  return DiagnosedSilenceableFailure::success();
}

void transform::MbbOpsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::onlyReadsPayload(effects);
}

//===----------------------------------------------------------------------===//
// transform.mir.match_asm
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MatchAsmOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
                             transform::TransformState &state) {
  StringRef needle = getContains();
  SmallVector<Operation *> matched;
  // Walk each root for mir.asm ops whose mnemonic contains `needle`. A mir.func
  // root is searched recursively; a bare op set is tested in place.
  for (Operation *root : state.getPayloadOps(getTarget()))
    root->walk([&](mir::AsmOp asmOp) {
      if (asmOp.getMnemonic().contains(needle))
        matched.push_back(asmOp);
    });
  results.set(cast<OpResult>(getResult()), matched);
  return DiagnosedSilenceableFailure::success();
}

void transform::MatchAsmOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::onlyReadsPayload(effects);
}

//===----------------------------------------------------------------------===//
// transform.mir.parent_mbb
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ParentMbbOp::apply(transform::TransformRewriter &rewriter,
                              transform::TransformResults &results,
                              transform::TransformState &state) {
  SmallVector<Operation *> groupOps(
      llvm::to_vector(state.getPayloadOps(getTarget())));
  if (groupOps.empty())
    return emitSilenceableError() << "group handle is empty";

  // Recover the single owning MBB number from the ops' block terminators.
  std::optional<int64_t> mbbNum;
  for (Operation *op : groupOps) {
    Block *blk = op->getBlock();
    auto term = dyn_cast<mir::TermOp>(blk->back());
    if (!term || !term.getMbb())
      return emitSilenceableError()
             << "group op is not inside a mir MBB block";
    int64_t n = (int64_t)*term.getMbb();
    if (mbbNum && *mbbNum != n)
      return emitSilenceableError()
             << "group spans multiple MBBs (" << *mbbNum << " and " << n << ")";
    mbbNum = n;
  }

  auto func = groupOps.front()->getParentOfType<mir::FuncOp>();
  if (!func)
    return emitSilenceableError() << "group op is not nested under a mir.func";

  // Return the MBB's primary block terminator as the single !mbb representative.
  Operation *rep = nullptr;
  for (Block &blk : func.getBody()) {
    if (blk.empty())
      continue;
    auto term = dyn_cast<mir::TermOp>(blk.back());
    if (!term || !term.getMbb() || (int64_t)*term.getMbb() != *mbbNum)
      continue;
    rep = term;
    break;
  }
  if (!rep)
    return emitSilenceableError()
           << "could not find the terminator of MBB " << *mbbNum;
  SmallVector<Operation *> reps{rep};
  results.set(cast<OpResult>(getResult()), reps);
  return DiagnosedSilenceableFailure::success();
}

void transform::ParentMbbOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::onlyReadsPayload(effects);
}

//===----------------------------------------------------------------------===//
// transform.mir.split
//===----------------------------------------------------------------------===//

namespace {
// buildSchedGraph needs a concrete ScheduleDAGInstrs. Used only for the
// dependency graph, never to schedule; schedule() is a no-op.
class ConeDAG : public llvm::ScheduleDAGInstrs {
public:
  explicit ConeDAG(llvm::MachineFunction &MF)
      : llvm::ScheduleDAGInstrs(MF, /*MLI=*/nullptr,
                                /*RemoveKillFlags=*/false) {}
  void schedule() override {}
};
} // namespace

DiagnosedSilenceableFailure
transform::SplitOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {
  mir::MirMMIExtension *ext = state.getExtension<mir::MirMMIExtension>();
  if (!ext)
    return emitDefiniteFailure() << "no MirMMIExtension installed on the state";
  llvm::MachineModuleInfo &MMI = ext->getMMI();

  //---- 1. Group ops: non-empty, all mir.asm, all in one (primary) block. ----
  SmallVector<Operation *> groupOps(
      llvm::to_vector(state.getPayloadOps(getGroup())));
  if (groupOps.empty())
    return emitSilenceableError() << "group must not be empty";
  Block *primary = groupOps.front()->getBlock();
  for (Operation *op : groupOps) {
    if (!isa<mir::AsmOp>(op))
      return emitSilenceableError()
             << "group op is not a mir.asm (terminators are not allowed)";
    if (op->getBlock() != primary)
      return emitSilenceableError() << "group ops span multiple blocks";
  }

  //---- 2. Owning func + MBB number (from the primary block's terminator). ----
  auto func = groupOps.front()->getParentOfType<mir::FuncOp>();
  if (!func)
    return emitSilenceableError() << "group is not nested under a mir.func";
  StringRef funcSym = func.getSymName();
  auto primaryTerm = dyn_cast<mir::TermOp>(primary->back());
  if (!primaryTerm || !primaryTerm.getMbb())
    return emitSilenceableError()
           << "the group's block has no mir.term carrying an mbb number";
  int64_t mbbNum = (int64_t)*primaryTerm.getMbb();

  //---- 3. Resolve the real MachineFunction + MachineBasicBlock. ----
  const llvm::Function *F = ext->getModule().getFunction(funcSym);
  llvm::MachineFunction *MF = F ? MMI.getMachineFunction(*F) : nullptr;
  if (!MF)
    return emitSilenceableError()
           << "no MachineFunction for symbol '" << funcSym << "'";
  llvm::MachineBasicBlock *MBB = MF->getBlockNumbered((unsigned)mbbNum);
  if (!MBB)
    return emitSilenceableError() << "no MBB numbered " << mbbNum;

  //---- 4. Positional zip: primary-block asm ops <-> non-terminator MIs. ----
  // Non-terminator MIs map 1:1 to primary-block asm ops in layout order.
  SmallVector<mir::AsmOp> asmOps;
  for (Operation &op : *primary)
    if (auto a = dyn_cast<mir::AsmOp>(op))
      asmOps.push_back(a);
  SmallVector<llvm::MachineInstr *> nonTermMIs;
  for (llvm::MachineInstr &MI : *MBB)
    if (!MI.isTerminator())
      nonTermMIs.push_back(&MI);
  if (asmOps.size() != nonTermMIs.size())
    return emitDefiniteFailure()
           << "mirror/data-plane mismatch: " << asmOps.size()
           << " asm ops vs " << nonTermMIs.size() << " non-terminator MIs";
  llvm::DenseMap<llvm::MachineInstr *, Operation *> mi2op;
  llvm::DenseMap<Operation *, llvm::MachineInstr *> op2mi;
  for (size_t i = 0, e = asmOps.size(); i < e; ++i) {
    mi2op[nonTermMIs[i]] = asmOps[i];
    op2mi[asmOps[i]] = nonTermMIs[i];
  }

  // Group ops -> seed MIs.
  SmallVector<llvm::MachineInstr *> groupMIs;
  for (Operation *op : groupOps) {
    auto it = op2mi.find(op);
    if (it == op2mi.end())
      return emitSilenceableError()
             << "group op does not belong to the target MBB";
    groupMIs.push_back(it->second);
  }

  //---- 5. Predecessor check: only fallthrough preds are supported here. ----
  for (llvm::MachineBasicBlock *P : MBB->predecessors())
    for (llvm::MachineInstr &T : P->terminators())
      for (const llvm::MachineOperand &MO : T.operands())
        if (MO.isMBB() && MO.getMBB() == MBB)
          return emitSilenceableError()
                 << "predecessor bb." << P->getNumber()
                 << " reaches the split MBB by an explicit branch "
                    "(not yet supported)";

  //---- 6. Dependency cone via buildSchedGraph (transitive Preds closure). ----
  // Driver sequence: startBlock -> enterRegion -> build -> exitRegion ->
  // finishBlock.
  ConeDAG dag(*MF);
  llvm::MachineBasicBlock::iterator RegionEnd = MBB->getFirstTerminator();
  dag.startBlock(MBB);
  dag.enterRegion(MBB, MBB->begin(), RegionEnd,
                  std::distance(MBB->begin(), RegionEnd));
  dag.buildSchedGraph(/*AA=*/nullptr);

  llvm::DenseSet<llvm::SUnit *> coneSU;
  SmallVector<llvm::SUnit *> worklist;
  for (llvm::MachineInstr *MI : groupMIs) {
    llvm::SUnit *su = dag.getSUnit(MI);
    if (!su)
      return emitDefiniteFailure() << "no SUnit for a group instruction";
    if (coneSU.insert(su).second)
      worklist.push_back(su);
  }
  while (!worklist.empty()) {
    llvm::SUnit *su = worklist.pop_back_val();
    for (const llvm::SDep &pred : su->Preds) {
      llvm::SUnit *p = pred.getSUnit();
      if (!p || !p->getInstr())
        continue; // skip EntrySU / boundary nodes
      if (coneSU.insert(p).second)
        worklist.push_back(p);
    }
  }
  dag.exitRegion();
  dag.finishBlock();

  llvm::DenseSet<llvm::MachineInstr *> coneMIs;
  llvm::SmallPtrSet<Operation *, 16> coneOpSet;
  for (llvm::SUnit *su : coneSU) {
    llvm::MachineInstr *MI = su->getInstr();
    coneMIs.insert(MI);
    coneOpSet.insert(mi2op[MI]);
  }

  //---- 7. Data-plane surgery: a new predecessor holds the cone. ----
  llvm::MachineBasicBlock *NewPred =
      MF->CreateMachineBasicBlock(MBB->getBasicBlock());
  MF->insert(MBB->getIterator(), NewPred); // layout: NewPred right before MBB
  // Move cone MIs into NewPred, preserving their original relative order.
  for (llvm::MachineInstr &MI : llvm::make_early_inc_range(*MBB))
    if (coneMIs.contains(&MI))
      NewPred->splice(NewPred->end(), MBB, MI.getIterator());
  // Redirect MBB's (fallthrough) predecessors to NewPred; NewPred -> MBB.
  for (llvm::MachineBasicBlock *P :
       SmallVector<llvm::MachineBasicBlock *>(MBB->predecessors()))
    P->replaceSuccessor(MBB, NewPred);
  NewPred->addSuccessor(MBB);
  int64_t newPredNum = NewPred->getNumber();

  //---- 8. Mirror-plane surgery (only MOVE ops + fix successors). ----
  Location loc = func->getLoc();
  Block *newBlk = new Block();
  primary->getParent()->getBlocks().insert(primary->getIterator(), newBlk);
  // Move cone asm ops into newBlk, preserving order (matches the MI move above).
  for (Operation &op : llvm::make_early_inc_range(*primary))
    if (coneOpSet.contains(&op))
      op.moveBefore(newBlk, newBlk->end());
  // Synthesize the fallthrough terminator: mir.term "" [primary] {mbb=newPred}.
  OpBuilder b(func.getContext());
  b.setInsertionPointToEnd(newBlk);
  OperationState st(loc, mir::TermOp::getOperationName());
  st.addAttribute("mnemonic", b.getStringAttr(""));
  st.addSuccessors(primary);
  st.addAttribute("mbb", b.getI64IntegerAttr(newPredNum));
  b.create(st);
  // Redirect every mirror edge into `primary` (except newBlk's own) to newBlk.
  for (Block &blk : func.getBody()) {
    if (&blk == newBlk || blk.empty())
      continue;
    Operation *term = &blk.back();
    for (unsigned i = 0, e = term->getNumSuccessors(); i < e; ++i)
      if (term->getSuccessor(i) == primary)
        term->setSuccessor(newBlk, i);
  }

  //---- 9. Handle bookkeeping. ----
  // `$group` is consumed; each result block is returned as one !mbb
  // representative (its `mir.term`). No payload op is erased; widen a result with
  // `mbb_ops` to reach the moved ops.
  SmallVector<Operation *> predReps{&newBlk->back()};
  results.set(cast<OpResult>(getPred()), predReps);

  SmallVector<Operation *> remReps{&primary->back()};
  results.set(cast<OpResult>(getRemainder()), remReps);
  return DiagnosedSilenceableFailure::success();
}

void transform::SplitOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getGroupMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// transform.mir.schedule
//===----------------------------------------------------------------------===//

LogicalResult transform::ScheduleOp::verify() {
  if (getScheduler() && *getScheduler() != "coexec")
    return emitOpError("only the 'coexec' scheduler is supported, got '")
           << *getScheduler() << "'";
  return success();
}

DiagnosedSilenceableFailure
transform::ScheduleOp::apply(transform::TransformRewriter &rewriter,
                             transform::TransformResults &results,
                             transform::TransformState &state) {
  mir::MirMMIExtension *ext = state.getExtension<mir::MirMMIExtension>();
  if (!ext)
    return emitDefiniteFailure() << "no MirMMIExtension installed on the state";
  llvm::MachineModuleInfo &MMI = ext->getMMI();

  //---- 1. Recover the owning mir.func + MBB number from the handle. ----
  // The handle is a single !mbb representative; empty or multi-MBB is a
  // silenceable script error.
  SmallVector<Operation *> reps(
      llvm::to_vector(state.getPayloadOps(getTarget())));
  if (reps.empty())
    return emitSilenceableError() << "schedule handle is empty";
  if (reps.size() != 1)
    return emitSilenceableError()
           << "schedule expects exactly one MBB, got " << reps.size();
  auto termRep = dyn_cast<mir::TermOp>(reps.front());
  if (!termRep || !termRep.getMbb())
    return emitSilenceableError()
           << "schedule handle op is not a mir.term representative";
  std::optional<int64_t> mbbNum = (int64_t)*termRep.getMbb();
  Block *primary = termRep->getBlock();
  auto func = termRep->getParentOfType<mir::FuncOp>();
  if (!func)
    return emitSilenceableError() << "handle op is not nested under a mir.func";
  StringRef funcSym = func.getSymName();

  //---- 2. Resolve the real MachineFunction + MachineBasicBlock. ----
  const llvm::Function *F = ext->getModule().getFunction(funcSym);
  llvm::MachineFunction *MF = F ? MMI.getMachineFunction(*F) : nullptr;
  if (!MF)
    return emitSilenceableError()
           << "no MachineFunction for symbol '" << funcSym << "'";
  llvm::MachineBasicBlock *MBB = MF->getBlockNumbered((unsigned)*mbbNum);
  if (!MBB)
    return emitSilenceableError() << "no MBB numbered " << *mbbNum;

  //---- 3. Positional zip (PRE-schedule): primary asm ops <-> non-term MIs. ----
  // MachineInstr identity is stable across scheduling; this MI*->op map stays
  // valid for the reconcile.
  SmallVector<mir::AsmOp> asmOps;
  for (Operation &op : *primary)
    if (auto a = dyn_cast<mir::AsmOp>(op))
      asmOps.push_back(a);
  SmallVector<llvm::MachineInstr *> nonTermMIs;
  for (llvm::MachineInstr &MI : *MBB)
    if (!MI.isTerminator())
      nonTermMIs.push_back(&MI);
  if (asmOps.size() != nonTermMIs.size())
    return emitDefiniteFailure()
           << "mirror/data-plane mismatch: " << asmOps.size()
           << " asm ops vs " << nonTermMIs.size() << " non-terminator MIs";
  llvm::DenseMap<llvm::MachineInstr *, Operation *> mi2op;
  for (size_t i = 0, e = asmOps.size(); i < e; ++i)
    mi2op[nonTermMIs[i]] = asmOps[i];

  //---- 4. Build the analyses (recompute per call; no PassManager). ----
  // LiveIntervals cannot be stack-constructed (pass-manager-only); a minimal
  // MachineFunction analysis manager provides it and MachineLoopInfo. The
  // registered analyses (PassInstrumentation, SlotIndexes, MachineDominatorTree,
  // LiveIntervals, MachineLoop) are all in LLVMCodeGen. RegisterClassInfo is
  // hand-built.
  llvm::MachineFunctionAnalysisManager MFAM;
  MFAM.registerPass([&] { return llvm::PassInstrumentationAnalysis(); });
  MFAM.registerPass([&] { return llvm::SlotIndexesAnalysis(); });
  MFAM.registerPass([&] { return llvm::MachineDominatorTreeAnalysis(); });
  MFAM.registerPass([&] { return llvm::LiveIntervalsAnalysis(); });
  MFAM.registerPass([&] { return llvm::MachineLoopAnalysis(); });

  llvm::LiveIntervals &LIS = MFAM.getResult<llvm::LiveIntervalsAnalysis>(*MF);
  llvm::MachineLoopInfo &MLI = MFAM.getResult<llvm::MachineLoopAnalysis>(*MF);
  llvm::RegisterClassInfo RCI;
  RCI.runOnMachineFunction(*MF);

  llvm::MachineSchedContext Ctx;
  Ctx.MF = MF;
  Ctx.MLI = &MLI;
  Ctx.TM = &MF->getTarget();
  Ctx.AA = nullptr;
  Ctx.LIS = &LIS;
  Ctx.RegClassInfo = &RCI;

  //---- 5. CoExec strategy + plain DAG + the two coexec mutations. ----
  // Plain ScheduleDAGMILive (not GCNScheduleDAGMILive); the strategy uses the DAG
  // generically. Mutation order matches createGCNCoExecMachineScheduler:
  // IGroupLP(Initial) then BarrierLatency.
  auto Strat = std::make_unique<llvm::AMDGPUCoExecSchedStrategy>(&Ctx);
  llvm::AMDGPUCoExecSchedStrategy *S = Strat.get();
  llvm::ScheduleDAGMILive DAG(&Ctx, std::move(Strat));
  DAG.addMutation(llvm::createIGroupLPDAGMutation(
      llvm::AMDGPU::SchedulingPhase::Initial));
  DAG.addMutation(llvm::createAMDGPUBarrierLatencyDAGMutation(MF));

  //---- 6. Drive exactly this block's single region. ----
  llvm::MachineBasicBlock::iterator RegionBegin = MBB->begin();
  llvm::MachineBasicBlock::iterator RegionEnd = MBB->getFirstTerminator();
  unsigned NumInstrs = std::distance(RegionBegin, RegionEnd);
  if (NumInstrs == 0)
    return DiagnosedSilenceableFailure::success();

  DAG.startBlock(MBB);
  DAG.enterRegion(MBB, RegionBegin, RegionEnd, NumInstrs);

  // Seed the GCN RP trackers: downward from the region's live-ins, upward from
  // its live-outs.
  if (S->useGCNTrackers()) {
    llvm::MachineInstr &FirstMI = *RegionBegin;
    llvm::MachineInstr &LastMI = *std::prev(RegionEnd);
    S->getDownwardTracker()->reset(MF->getRegInfo(),
                                   llvm::getLiveRegsBefore(FirstMI, LIS));
    S->getUpwardTracker()->reset(MF->getRegInfo(),
                                 llvm::getLiveRegsAfter(LastMI, LIS));
  }

  DAG.schedule(); // real reorder
  DAG.exitRegion();
  DAG.finishBlock();

  //---- 7. Reconcile the mirror: permute asm ops to the new MI order. ----
  // Move each MI's asm op before the terminator, in the reordered sequence.
  Operation *term = &primary->back();
  for (llvm::MachineInstr &MI : *MBB) {
    if (MI.isTerminator())
      continue;
    auto it = mi2op.find(&MI);
    if (it != mi2op.end())
      it->second->moveBefore(term);
  }
  return DiagnosedSilenceableFailure::success();
}

void transform::ScheduleOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  // Handle read, not consumed; payload mutated.
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// Dialect extension registration
//===----------------------------------------------------------------------===//

namespace {
class MirTransformDialectExtension
    : public transform::TransformDialectExtension<
          MirTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MirTransformDialectExtension)

  using Base::Base;

  void init() {
    // The payload ops live in the mir dialect, loaded wherever these ops run.
    declareDependentDialect<mir::MirDialect>();

    // Register the !transform.mir.mbb handle type. The (void)&generated* lines
    // silence unused-static warnings for its unused parser/printer.
    registerTypes<
#define GET_TYPEDEF_LIST
#include "MirTransformOpsTypes.cpp.inc"
        >();
    (void)&generatedTypeParser;
    (void)&generatedTypePrinter;

    registerTransformOps<
#define GET_OP_LIST
#include "MirTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mir::registerMirTransformExtension(DialectRegistry &registry) {
  registry.addExtensions<MirTransformDialectExtension>();
}
