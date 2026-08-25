//===- MirDialect.cpp - AMDGPU MIR mirror dialect registration ------------===//
//
// Registers the `mir` dialect: its !mir.reg type, the Domain enum, and all ops.
// Enum + type classes are included before the dialect defs: with
// useDefaultTypePrinterParser, the dialect's parseType/printType route through
// the generatedType{Parser,Printer} the type classes define.
//
//===----------------------------------------------------------------------===//

#include "MirDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mir;

// Domain enum definitions (symbolize/stringify).
#include "MirOpsEnums.cpp.inc"

// !mir.reg type definitions (also emits generatedTypeParser/Printer).
#define GET_TYPEDEF_CLASSES
#include "MirOpsTypes.cpp.inc"

// Dialect definitions (constructor + default type printer/parser dispatch).
#include "MirOpsDialect.cpp.inc"

// Op definitions.
#define GET_OP_CLASSES
#include "MirOps.cpp.inc"

// Readable block labels from the terminator's attrs: `^bb<mbb>` for a primary
// block, `^bb<mbb>_synth<split>` for a split MBB's forwarder.
void FuncOp::getAsmBlockNames(OpAsmSetBlockNameFn setNameFn) {
  for (Block &blk : getBody()) {
    if (blk.empty())
      continue;
    auto term = llvm::dyn_cast<TermOp>(&blk.back());
    if (!term)
      continue;
    std::optional<uint64_t> mbb = term.getMbb();
    if (!mbb)
      continue;
    std::string name = ("bb" + llvm::Twine(*mbb)).str();
    if (std::optional<uint64_t> split = term.getSplit())
      name += ("_synth" + llvm::Twine(*split)).str();
    setNameFn(&blk, name);
  }
}

void MirDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "MirOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "MirOps.cpp.inc"
      >();
}
