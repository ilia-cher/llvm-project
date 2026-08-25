//===- mir-opt.cpp - opt driver for the mir mirror dialect ----------------===//
//
// An mlir-opt-style driver that registers only the `mir` dialect (plus builtin):
//
//   mir-opt snippet.mlir                 # parse + print
//   mir-opt --verify-roundtrip in.mlir   # parse, print, re-parse, compare
//
//===----------------------------------------------------------------------===//

#include "MirDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mir::MirDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "mir mirror dialect opt driver\n", registry));
}
