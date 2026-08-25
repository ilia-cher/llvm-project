//===- mir-transform.cpp - MIR round-trip / transform driver --------------===//
//
// Read a MIR file, parse it into an LLVM Module plus its MachineFunctions, then
// serialize it back out as MIR, emit the `mir` mirror, or run a transform script
// over the mirror (--emit / --transform-ir).
//
// Parse sequence: createMIRParser -> parseIRModule -> createTargetMachine ->
// setDataLayout -> parseMachineFunctions. Specialized for AMDGPU: only the
// AMDGPU target is initialized.
//
//===----------------------------------------------------------------------===//

#include "MirImport.h"
#include "MirTransformOps.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

static cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input .mir file>"));
static cl::opt<std::string> OutputFilename("o", cl::init("-"),
                                           cl::value_desc("filename"),
                                           cl::desc("Output file ('-' = stdout)"));

// AMDGPU-specific overrides. Normally unnecessary: the MIR's per-function
// attributes already carry target-cpu / target-features.
static cl::opt<std::string> TripleName("mtriple", cl::init(""),
                                       cl::desc("Override target triple"));
static cl::opt<std::string> MCPU("mcpu", cl::init(""),
                                 cl::desc("Override target-cpu, e.g. gfx1250"));
static cl::opt<std::string>
    MAttrs("mattr", cl::init(""),
           cl::desc("Override target-features, e.g. +wavefrontsize32"));

// Output form: round-trip the MIR (default) or emit the `mir` mirror module.
static cl::opt<std::string>
    Emit("emit", cl::init("mir"),
         cl::value_desc("mir|mlir"),
         cl::desc("Output form: 'mir' round-trips MIR, 'mlir' emits the mirror"));

// Run an MLIR transform-dialect script over the mirror. When set, this takes
// precedence over --emit: build the mirror, then interpret the script.
static cl::opt<std::string>
    TransformIR("transform-ir", cl::init(""), cl::value_desc("script.mlir"),
                cl::desc("Run a transform-dialect script over the mirror"));

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Specialize for AMDGPU: initialize only this target (the InitializeAll*
  // aggregates would reference X86/NVPTX symbols this tool does not link).
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeAMDGPUAsmPrinter();

  cl::ParseCommandLineOptions(
      argc, argv, "mir-mirror: parse a MIR file and re-serialize it (step 1)\n");

  LLVMContext Context;
  SMDiagnostic Err;

  // 1. Parse the embedded LLVM IR module out of the MIR file.
  std::unique_ptr<MIRParser> MIR =
      createMIRParserFromFile(InputFilename, Err, Context);
  if (!MIR) {
    Err.print(argv[0], errs());
    return 1;
  }
  std::unique_ptr<Module> M = MIR->parseIRModule();
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  // 2. Resolve the triple (CLI override wins, else the module's own).
  Triple TT = TripleName.empty() ? M->getTargetTriple()
                                 : Triple(Triple::normalize(TripleName));
  if (!TripleName.empty())
    M->setTargetTriple(TT);

  std::string LookupErr;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, LookupErr);
  if (!TheTarget) {
    errs() << argv[0] << ": " << LookupErr << "\n";
    return 1;
  }

  // 3. Build the TargetMachine and reset the module's data layout from it.
  TargetOptions Options;
  std::unique_ptr<TargetMachine> TM(TheTarget->createTargetMachine(
      TT, MCPU, MAttrs, Options, /*RM=*/std::nullopt));
  if (!TM) {
    errs() << argv[0] << ": could not allocate target machine\n";
    return 1;
  }
  M->setDataLayout(TM->createDataLayout());

  // 4. Parse the machine functions into a MachineModuleInfo.
  MachineModuleInfo MMI(TM.get());
  if (MIR->parseMachineFunctions(*M, MMI)) {
    errs() << argv[0] << ": failed to parse machine functions\n";
    return 1;
  }

  // 5. Emit. Open the output file first.
  std::error_code EC;
  ToolOutputFile Out(OutputFilename, EC, sys::fs::OF_Text);
  if (EC) {
    errs() << argv[0] << ": " << EC.message() << "\n";
    return 1;
  }

  if (Emit != "mir" && Emit != "mlir") {
    errs() << argv[0] << ": --emit must be 'mir' or 'mlir'\n";
    return 1;
  }

  // The mirror is built only when a script runs or MLIR is emitted; a plain
  // `--emit=mir` needs none. The mir + transform dialects share one context for
  // interop between the mirror and the parsed script.
  bool NeedMirror = (Emit == "mlir") || !TransformIR.empty();
  mlir::DialectRegistry Registry;
  Registry.insert<mir::MirDialect, mlir::transform::TransformDialect>();
  mir::registerMirTransformExtension(Registry);
  mlir::MLIRContext MlirCtx(Registry);
  mlir::OwningOpRef<mlir::ModuleOp> Mirror;
  if (NeedMirror) {
    MlirCtx.loadAllAvailableDialects();
    Mirror = mir::importMirModule(*M, MMI, MlirCtx);
  }

  // Transform stage: apply the script to the mirror. Its console output (e.g.
  // print_mbb) is the transform's own side-channel and goes to stdout; the
  // emitted artifact below still respects -o.
  if (!TransformIR.empty()) {
    mlir::OwningOpRef<mlir::ModuleOp> TransformModule;
    if (mlir::failed(mlir::transform::detail::parseTransformModuleFromFile(
            &MlirCtx, TransformIR, TransformModule))) {
      errs() << argv[0] << ": failed to parse transform IR '" << TransformIR
             << "'\n";
      return 1;
    }

    // Locate the entry named sequence (@__transform_main).
    mlir::transform::TransformOpInterface Entry =
        mlir::transform::detail::findTransformEntryPoint(Mirror->getOperation(),
                                                         *TransformModule);
    if (!Entry) {
      errs() << argv[0] << ": no transform entry point found\n";
      return 1;
    }

    // Install the data plane on the transform state, then run the script once
    // per mir.func with that func as the payload root.
    auto StateInit = [&](mlir::transform::TransformState &State) {
      State.addExtension<mir::MirMMIExtension>(MMI, *M, outs());
    };
    mlir::transform::TransformOptions Options;
    for (mlir::Operation &Op : Mirror->getBody()->getOperations()) {
      auto Func = llvm::dyn_cast<mir::FuncOp>(&Op);
      if (!Func)
        continue;
      if (mlir::failed(mlir::transform::applyTransforms(
              Func, Entry, /*extraMapping=*/{}, Options,
              /*enforceToplevelTransformOp=*/false, StateInit))) {
        errs() << argv[0] << ": transform application failed\n";
        return 1;
      }
    }
  }

  // Emit stage: write the requested view to -o.
  if (Emit == "mlir") {
    // The (transformed) mirror module.
    Mirror->print(Out.os());
    Out.os() << "\n";
  } else {
    // Round-trip MIR: the module IR block first, then one YAML document per
    // MachineFunction (the split printMIR expects).
    printMIR(Out.os(), *M);
    for (Function &F : *M)
      if (MachineFunction *MF = MMI.getMachineFunction(F))
        printMIR(Out.os(), MMI, *MF);
  }
  Out.keep();

  return 0;
}
