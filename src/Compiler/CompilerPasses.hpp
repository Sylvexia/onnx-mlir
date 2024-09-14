/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- CompilerPasses.hpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for configuring and adding passes.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "mlir/Pass/PassManager.h"

namespace onnx_mlir {
// Configures passes up front based on command line options.
void configurePasses();

void addONNXToMLIRPasses(mlir::PassManager &pm, bool targetCPU);
void addONNXToKrnlPasses(mlir::PassManager &pm, int optLevel, bool enableCSE,
    bool enableInstrumentONNXSignature, std::string ONNXOpsStatFilename);
void addKrnlToAffinePasses(mlir::PassManager &pm);
void addKrnlToLLVMPasses(
    mlir::OpPassManager &pm, std::string outputNameNoExt, bool enableCSE);
InputIRLevelType determineInputIRLevel(mlir::OwningOpRef<mlir::ModuleOp> &module);
void addONNXPasses(mlir::PassManager &pm);
void addKrnlPasses(mlir::PassManager &pm);
void addAffinePasses(mlir::PassManager &pm);
void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module, mlir::PassManager &pm,
    EmissionTargetType emissionTarget, std::string outputNameNoExt);
} // namespace onnx_mlir
