// ./onnx-mlir-opt --convert-onnx-to-krnl --convert-krnl-to-affine /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test_gemm.mlir
func.func private @test_gemm(%arg0: tensor<1x3136xf32>, %arg1: tensor<128x3136xf32>, %arg2: tensor<128xf32>) -> tensor<1x128xf32> {
    %16 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "/fc1/Gemm", transA = 0 : si64, transB = 1 : si64} : (tensor<1x3136xf32>, tensor<128x3136xf32>, tensor<128xf32>) -> tensor<1x128xf32>
    return %16 : tensor<1x128xf32>
}

// module {
//   func.func private @test_gemm(%arg0: memref<1x3136xf32>, %arg1: memref<128x3136xf32>, %arg2: memref<128xf32>) -> memref<1x128xf32> {
//     %c1 = arith.constant 1 : index
//     %c3136 = arith.constant 3136 : index
//     %c3136_0 = arith.constant 3136 : index
//     %c128 = arith.constant 128 : index
//     %c1_1 = arith.constant 1 : index
//     %c128_2 = arith.constant 128 : index
//     %alloc = memref.alloc() {alignment = 128 : i64} : memref<1x128xf32>
//     %cst = arith.constant 1.000000e+00 : f32
//     %cst_3 = arith.constant 1.000000e+00 : f32
//     %cst_4 = arith.constant 0.000000e+00 : f32
//     %0:3 = krnl.define_loops 3
//     %c0 = arith.constant 0 : index
//     krnl.iterate(%0#0, %0#1) with (%0#0 -> %arg3 = 0 to 1, %0#1 -> %arg4 = 0 to 128, %0#2 -> %arg5 = 0 to 3136){
//       %1:2 = krnl.get_induction_var_value(%0#0, %0#1) : (!krnl.loop, !krnl.loop) -> (index, index)
//       %alloca = memref.alloca() : memref<f32>
//       krnl.store %cst_4, %alloca[] : memref<f32>
//       krnl.iterate(%0#2) with (){
//         %7 = krnl.get_induction_var_value(%0#2) : (!krnl.loop) -> index
//         %8 = krnl.load %arg0[%1#0, %7] : memref<1x3136xf32>
//         %9 = krnl.load %arg1[%1#1, %7] : memref<128x3136xf32>
//         %10 = arith.mulf %8, %9 : f32
//         %11 = krnl.load %alloca[] : memref<f32>
//         %12 = arith.addf %10, %11 : f32
//         krnl.store %12, %alloca[] : memref<f32>
//       }
//       %2 = krnl.load %alloca[] : memref<f32>
//       %3 = arith.mulf %cst, %2 : f32
//       %c128_5 = arith.constant 128 : index
//       %c1_6 = arith.constant 1 : index
//       %true = arith.constant true
//       %c0_7 = arith.constant 0 : index
//       %4 = krnl.load %arg2[%1#1] : memref<128xf32>
//       %5 = arith.mulf %cst_3, %4 : f32
//       %6 = arith.addf %3, %5 : f32
//       krnl.store %6, %alloc[%1#0, %1#1] : memref<1x128xf32>
//     }
//     return %alloc : memref<1x128xf32>
//   }
// }

