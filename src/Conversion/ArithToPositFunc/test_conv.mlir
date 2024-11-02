// ./onnx-mlir-opt --convert-onnx-to-krnl /home/sylvex/onnx-mlir/src/Conversion/ArithToPositFunc/test_conv.mlir
func.func private @test_conv_no_bias_no_pad(%arg0 : tensor<1x32x14x14xf32>, %arg1 : tensor<64x32x3x3xf32>, %arg2 : tensor<64xf32>) -> tensor<1x64x14x14xf32> {
    %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "/conv2/Conv", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x32x14x14xf32>, tensor<64x32x3x3xf32>, tensor<64xf32>) -> tensor<1x64x14x14xf32>
    return %0 : tensor<1x64x14x14xf32>
}

// #map = affine_map<(d0) -> (d0 * 64)>
// #map1 = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// #map2 = affine_map<(d0, d1) -> (d0 * 32)>
// #map3 = affine_map<(d0) -> (-d0 + 1)>
// #map4 = affine_map<(d0) -> (-d0 + 1, 0)>
// #map5 = affine_map<(d0) -> (-d0 + 15)>
// #map6 = affine_map<(d0) -> (-d0 + 15, 3)>
// #map7 = affine_map<(d0, d1) -> (-d1 + 1)>
// #map8 = affine_map<(d0, d1) -> (-d1 + 1, 0)>
// #map9 = affine_map<(d0, d1) -> (-d1 + 15)>
// #map10 = affine_map<(d0, d1) -> (-d1 + 15, 3)>
// #map11 = affine_map<(d0)[s0] -> (d0 + s0)>
// #map12 = affine_map<(d0, d1)[s0, s1] -> (d1 - s1)>
// #map13 = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 - s2)>
// module {
//   func.func private @test_conv_no_bias_no_pad(%arg0: memref<1x32x14x14xf32>, %arg1: memref<64x32x3x3xf32>, %arg2: memref<64xf32>) -> memref<1x64x14x14xf32> {
//     %c3 = arith.constant 3 : index
//     %c3_0 = arith.constant 3 : index
//     %c1 = arith.constant 1 : index
//     %c1_1 = arith.constant 1 : index
//     %c1_2 = arith.constant 1 : index
//     %c1_3 = arith.constant 1 : index
//     %c1_4 = arith.constant 1 : index
//     %c64 = arith.constant 64 : index
//     %c0 = arith.constant 0 : index
//     %c1_5 = arith.constant 1 : index
//     %c14 = arith.constant 14 : index
//     %c1_6 = arith.constant 1 : index
//     %c1_7 = arith.constant 1 : index
//     %c2 = arith.constant 2 : index
//     %c3_8 = arith.constant 3 : index
//     %c2_9 = arith.constant 2 : index
//     %c16 = arith.constant 16 : index
//     %c13 = arith.constant 13 : index
//     %c14_10 = arith.constant 14 : index
//     %c14_11 = arith.constant 14 : index
//     %c1_12 = arith.constant 1 : index
//     %c1_13 = arith.constant 1 : index
//     %c2_14 = arith.constant 2 : index
//     %c3_15 = arith.constant 3 : index
//     %c2_16 = arith.constant 2 : index
//     %c16_17 = arith.constant 16 : index
//     %c13_18 = arith.constant 13 : index
//     %c14_19 = arith.constant 14 : index
//     %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x64x14x14xf32>
//     %c1_20 = arith.constant 1 : index
//     %cst = arith.constant 0.000000e+00 : f32
//     %c32 = arith.constant 32 : index
//     %c0_21 = arith.constant 0 : index
//     %c1_22 = arith.constant 1 : index
//     %0:3 = krnl.define_loops 3
//     krnl.iterate(%0#0, %0#1, %0#2) with (%0#0 -> %arg3 = 0 to 1, %0#1 -> %arg4 = 0 to 1, %0#2 -> %arg5 = 0 to 64){
//       %1:3 = krnl.get_induction_var_value(%0#0, %0#1, %0#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
//       %c64_23 = arith.constant 64 : index
//       %2 = affine.apply #map(%1#1)
//       %3 = affine.apply #map1(%1#1, %1#2)
//       %c32_24 = arith.constant 32 : index
//       %4 = affine.apply #map2(%1#1, %1#2)
//       %5:2 = krnl.define_loops 2
//       %c14_25 = arith.constant 14 : index
//       %c14_26 = arith.constant 14 : index
//       krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg6 = 0 to 14, %5#1 -> %arg7 = 0 to 14){
//         %6:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
//         %7:3 = krnl.define_loops 3
//         %c32_27 = arith.constant 32 : index
//         %c14_28 = arith.constant 14 : index
//         %c14_29 = arith.constant 14 : index
//         %c3_30 = arith.constant 3 : index
//         %c3_31 = arith.constant 3 : index
//         %c1_32 = arith.constant 1 : index
//         %c1_33 = arith.constant 1 : index
//         %c1_34 = arith.constant 1 : index
//         %8 = affine.apply #map3(%6#0)
//         %c0_35 = arith.constant 0 : index
//         %9 = affine.max #map4(%6#0)
//         %10 = affine.apply #map5(%6#0)
//         %11 = affine.min #map6(%6#0)
//         %c14_36 = arith.constant 14 : index
//         %c14_37 = arith.constant 14 : index
//         %c3_38 = arith.constant 3 : index
//         %c3_39 = arith.constant 3 : index
//         %c1_40 = arith.constant 1 : index
//         %c1_41 = arith.constant 1 : index
//         %c1_42 = arith.constant 1 : index
//         %12 = affine.apply #map7(%6#0, %6#1)
//         %c0_43 = arith.constant 0 : index
//         %13 = affine.max #map8(%6#0, %6#1)
//         %14 = affine.apply #map9(%6#0, %6#1)
//         %15 = affine.min #map10(%6#0, %6#1)
//         %16 = krnl.iterate(%7#0, %7#1, %7#2) with (%7#0 -> %arg8 = 0 to 32, %7#1 -> %arg9 = max #map4(%6#0) to min #map6(%6#0), %7#2 -> %arg10 = max #map8(%6#0, %6#1) to min #map10(%6#0, %6#1)) iter_args(%arg11 = %cst) -> (f32){
//           %19:3 = krnl.get_induction_var_value(%7#0, %7#1, %7#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
//           %20 = affine.apply #map11(%19#0)[%4]
//           %c1_44 = arith.constant 1 : index
//           %21 = affine.apply #map12(%19#0, %19#1)[%4, %8]
//           %c1_45 = arith.constant 1 : index
//           %22 = affine.apply #map13(%19#0, %19#1, %19#2)[%4, %8, %12]
//           %23 = krnl.load %arg0[%1#0, %20, %21, %22] : memref<1x32x14x14xf32>
//           %24 = krnl.load %arg1[%3, %19#0, %19#1, %19#2] : memref<64x32x3x3xf32>
//           %25 = arith.mulf %23, %24 : f32
//           %26 = arith.addf %arg11, %25 : f32
//           krnl.yield %26 : f32
//         }
//         %17 = krnl.load %arg2[%3] : memref<64xf32>
//         %18 = arith.addf %16, %17 : f32
//         krnl.store %18, %alloc[%1#0, %3, %6#0, %6#1] : memref<1x64x14x14xf32>
//       }
//     }
//     return %alloc : memref<1x64x14x14xf32>
//   }
// }

