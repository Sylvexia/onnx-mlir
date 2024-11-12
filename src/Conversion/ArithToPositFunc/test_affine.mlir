func.func @test_affineStore(%in0: f32, %in1: f32, %arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  %alloc = memref.alloca() : memref<1x64x14x14xf32>
  affine.store %in0, %alloc[%arg0, %arg1, %arg2, %arg3] : memref<1x64x14x14xf32>
  return
}

func.func @test_affineLoad(%in0: f32, %arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  %alloc = memref.alloca() : memref<32x1x3x3xf32>
  %0 = affine.load %alloc[%arg0, %arg1, %arg2, %arg3] : memref<32x1x3x3xf32>
  %2 = arith.addf %0, %in0 : f32
  return
}

func.func @test_affineLoadStoreMixed
  (%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: f32) 
  -> memref<1x288xf32>
{
  %alloc = memref.alloca() : memref<32x1x3x3xf32>
  %0 = affine.load %alloc[%arg0, %arg1, %arg2, %arg3] : 
    memref<32x1x3x3xf32>
  %cst_0 = arith.constant 6.9 : f32
  %1 = arith.addf %0, %arg4 : f32
  %2 = arith.addf %1, %cst_0 : f32
  affine.store %2, %alloc[%arg0, %arg1, %arg2, %arg3] : 
    memref<32x1x3x3xf32>
  %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], 
    sizes: [1, 288], strides: [288, 1] 
    : memref<32x1x3x3xf32> to memref<1x288xf32>
  return %reinterpret_cast : memref<1x288xf32>
}

func.func @test_mixed(%arg0: memref<64xf32>) -> f32 {
  %cst_0 = arith.constant 6.9 : f32
  %0 = arith.constant 0 : index
  %arg3 = affine.load %arg0[%0] : memref<64xf32>
  %arg4 = arith.addf %arg3, %cst_0 : f32
  return %arg4 : f32
}

func.func @test_affineForLoop(%arg0: memref<64xf32>) -> f32 {
  %cst_0 = arith.constant 0.0 : f32
  %0 = affine.for %arg1 = 0 to 64 iter_args(%arg2 = %cst_0) -> (f32) {
    %arg3 = affine.load %arg0[%arg1] : memref<64xf32>
    %arg4 = arith.addf %arg3, %arg2 : f32
    affine.yield %arg4 : f32
  }
  return %0 : f32
}

func.func @test_affineForLoopHighDim(%arg0: f32) {
  %alloc = memref.alloc() : memref<1x64x14x14xf32>
  %cst_0 = arith.constant 0.0 : f32
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 64 {
      affine.for %arg5 = 0 to 14 {
        %0 = affine.for %arg6 = 0 to 14 iter_args(%arg7 = %arg0) -> (f32){
          affine.store %arg0, %alloc[%arg4, %arg5, %arg6, %arg3] : memref<1x64x14x14xf32>
          affine.yield %arg0 : f32
        }
      }
    }
  }
  return
}