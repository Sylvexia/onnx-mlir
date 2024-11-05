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
  %1 = arith.addf %0, %arg4 : f32
  affine.store %1, %alloc[%arg0, %arg1, %arg2, %arg3] : 
    memref<32x1x3x3xf32>
  %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], 
    sizes: [1, 288], strides: [288, 1] 
    : memref<32x1x3x3xf32> to memref<1x288xf32>
  return %reinterpret_cast : memref<1x288xf32>
}