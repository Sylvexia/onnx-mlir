func.func @test_memrefReturn(%arg0: memref<32x1x3x3xf32>) -> memref<32x1x3x3xf32> {
  return %arg0 : memref<32x1x3x3xf32>
}

func.func @test_memrefAlloca(%arg0: memref<8x64xf32>) -> memref<8x64xf32> {
  %0 = memref.alloca() : memref<8x64xf32>
  return %0 : memref<8x64xf32>
}

func.func @test_memrefAlloca2(%arg0: memref<i32>) -> memref<i32> {
  %alloca = memref.alloca() : memref<i32>
  return %alloca : memref<i32>
}

func.func @test_memrefLoad(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> f32 {
  %alloc_4 = memref.alloca() : memref<1x64x14x14xf32>
  %0 = memref.load %alloc_4[%arg0, %arg0, %arg0, %arg0] : memref<1x64x14x14xf32>
  // %0 = arith.constant 1.0 : f32
  return %0 : f32
}

func.func @test_memrefreinterpret(%arg0: memref<1x64x7x7xf32>) -> memref<1x3136xf32> {
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 3136], strides: [3136, 1] : memref<1x64x7x7xf32> to memref<1x3136xf32>
  return %reinterpret_cast : memref<1x3136xf32>
}