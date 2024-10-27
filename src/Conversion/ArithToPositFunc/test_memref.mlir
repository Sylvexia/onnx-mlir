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