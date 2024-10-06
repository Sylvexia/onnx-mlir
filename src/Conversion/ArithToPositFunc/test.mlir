func.func @test_arith(%arg0 : f32, %arg1 : f32) {
  %0 = arith.constant 1.0 : f32
  %1 = arith.constant 2.0 : f32
  %2 = arith.addf %arg0, %arg1 : f32
  return
}

func.func @test_arith_prop(%arg0 : f32, %arg1 : f32) {
  %0 = arith.constant 1.0 : f32
  %1 = arith.constant 2.0 : f32
  %2 = arith.addf %0, %1 : f32
  return
}

func.func @test_arith_const(%arg0 : f32, %arg1 : f32) {
  %0 = arith.constant 1.0 : f32
  %1 = arith.constant 2.0 : f32
  %2 = arith.addf %arg0, %1 : f32
  return
}

func.func @test_const(%float1: f32, %float2: f32) {
  %0 = arith.constant 1.69 : f32
  %1 = arith.constant 3.14 : f32
  return
}

func.func @test_const_return(%float1: f32, %float2: f32) -> (f32, f32) {
  %0 = arith.constant 2.68 : f32
  %1 = arith.constant 6.9 : f32
  return %0, %1 : f32, f32
}
