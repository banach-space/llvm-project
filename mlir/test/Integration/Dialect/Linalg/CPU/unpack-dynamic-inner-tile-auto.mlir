// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:  -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:  --lower-vector-mask |\
// DEFINE: mlir-opt \
// DEFINE:  -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = main
// DEFINE: %{run} = mlir-runner %t -e %{entry_point} -entry-point-result=void \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_runner_utils

// RUN: rm -f %t && %{compile} && %{run} | FileCheck %s

/// End-to-end test for linalg.unpack where one of the inner tile sizes is
/// dynamic.

//=============================================================================
// WIP NOTES
//=============================================================================
// This is a hand-modified version of:
// * mlir/test/Integration/Dialect/Linalg/CPU/unpack-dynamic-inner-tile.mlir
//
// This implementation works correctly.

//=============================================================================
// Wrapper for `linalg.unpack`
//=============================================================================
func.func @unpack(%src: tensor<9x?x8x8xi32>) {
  %c8 = arith.constant 8 : index

  %dest = tensor.empty() : tensor<72x67xi32>

  %unpack = linalg.unpack %src
    outer_dims_perm = [0, 1]
    inner_dims_pos = [0, 1]
    inner_tiles = [8, 8] into %dest : tensor<9x?x8x8xi32> -> tensor<72x67xi32>

  %unpack_cast = tensor.cast %unpack : tensor<72x67xi32> to tensor<*xi32>

  call @printMemrefI32(%unpack_cast) : (tensor<*xi32>) -> ()

  return
}

//=============================================================================
// MAIN entry point
//=============================================================================
func.func @main() {
  // Allocate and initialise the inputs
  %c64 = arith.constant 64 : index
  %A_alloc = tensor.empty(%c64) : tensor<9x?x8x8xi32>

  %A = arith.constant dense<123> : tensor<9x9x8x8xi32>
  %A_sc = tensor.cast %A: tensor<9x9x8x8xi32> to tensor<9x?x8x8xi32>

  func.call @unpack(%A_sc) : (tensor<9x?x8x8xi32>) -> ()

  return
}

//=============================================================================
// TD sequence
//=============================================================================
module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.consume}) {
    %pack = transform.structured.match ops{["linalg.unpack"]} in %module : (!transform.any_op) -> !transform.any_op

    // 1. Tile so that we can decompose linalg.pack
    // Ops (see step 2)
    %c8 = transform.param.constant 8 : i64 -> !transform.param<i64>
    // COMMENT OUT TO MAKE THIS WORK WITH THE UPSTREAM IMPLEMENTATION!
    %tiled_pack_op_p, %loops:2 = transform.structured.tile_using_for %pack tile_sizes [%c8, 8]
       : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // %tiled_pack_op_p, %loops:4 = transform.structured.tile_using_for %pack tile_sizes [1, 1, %c8, 8]
    //    : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // 2. Decompose the tiled unpack Op into tensor.extract_slice + tensor.insert_slice:
    %func_op = transform.get_parent_op %tiled_pack_op_p {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.linalg.decompose_pack_unpack
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    // 3. Vectorize tensor.insert_slice - NOTE VECTOR SIZES
    // Vector sizes match the inner tiles in the payload IR.
    %slice = transform.structured.match ops{["tensor.insert_slice"]} in %func_op : (!transform.op<"func.func">) -> !transform.any_op
    transform.structured.vectorize %slice vector_sizes [8, 8] : !transform.any_op

    // 4. Bufferize before lowering to LLVM
    %bufferize = transform.bufferization.one_shot_bufferize %module
      {bufferize_function_boundaries=true} : (!transform.any_op) -> !transform.any_op

    // 5. Canonicalize
    %func_op_bufferized = transform.structured.match ops{["func.func"]} in %bufferize : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op_bufferized {
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">

    transform.yield
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
