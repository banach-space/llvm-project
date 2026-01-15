// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:  -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:  --lower-vector-mask |\
// DEFINE: mlir-opt \
// DEFINE:  -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = main
// DEFINE: %{run} = mlir-runner %t -e %{entry_point} -entry-point-result=void \
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%native_mlir_runner_utils

// RUN: rm -f %t && %{compile} && %{run} | FileCheck %s

//=============================================================================
// WIP NOTES
//=============================================================================
// This is a hand-modified version of:
// * mlir/test/Integration/Dialect/Linalg/CPU/unpack-dynamic-inner-tile.mlir
//
// This implementation seems correct, but the generated output is not:
//
// [123, 123, 123, 123, 123, 123, ........., 123, 123, 123]
// [  0    0,   0,   0,   0, 123, ........., 123, 123, 123]
// [  0    0,   0,   0,   0, 123, ........., 123, 123, 123]
// ...
// [  0    0,   0,   0,   0, 123, ........., 123, 123, 123]
//
// It should be this instead:
// [123, 123, 123, 123, 123, 123, ........., 123, 123, 123]
// [123, 123, 123, 123, 123, 123, ........., 123, 123, 123]
// [123, 123, 123, 123, 123, 123, ........., 123, 123, 123]
// ...
// [123, 123, 123, 123, 123, 123, ........., 123, 123, 123]
//
// WHY? Looks like something goes wrong when vectorising linalg.unpack (look at
// the generated `vector.transfer_write`). See also the notes near `scf.for`.

//=============================================================================
// Helper methods
//=============================================================================
// Indirect calls to `@printMemref` can help bufferization (at least in some
// cases)
func.func @print(%dest: tensor<16x11xi32>) -> () {
  %dest_cast = tensor.cast %dest : tensor<16x11xi32> to tensor<*xi32>
  func.call @printMemrefI32(%dest_cast) : (tensor<*xi32>) -> ()

  return
}

//=============================================================================
// Wrapper for `linalg.unpack`
//=============================================================================
func.func @unpack(%src: tensor<2x?x8x8xi32>) {
  %dest = tensor.empty() : tensor<16x11xi32>
  %unpack = call @unpack_manually_tiled(%dest, %src) : (tensor<16x11xi32>, tensor<2x?x8x8xi32>) -> tensor<16x11xi32>

  // DEBUGGING - print the unpacked tensor
  %unpack_cast = tensor.cast %unpack : tensor<16x11xi32> to tensor<*xi32>
  call @printMemrefI32(%unpack_cast) : (tensor<*xi32>) -> ()

  return
}

//=============================================================================
// Manually tiled linalg.unpack
//=============================================================================
#map0 = affine_map<(d0) -> (16 - d0, 8)>
#map1 = affine_map<(d0) -> (11 - d0, 8)>

func.func @unpack_manually_tiled(%dest: tensor<16x11xi32>, %src: tensor<2x?x8x8xi32>) -> tensor<16x11xi32> {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  %tile_size_0 = arith.constant 8 : index
  %tile_size_1 = arith.constant 8 : index
  %ts0_x_ts1 = arith.muli %tile_size_0, %tile_size_1 : index

  %c16 = arith.constant 16 : index
  %c11 = arith.constant 11 : index
  %c9 = arith.constant 9 : index
  %c2 = arith.constant 2 : index

  %dim_1 = tensor.dim %src, %c1 : tensor<2x?x8x8xi32>

  // DEBUGGING - print the init state of the destination tensor
  // func.call @print(%dest) : (tensor<16x11xi32>) -> ()

  %0 = scf.for %idx_0 = %c0 to %c2 step %c1 iter_args(%arg0 = %dest) -> (tensor<16x11xi32>) {
    // DEBUGGING - only process the final "output" column. This helps to see that two things happen:
    //  * In row = %out_row (i.e. the current row), only the trailing 3 columns are written to (CORRECT!)
    //  * In row = %out_row + 1, 5 leading columns are populated with 0s (INCORRECT!)
    // To illustrate, the following happens:
    //        [%out_row]  ........................ 123 123 123
    //    [%out_row + 1]  0 0 0 0 0 ..........................
    // This implies issues with masking or `vector.transfer_write` lowering.
    //
    // Uncomment this line to experiment.
    %1 = scf.for %idx_1 = %c0 to %dim_1 step %c1 iter_args(%arg1 = %arg0) -> (tensor<16x11xi32>) {
    // %1 = scf.for %idx_1 = %c0 to %dim_1 step %c1 iter_args(%arg1 = %arg0) -> (tensor<16x11xi32>) {
      %2 = scf.for %idx_2 = %c0 to %c8 step %c8 iter_args(%arg2 = %arg1) -> (tensor<16x11xi32>) {
        %3 = scf.for %idx_3 = %c0 to %c8 step %c8 iter_args(%arg3 = %arg2) -> (tensor<16x11xi32>) {

          %slice_in = tensor.extract_slice %src[%idx_0, %idx_1, %idx_2, %idx_3] [1, 1, 8, 8] [1, 1, 1, 1] : tensor<2x?x8x8xi32> to tensor<1x1x8x8xi32>
          // (0, 0), (0, 8), (0, 16), ..., (0, 64)
          // (8, 0), (8, 8), (8, 16), ..., (0, 64)
          // ...
          %out_row = arith.muli %idx_0, %tile_size_0 : index
          %out_col = arith.muli %idx_1, %tile_size_1 : index
          %out_dim_0 = affine.min #map0(%out_row)
          %out_dim_1 = affine.min #map1(%out_col)

          // e.g. tensor<8x8xi32> for most tiles
          %init = tensor.empty(%out_dim_0, %out_dim_1) : tensor<?x?xi32>

          // DEBUGGING - print the indices
          // vector.print str "OUT DIM 0: "
          // vector.print %out_dim_0 : index
          // vector.print str "OUT DIM 1: "
          // vector.print %out_dim_1 : index
          // vector.print str "OUT ROW: "
          // vector.print %out_row : index
          // vector.print str "OUT COL: "
          // vector.print %out_col : index

          %unpack = linalg.unpack %slice_in
            outer_dims_perm = [0, 1]
            inner_dims_pos = [0, 1]
            inner_tiles = [8, 8] into %init : tensor<1x1x8x8xi32> -> tensor<?x?xi32>

          %slice_out = tensor.insert_slice %unpack
            into %arg3[%out_row, %out_col] [%out_dim_0, %out_dim_1] [1, 1]
            : tensor<?x?xi32> into tensor<16x11xi32>

          // DEBUGGING - print the unpacked slice
          %unpack_cast = tensor.cast %unpack : tensor<?x?xi32> to tensor<*xi32>
          func.call @printMemrefI32(%unpack_cast) : (tensor<*xi32>) -> ()

          scf.yield %slice_out : tensor<16x11xi32>
        }
        scf.yield %3 : tensor<16x11xi32>
      }
      scf.yield %2 : tensor<16x11xi32>
    }
    scf.yield %1 : tensor<16x11xi32>
  }
  return %0 : tensor<16x11xi32>
}

//=============================================================================
// MAIN entry point
//=============================================================================
func.func @main() {
  // INITIALIZE THE TENSOR TO TILE
  %A = arith.constant dense<123> : tensor<2x2x8x8xi32>
  %A_sc = tensor.cast %A: tensor<2x2x8x8xi32> to tensor<2x?x8x8xi32>

  func.call @unpack(%A_sc) : (tensor<2x?x8x8xi32>) -> ()

  return
}

//=============================================================================
// TD sequence
//=============================================================================
module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.consume}) {
    %pack = transform.structured.match ops{["linalg.unpack"]} in %module : (!transform.any_op) -> !transform.any_op

    // 1. DO NOT TILE! (we tiled manually)

    // 2. Decompose the tiled unpack Op into tensor.extract_slice + tensor.insert_slice:
    %func_op = transform.get_parent_op %pack {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">
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
