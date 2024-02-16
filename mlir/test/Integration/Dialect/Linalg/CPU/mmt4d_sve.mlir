// DEFINE: %{compile} =  mlir-opt %s \
// DEFINE:    -transform-interpreter -test-transform-dialect-erase-schedule \
// DEFINE:    -one-shot-bufferize -func-bufferize -cse -canonicalize -convert-vector-to-scf -test-lower-to-llvm -o %t
// DEFINE: %{entry_point} = matmul_f32
// DEFINE: %{run} = %mcr_aarch64_cmd %t -e %{entry_point} -entry-point-result=void --march=aarch64 --mattr="+sve"\
// DEFINE:    -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils

// RUN: %{compile}

// RUN: %{run} | FileCheck %s --check-prefix=F32

// REDEFINE: %{entry_point} = matmul_mixed_ty
// RUN: %{run} | FileCheck %s --check-prefix=MIXED

func.func @matmul_f32(%A : tensor<?x?x?x?xf32>, %B : tensor<?x?x?x?xf32>, %C_dyn : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
{
  %C_out = linalg.mmt4d ins(%A, %B: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%C_dyn: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  return %C_out : tensor<?x?x?x?xf32>
}

module @transforms attributes { transform.with_named_sequence } {
  // Entry point. This takes as the only argument the root operation (typically
  // pass root) given to the transform interpreter.
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
   %mmt4d = transform.collect_matching @match_mmt4d in %module : (!transform.any_op) -> (!transform.any_op)
   %func = transform.get_parent_op %mmt4d {isolated_from_above} : (!transform.any_op) -> !transform.op<"func.func">

   // Step 1: Tile
   // Tile parallel dims
   %tiled_linalg_op_p, %loops:4 = transform.structured.tile_using_for %mmt4d[1, 1, 0, 3, 3, 0]
     : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
   // Tile reduction dims
   %tiled_linalg_op_r, %loops2:2 = transform.structured.tile_using_for %tiled_linalg_op_p[0, 0, 1, 0, 0, 1]
     : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

   // Step 2: Vectorize
   transform.structured.vectorize %tiled_linalg_op_r vector_sizes [1, 1, 1, 8, 8, 1] : !transform.any_op


   // Step 3: Simplify
   // vector.multi_reduction --> vector.contract
   // Generates a 6-dim vector.contract with the dim matching the original MMT4D Op
   // and with the following split into parallel and reduction dims:
   //    * parallel, parallel, reduction, parallel, parallel, reduction
   transform.apply_patterns to %func {
     transform.apply_patterns.vector.reduction_to_contract
     transform.apply_patterns.vector.transfer_permutation_patterns
     transform.apply_patterns.vector.lower_create_mask
     transform.apply_patterns.vector.lower_masked_transfers
     // Reduce the rank of xfer ops. This transforms vector.contract to be
     // more matmul-like and to enable the lowering to outer product Ops.
   } : !transform.op<"func.func">

   // Hoisting and LICM - not strictly required
   %func_h = transform.structured.hoist_redundant_vector_transfers %func
     : (!transform.op<"func.func">) -> !transform.op<"func.func">
   %all_loops = transform.structured.match interface{LoopLikeInterface} in %func_h
     : (!transform.op<"func.func">) -> !transform.any_op
   transform.apply_licm to %all_loops : !transform.any_op
   transform.loop.hoist_loop_invariant_subsets %all_loops : !transform.any_op

   // Simplify the 6-dim vector.contract into a 3-dim matmul-like
   // vector.contract with the following split into parallel and reduction
   // dims:
   //    * parallel, parallel, reduction
   transform.apply_patterns to %func_h {
     transform.apply_patterns.vector.reduction_to_contract
     transform.apply_patterns.vector.cast_away_vector_leading_one_dim
     transform.apply_patterns.vector.lower_masks
     transform.apply_patterns.canonicalization
   } : !transform.op<"func.func">
    transform.yield
  }

  transform.named_sequence @match_mmt4d(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.mmt4d"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }
}

func.func private @printMemrefI32(%ptr : tensor<*xi32>)
