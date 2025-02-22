// RUN: mlir-opt %s --transform-dialect-check-uses --split-input-file --verify-diagnostics

func.func @use_after_free_branching_control_flow() {
  // expected-note @below {{allocated here}}
  %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
  transform.test_transform_op_with_regions {
    "transform.test_branching_transform_op_terminator"() : () -> ()
  },
  {
  ^bb0:
    "transform.test_branching_transform_op_terminator"()[^bb1, ^bb2] : () -> ()
  ^bb1:
    // expected-note @below {{freed here}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb2:
    "transform.test_branching_transform_op_terminator"()[^bb3] : () -> ()
  ^bb3:
    // expected-warning @below {{operand #0 may be used after free}}
    transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg0: !transform.any_op):
    }
    "transform.test_branching_transform_op_terminator"() : () -> ()
  }
  return
}
