module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %func: !transform.op<"mir.func"> {transform.readonly}) {
    transform.mir.foreach_mbb %func : !transform.op<"mir.func"> {
    ^bb0(%mbb: !transform.mir.mbb):
      transform.mir.print_mbb %mbb : !transform.mir.mbb
    }
    transform.yield
  }
}
