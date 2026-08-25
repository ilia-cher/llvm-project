
module attributes {transform.with_named_sequence} {

  transform.named_sequence @mbb_has_wmma(
      %mbb: !transform.mir.mbb {transform.readonly}) -> !transform.mir.mbb {
    %ops = transform.mir.mbb_ops %mbb
        : (!transform.mir.mbb) -> !transform.any_op
    %wmmas = transform.mir.match_asm %ops contains "WMMA"
        : (!transform.any_op) -> !transform.any_op
    %m = transform.mir.parent_mbb %wmmas
        : (!transform.any_op) -> !transform.mir.mbb
    transform.yield %m : !transform.mir.mbb
  }

  transform.named_sequence @__transform_main(
      %func: !transform.op<"mir.func"> {transform.readonly}) {

    %all = transform.mir.get_mbbs %func
        : (!transform.op<"mir.func">) -> !transform.mir.mbb

    %wmma_mbbs = transform.mir.collect_mbbs @mbb_has_wmma in %all
        : (!transform.mir.mbb) -> !transform.mir.mbb

    transform.foreach %wmma_mbbs : !transform.mir.mbb {
    ^bb0(%wmma_mbb: !transform.mir.mbb):

      %ops = transform.mir.mbb_ops %wmma_mbb
          : (!transform.mir.mbb) -> !transform.any_op
      %wmmas = transform.mir.match_asm %ops contains "WMMA"
          : (!transform.any_op) -> !transform.any_op
      %pred, %rem = transform.mir.split %wmmas
          : (!transform.any_op) -> (!transform.mir.mbb, !transform.mir.mbb)
    }

    transform.yield
  }
}
