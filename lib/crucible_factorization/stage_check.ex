defmodule CrucibleFactorization.StageCheck do
  @moduledoc "Public namespace wrapper for `Crucible.Factorization.StageCheck`."

  defdelegate compare_stage_tensors(stage_tensors, reference_stage_tensors, opts \\ []),
    to: Crucible.Factorization.StageCheck

  defdelegate checks_passed?(checks), to: Crucible.Factorization.StageCheck
  defdelegate tensor_summary(tensor, opts \\ []), to: Crucible.Factorization.StageCheck
end
