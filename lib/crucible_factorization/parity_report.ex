defmodule CrucibleFactorization.ParityReport do
  @moduledoc "Public namespace wrapper for `Crucible.Factorization.ParityReport`."

  defdelegate compare_tensors(computed, reference, opts \\ []),
    to: Crucible.Factorization.ParityReport

  defdelegate tensor_set_summary(tensors), to: Crucible.Factorization.ParityReport
end
