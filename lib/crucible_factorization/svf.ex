defmodule CrucibleFactorization.SVF do
  @moduledoc "Public namespace wrapper for `Crucible.Factorization.SVF`."

  defdelegate decompose(delta_tensor, opts \\ []), to: Crucible.Factorization.SVF
  defdelegate decompose!(delta_tensor, opts \\ []), to: Crucible.Factorization.SVF
  defdelegate reconstruct(base_tensor, svf, opts \\ []), to: Crucible.Factorization.SVF
  defdelegate reconstruct!(base_tensor, svf, opts \\ []), to: Crucible.Factorization.SVF
end
