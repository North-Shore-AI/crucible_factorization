defmodule Crucible.Factorization.ParityReport do
  @moduledoc "Math-only parity summaries for factorization outputs."

  alias Crucible.Factorization.StageCheck

  @doc "Compares two tensors and returns numeric error metrics."
  @spec compare_tensors(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: map()
  def compare_tensors(%Nx.Tensor{} = computed, %Nx.Tensor{} = reference, opts \\ []) do
    opts = Keyword.validate!(opts, stage: "tensor")

    [check] =
      StageCheck.compare_stage_tensors(
        %{opts[:stage] => computed},
        %{opts[:stage] => reference},
        include_alt_hashes: false,
        include_tensor_summaries: false
      )

    check
  end

  @doc "Returns a compact summary of a named tensor set."
  @spec tensor_set_summary(%{String.t() => Nx.Tensor.t()}) :: [map()]
  def tensor_set_summary(tensors) when is_map(tensors) do
    tensors
    |> Enum.sort_by(fn {name, _tensor} -> name end)
    |> Enum.map(fn {name, tensor} ->
      Map.put(StageCheck.tensor_summary(tensor, include_alt_hashes: false), "name", name)
    end)
  end
end
