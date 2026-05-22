defmodule Crucible.Factorization.SVF do
  @moduledoc "Low-rank singular-vector-field helpers built on thin SVD."

  alias Crucible.Factorization.SVD
  alias CrucibleFactorization.StageTiming

  @doc "Decomposes a delta tensor into low-rank SVD components."
  @spec decompose(Nx.Tensor.t(), keyword()) :: {:ok, map()} | {:error, Exception.t()}
  def decompose(%Nx.Tensor{} = delta_tensor, opts \\ []) do
    {:ok, decompose!(delta_tensor, opts)}
  rescue
    exception -> {:error, exception}
  end

  @doc "Decomposes a delta tensor, raising on invalid input."
  @spec decompose!(Nx.Tensor.t(), keyword()) :: map()
  def decompose!(%Nx.Tensor{} = delta_tensor, opts \\ []) do
    result = SVD.thin!(delta_tensor, Keyword.put_new(opts, :compute_type, :f32))

    %{
      u: result.u,
      s: result.s,
      v: result.v,
      rank: result.rank,
      source_type: Nx.type(delta_tensor),
      backend_label: result.backend_label,
      decompose_timing_ms: result.decompose_timing_ms,
      force_sync_timing_ms: result.force_sync_timing_ms
    }
  end

  @doc "Reconstructs `base_tensor + low_rank_delta` from SVF components."
  @spec reconstruct(Nx.Tensor.t(), map(), keyword()) ::
          {:ok, Nx.Tensor.t()} | {:error, Exception.t()}
  def reconstruct(%Nx.Tensor{} = base_tensor, svf, opts \\ []) when is_map(svf) do
    {:ok, reconstruct!(base_tensor, svf, opts)}
  rescue
    exception -> {:error, exception}
  end

  @doc "Reconstructs `base_tensor + low_rank_delta`, raising on invalid input."
  @spec reconstruct!(Nx.Tensor.t(), map(), keyword()) :: Nx.Tensor.t()
  def reconstruct!(%Nx.Tensor{} = base_tensor, svf, opts \\ []) when is_map(svf) do
    opts = Keyword.validate!(opts, force_sync?: false, sync_fun: nil)
    zeros = Nx.broadcast(0.0, {Nx.axis_size(svf.s, 0)})

    {result, _timing} =
      StageTiming.measure(
        fn ->
          delta =
            svf
            |> SVD.reconstruct(zeros, output_type: Nx.type(base_tensor))
            |> Nx.as_type(Nx.type(base_tensor))

          Nx.add(base_tensor, delta)
        end,
        force_sync?: opts[:force_sync?],
        sync_fun: opts[:sync_fun] || (&CrucibleFactorization.Backend.force/1)
      )

    result
  end
end
