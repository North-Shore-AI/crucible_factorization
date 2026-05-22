defmodule CrucibleFactorization.TensorHash do
  @moduledoc "Deterministic tensor hashing helpers."

  @doc "Computes tensor SHA-256 in lowercase hex after host transfer."
  @spec tensor_sha256(Nx.Tensor.t()) :: String.t()
  def tensor_sha256(%Nx.Tensor{} = tensor) do
    binary =
      tensor
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_binary()

    :crypto.hash(:sha256, binary)
    |> Base.encode16(case: :lower)
  end
end
