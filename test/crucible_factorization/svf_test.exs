defmodule CrucibleFactorization.SVFTest do
  use ExUnit.Case, async: true

  alias CrucibleFactorization.SVF

  test "decompose and reconstruct round trip a rank-1 delta" do
    base = Nx.tensor([[1.0, 1.0], [1.0, 1.0]], type: :f32)
    delta = Nx.tensor([[2.0, 4.0], [1.0, 2.0]], type: :f32)

    assert {:ok, svf} = SVF.decompose(delta, rank: 1)
    assert {:ok, reconstructed} = SVF.reconstruct(base, svf)

    assert_close(reconstructed, Nx.add(base, delta), 1.0e-4)
  end

  defp assert_close(left, right, tolerance) do
    max_abs =
      left
      |> Nx.subtract(right)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert max_abs <= tolerance
  end
end
