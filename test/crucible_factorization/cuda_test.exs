defmodule CrucibleFactorization.CudaTest do
  use ExUnit.Case, async: false

  alias CrucibleFactorization.SVD

  @tag :cuda
  test "thin/2 runs on EXLA CUDA" do
    assert Code.ensure_loaded?(EXLA.Client)
    assert Map.get(EXLA.Client.get_supported_platforms(), :cuda, 0) > 0

    matrix = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)

    assert {:ok, result} =
             SVD.thin(matrix,
               rank: 2,
               backend: {EXLA.Backend, client: :cuda},
               force_sync?: true
             )

    assert result.rank == 2
    assert result.backend_label =~ "EXLA.Backend"
  end
end
