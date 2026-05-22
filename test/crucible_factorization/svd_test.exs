defmodule CrucibleFactorization.SVDTest do
  use ExUnit.Case, async: true

  alias CrucibleFactorization.{Errors, SVD}

  test "thin/2 reconstructs a rank-1 exact matrix" do
    matrix = Nx.tensor([[2.0, 4.0], [1.0, 2.0]], type: :f32)

    assert {:ok, result} = SVD.thin(matrix, rank: 1, compute_type: :f32, force_sync?: true)
    reconstructed = SVD.reconstruct(result, Nx.broadcast(0.0, {1}))

    assert_close(reconstructed, matrix, 1.0e-4)
    assert result.rank == 1
    assert result.compute_type == :f32
    assert is_integer(result.decompose_timing_ms)
    assert is_integer(result.force_sync_timing_ms)
    assert is_binary(result.backend_label)
  end

  test "thin/2 supports rank-k approximate reconstruction" do
    matrix = Nx.tensor([[3.0, 1.0], [1.0, 3.0], [0.5, 0.25]], type: :f32)

    assert {:ok, rank1} = SVD.thin(matrix, rank: 1, compute_type: :f32)
    assert {:ok, rank2} = SVD.thin(matrix, rank: 2, compute_type: :f32)

    rank1_error = reconstruction_error(rank1, matrix)
    rank2_error = reconstruction_error(rank2, matrix)

    assert rank2_error <= rank1_error
    assert rank2_error < 1.0e-4
  end

  test "thin/2 rejects bad rank" do
    assert {:error, %Errors{message: message}} =
             SVD.thin(Nx.tensor([[1.0, 2.0]], type: :f32), rank: 2)

    assert message =~ "exceeds maximum rank"
  end

  test "sync timing instrumentation calls custom sync once" do
    parent = self()
    matrix = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)

    assert {:ok, result} =
             SVD.thin(matrix,
               rank: 1,
               force_sync?: true,
               sync_fun: fn value ->
                 send(parent, :synced)
                 value
               end
             )

    assert_receive :synced
    assert is_integer(result.force_sync_timing_ms)
  end

  test "backend option reports a backend label" do
    matrix = Nx.tensor([[1.0, 0.0], [0.0, 1.0]], type: :f32)

    assert {:ok, result} = SVD.thin(matrix, backend: Nx.BinaryBackend)
    assert result.backend_label == "Nx.BinaryBackend"
  end

  defp reconstruction_error(result, matrix) do
    result
    |> SVD.reconstruct(Nx.broadcast(0.0, {result.rank}))
    |> Nx.subtract(matrix)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
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
