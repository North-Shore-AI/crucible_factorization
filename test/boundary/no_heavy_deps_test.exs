defmodule CrucibleFactorization.Boundary.NoHeavyDepsTest do
  use ExUnit.Case, async: true

  @forbidden ~w(
    Bumblebee.
    Axon.
    Req.
    HfHub.
    Inference.
    AITrace.
    SelfHostedInferenceCore.
    Trinity.
  )

  test "factorization package does not import runtime or provider dependencies" do
    assert forbidden_hits() == []
  end

  defp forbidden_hits do
    "lib/**/*.{ex,exs}"
    |> Path.wildcard()
    |> Enum.flat_map(&file_hits/1)
  end

  defp file_hits(path) do
    body = File.read!(path)

    Enum.flat_map(@forbidden, fn token ->
      if String.contains?(body, token), do: [{path, token}], else: []
    end)
  end
end
