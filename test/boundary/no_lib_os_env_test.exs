defmodule CrucibleFactorization.Boundary.NoLibOsEnvTest do
  use ExUnit.Case, async: true

  @forbidden ~w(
    System.get_env
    System.fetch_env
    System.fetch_env!
    System.put_env
    System.delete_env
  )

  test "no lib/** source calls direct OS env APIs" do
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
