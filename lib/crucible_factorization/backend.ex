defmodule CrucibleFactorization.Backend do
  @moduledoc "Backend labeling and synchronization helpers for Nx tensors."

  @doc "Returns a compact backend label for a tensor or backend option."
  @spec label(term()) :: String.t()
  def label(nil), do: "current"
  def label({module, opts}) when is_atom(module), do: "#{module_label(module)}#{inspect(opts)}"
  def label(module) when is_atom(module), do: module_label(module)

  def label(%Nx.Tensor{data: %backend_struct{}} = tensor) do
    inspected = inspect(tensor)

    cond do
      String.contains?(inspected, "EXLA.Backend<cuda") -> "EXLA.Backend<cuda:"
      String.contains?(inspected, "EXLA.Backend<host") -> "EXLA.Backend<host:"
      String.contains?(inspected, "EXLA.Backend<") -> "EXLA.Backend"
      true -> module_label(backend_struct)
    end
  end

  def label(other), do: inspect(other)

  @doc "Forces a tensor or tensor container to the binary backend."
  @spec force(term()) :: term()
  def force(%Nx.Tensor{} = tensor), do: Nx.backend_transfer(tensor, Nx.BinaryBackend)
  def force({a, b, c}), do: {force(a), force(b), force(c)}
  def force(list) when is_list(list), do: Enum.map(list, &force/1)
  def force(map) when is_map(map), do: Map.new(map, fn {key, value} -> {key, force(value)} end)
  def force(other), do: other

  defp module_label(module) do
    module
    |> Module.split()
    |> Enum.join(".")
  end
end
