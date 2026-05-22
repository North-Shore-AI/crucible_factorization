defmodule Crucible.Factorization.StageCheck do
  @moduledoc "Shared tensor comparisons for factorization parity checks."

  alias CrucibleFactorization.{Backend, TensorHash}

  @doc "Compares computed stage tensors with reference stage tensors."
  def compare_stage_tensors(stage_tensors, reference_stage_tensors, opts \\ [])
  def compare_stage_tensors(_stage_tensors, nil, _opts), do: []

  def compare_stage_tensors(stage_tensors, reference_stage_tensors, opts)
      when is_map(stage_tensors) do
    opts =
      Keyword.validate!(opts,
        include_alt_hashes: true,
        include_tensor_summaries: true,
        compute_byte_match: true
      )

    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      stage_tensors
      |> Enum.sort_by(fn {key, _tensor} -> key end)
      |> Enum.map(&compare_one_stage(&1, reference_stage_tensors, opts))
    end)
  end

  @doc "Returns true when every required stage check passed."
  def checks_passed?([]), do: nil

  def checks_passed?(checks) when is_list(checks) do
    Enum.all?(checks, fn check ->
      not check["required_for_functional_parity"] or check["functional_passed"]
    end)
  end

  @doc "Returns a JSON-safe summary for a tensor."
  def tensor_summary(tensor, opts \\ []) do
    opts = Keyword.validate!(opts, prefix_count: 8, include_alt_hashes: true, backend_label: nil)

    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      tensor = host_snapshot(tensor)
      tensor_f32 = Nx.as_type(tensor, :f32) |> host_snapshot()
      size = Nx.size(tensor)
      prefix_count = min(size, opts[:prefix_count])

      base = %{
        "shape" => shape_list(tensor),
        "type" => inspect(Nx.type(tensor)),
        "backend" => opts[:backend_label] || Backend.label(tensor),
        "snapshot_backend" => Backend.label(tensor),
        "size" => size,
        "sha256" => TensorHash.tensor_sha256(tensor),
        "min" => scalar(Nx.reduce_min(tensor_f32)),
        "max" => scalar(Nx.reduce_max(tensor_f32)),
        "sum" => scalar(Nx.sum(tensor_f32)),
        "prefix_f32" => prefix_f32(tensor, prefix_count)
      }

      if opts[:include_alt_hashes] do
        Map.merge(base, %{
          "sha256_as_f32" => TensorHash.tensor_sha256(Nx.as_type(tensor, :f32)),
          "sha256_as_bf16" => TensorHash.tensor_sha256(Nx.as_type(tensor, :bf16))
        })
      else
        base
      end
    end)
  end

  defp compare_one_stage({key, tensor}, reference_stage_tensors, opts) do
    case Map.fetch(reference_stage_tensors, key) do
      {:ok, reference_tensor} -> stage_check(key, tensor, reference_tensor, opts)
      :error -> missing_stage_check(key)
    end
  end

  defp stage_check(key, computed_tensor, reference_tensor, opts) do
    computed_tensor = host_snapshot(computed_tensor)
    reference_tensor = host_snapshot(reference_tensor)
    tolerance = stage_tolerance(key)
    shape_match = Nx.shape(computed_tensor) == Nx.shape(reference_tensor)
    byte_match = maybe_byte_match(computed_tensor, reference_tensor, opts)

    if shape_match do
      stage_value_check(key, computed_tensor, reference_tensor, tolerance, byte_match, opts)
    else
      %{
        "stage" => key,
        "required_for_functional_parity" => tolerance.required?,
        "byte_match" => byte_match,
        "shape_match" => false,
        "computed" => maybe_tensor_summary(computed_tensor, opts),
        "reference" => maybe_tensor_summary(reference_tensor, opts),
        "max_abs_error" => nil,
        "mean_abs_error" => nil,
        "mismatched_element_count" => nil,
        "tolerance" => %{
          "max_abs_error" => tolerance.max_abs,
          "mean_abs_error" => tolerance.mean_abs
        },
        "functional_passed" => false
      }
    end
  end

  defp stage_value_check(key, computed_tensor, reference_tensor, tolerance, byte_match, opts) do
    computed_f32 = Nx.as_type(computed_tensor, :f32)
    reference_f32 = Nx.as_type(reference_tensor, :f32)
    abs_diff = Nx.abs(Nx.subtract(computed_f32, reference_f32))
    max_abs = scalar(Nx.reduce_max(abs_diff))
    mean_abs = scalar(Nx.divide(Nx.sum(abs_diff), Nx.tensor(Nx.size(abs_diff), type: :f32)))

    mismatch_count =
      scalar(Nx.sum(Nx.as_type(Nx.not_equal(computed_tensor, reference_tensor), :s64)))

    %{
      "stage" => key,
      "required_for_functional_parity" => tolerance.required?,
      "byte_match" => byte_match,
      "shape_match" => true,
      "computed" => maybe_tensor_summary(computed_tensor, opts),
      "reference" => maybe_tensor_summary(reference_tensor, opts),
      "max_abs_error" => max_abs,
      "mean_abs_error" => mean_abs,
      "mismatched_element_count" => mismatch_count,
      "tolerance" => %{
        "max_abs_error" => tolerance.max_abs,
        "mean_abs_error" => tolerance.mean_abs
      },
      "functional_passed" => max_abs <= tolerance.max_abs and mean_abs <= tolerance.mean_abs
    }
  end

  defp maybe_byte_match(computed_tensor, reference_tensor, opts) do
    if opts[:compute_byte_match] do
      TensorHash.tensor_sha256(computed_tensor) == TensorHash.tensor_sha256(reference_tensor)
    end
  end

  defp maybe_tensor_summary(tensor, opts) do
    if opts[:include_tensor_summaries] do
      tensor_summary(tensor,
        prefix_count: 8,
        include_alt_hashes: opts[:include_alt_hashes]
      )
    else
      %{
        "shape" => shape_list(tensor),
        "type" => inspect(Nx.type(tensor)),
        "backend" => Backend.label(tensor),
        "size" => Nx.size(tensor),
        "summary_omitted" => true
      }
    end
  end

  defp missing_stage_check(key) do
    tolerance = stage_tolerance(key)

    %{
      "stage" => key,
      "required_for_functional_parity" => tolerance.required?,
      "byte_match" => false,
      "shape_match" => false,
      "missing_reference_stage" => true,
      "functional_passed" => not tolerance.required?
    }
  end

  defp stage_tolerance("stage.source_f32"), do: %{required?: true, max_abs: 0.0, mean_abs: 0.0}
  defp stage_tolerance("stage.offsets_f32"), do: %{required?: true, max_abs: 0.0, mean_abs: 0.0}

  defp stage_tolerance("stage.scaled_s"),
    do: %{required?: true, max_abs: 1.0e-6, mean_abs: 1.0e-8}

  defp stage_tolerance("stage.normalization"),
    do: %{required?: true, max_abs: 1.0e-6, mean_abs: 1.0e-6}

  defp stage_tolerance("stage.u_scaled"),
    do: %{required?: true, max_abs: 1.0e-6, mean_abs: 1.0e-8}

  defp stage_tolerance("stage.zero_source_f32"),
    do: %{required?: true, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp stage_tolerance("stage.matmul_pre_norm"),
    do: %{required?: true, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp stage_tolerance("stage.adapted_source_f32"),
    do: %{required?: true, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp stage_tolerance("stage.final_f32"),
    do: %{required?: true, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp stage_tolerance("stage.final_bf16"),
    do: %{required?: false, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp stage_tolerance(_key), do: %{required?: false, max_abs: 1.0e-3, mean_abs: 1.0e-5}

  defp prefix_f32(_tensor, 0), do: []

  defp prefix_f32(tensor, count) do
    tensor
    |> host_snapshot()
    |> Nx.as_type(:f32)
    |> Nx.reshape({Nx.size(tensor)})
    |> Nx.slice([0], [count])
    |> host_snapshot()
    |> Nx.to_flat_list()
  end

  defp scalar(tensor), do: tensor |> host_snapshot() |> Nx.to_number() |> finite_float()
  defp finite_float(value) when is_float(value), do: value
  defp finite_float(value), do: value
  defp shape_list(tensor), do: tensor |> Nx.shape() |> Tuple.to_list()
  defp host_snapshot(%Nx.Tensor{} = tensor), do: Nx.backend_transfer(tensor, Nx.BinaryBackend)
end
