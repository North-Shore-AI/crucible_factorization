defmodule Crucible.Factorization.SVD do
  @moduledoc """
  Nx implementation of the SVD/SVF tensor mechanics used by artifact export.
  """

  alias CrucibleFactorization.{Backend, Errors, StageTiming}

  @router_vector_key "trinity_router_es_vector"

  @type decomposition :: %{
          required(:u) => Nx.Tensor.t(),
          required(:s) => Nx.Tensor.t(),
          required(:v) => Nx.Tensor.t(),
          required(:shape) => tuple()
        }

  @type tensor_entry :: %{
          required(:path) => String.t(),
          required(:segments) => [term()] | nil,
          required(:tensor) => Nx.Tensor.t()
        }

  @doc "Runs a thin SVD and returns timing/backend metadata."
  @spec thin(Nx.Tensor.t(), keyword()) :: {:ok, map()} | {:error, Exception.t()}
  def thin(%Nx.Tensor{} = tensor, opts \\ []) do
    {:ok, thin!(tensor, opts)}
  rescue
    exception -> {:error, exception}
  end

  @doc "Runs a thin SVD and raises on invalid input."
  @spec thin!(Nx.Tensor.t(), keyword()) :: map()
  def thin!(%Nx.Tensor{} = tensor, opts \\ []) do
    opts =
      Keyword.validate!(opts,
        rank: nil,
        compute_type: :source,
        backend: nil,
        force_sync?: false,
        sync_fun: &Backend.force/1
      )

    tensor = maybe_transfer_backend(tensor, opts[:backend])
    rank = normalize_rank!(opts[:rank], Nx.shape(tensor))

    {{u, s, v}, timing} =
      StageTiming.measure(
        fn ->
          tensor
          |> svd_source_tensor!(opts[:compute_type])
          |> svd_tuple!(full_matrices?: false)
          |> slice_rank!(rank)
        end,
        force_sync?: opts[:force_sync?],
        sync_fun: opts[:sync_fun]
      )

    %{
      u: u,
      s: s,
      v: v,
      rank: rank,
      compute_type: opts[:compute_type],
      source_type: Nx.type(tensor),
      shape: Nx.shape(tensor),
      backend_label: opts[:backend] |> backend_label_from_option(u),
      decompose_timing_ms: timing.elapsed_ms,
      reconstruct_timing_ms: nil,
      force_sync_timing_ms: timing.force_sync_timing_ms
    }
  end

  @doc "Returns true when a tensor matches the matrix-selection rule."
  def decomposable_tensor?(%Nx.Tensor{} = tensor) do
    shape = Nx.shape(tensor)
    tuple_size(shape) > 1 and Enum.all?(Tuple.to_list(shape), &(&1 > 1))
  end

  def decomposable_tensor?(_), do: false

  @doc "Runs a reduced SVD over one tensor."
  def decompose_tensor(%Nx.Tensor{} = tensor, opts \\ []) do
    opts = Keyword.validate!(opts, full_matrices?: false, compute_type: :source)
    svd_source = svd_source_tensor!(tensor, opts[:compute_type])
    {u, s, v} = svd_tuple!(svd_source, opts)
    %{u: u, s: s, v: v, shape: Nx.shape(svd_source), source_type: Nx.type(tensor)}
  end

  @doc "Reconstructs a tensor from SVD components and scale offsets."
  def reconstruct(decomposition, scale_offsets, opts \\ [])

  def reconstruct(%{u: u, s: s, v: v}, scale_offsets, opts) do
    opts = Keyword.validate!(opts, v_layout: :nx, output_type: nil)
    scale_offsets = Nx.as_type(scale_offsets, Nx.type(s))
    scaled_s = Nx.multiply(s, Nx.add(scale_offsets, 1))
    normalization = Nx.divide(Nx.sum(s), Nx.sum(scaled_s))
    v_for_dot = v_for_reconstruction!(v, opts[:v_layout])

    reconstructed =
      u
      |> Nx.multiply(Nx.reshape(scaled_s, {1, Nx.axis_size(scaled_s, 0)}))
      |> Nx.dot(v_for_dot)
      |> Nx.multiply(normalization)

    case opts[:output_type] do
      nil -> reconstructed
      target_type -> Nx.as_type(reconstructed, target_type)
    end
  end

  @doc "Flattens nested tensor containers into stable string paths."
  def flatten_tensors(container) do
    container
    |> flatten_tensor_entries()
    |> Enum.map(fn %{path: path, tensor: tensor} -> {path, tensor} end)
  end

  @doc "Flattens nested tensor containers with stable paths and original segments."
  def flatten_tensor_entries(container) do
    container
    |> do_flatten([])
    |> Enum.sort_by(fn %{path: path} -> tensor_order_key(path) end)
  end

  @doc "Returns decomposable tensors from a nested params container."
  def decomposable_tensors(container, opts \\ []) do
    container
    |> decomposable_tensor_entries(opts)
    |> Enum.map(fn %{path: path, tensor: tensor} -> {path, tensor} end)
  end

  @doc "Returns decomposable tensor entries from a nested params container."
  def decomposable_tensor_entries(container, opts \\ []) do
    opts = Keyword.validate!(opts, path_filter: nil)
    filter = opts[:path_filter]

    container
    |> flatten_tensor_entries()
    |> Enum.filter(fn %{path: path, tensor: tensor} ->
      decomposable_tensor?(tensor) and path_matches?(path, filter)
    end)
  end

  @doc "Returns the path filter equivalent to selected layer indices."
  def layer_index_filter(nil), do: nil

  def layer_index_filter(indices) when is_list(indices) do
    indices = Enum.map(indices, &to_string/1)

    fn path ->
      not String.contains?(path, "decoder.blocks.") or
        Enum.any?(indices, &String.contains?(path, "decoder.blocks.#{&1}."))
    end
  end

  @doc "Counts singular values consumed by selected tensors."
  def singular_value_count(tensors) when is_list(tensors) do
    Enum.reduce(tensors, 0, fn item, acc ->
      tensor = tensor_from_item!(item)
      shape = tensor |> Nx.shape() |> Tuple.to_list()
      acc + Enum.min(shape)
    end)
  end

  @doc "Produces a deterministic manifest for selected tensors."
  def tensor_manifest(tensors) when is_list(tensors) do
    Enum.map(tensors, fn item ->
      {path, tensor} = path_and_tensor_from_item!(item)

      %{
        path: path,
        shape: Nx.shape(tensor),
        singular_values: tensor |> Nx.shape() |> Tuple.to_list() |> Enum.min(),
        type: Nx.type(tensor)
      }
    end)
  end

  @doc "Decomposes selected tensor entries and records scale-vector spans."
  def decompose_tensors(tensors, opts \\ []) when is_list(tensors) do
    opts = Keyword.validate!(opts, progress: nil, svd_compute_type: :source)
    progress = opts[:progress]
    total = length(tensors)

    tensors
    |> Enum.with_index(1)
    |> Enum.map(fn {item, index} ->
      {path, segments, tensor} = path_segments_and_tensor_from_item!(item)
      count = tensor |> Nx.shape() |> Tuple.to_list() |> Enum.min()

      emit_progress(progress, %{
        event: :decompose_started,
        index: index,
        total: total,
        path: path,
        shape: Nx.shape(tensor),
        type: Nx.type(tensor),
        singular_values: count
      })

      {elapsed_us, decomposition} =
        :timer.tc(fn -> decompose_tensor(tensor, compute_type: opts[:svd_compute_type]) end)

      emit_progress(progress, %{
        event: :decompose_finished,
        index: index,
        total: total,
        path: path,
        u_backend: Backend.label(decomposition.u),
        s_backend: Backend.label(decomposition.s),
        v_backend: Backend.label(decomposition.v),
        elapsed_ms: div(elapsed_us, 1000)
      })

      %{
        path: path,
        segments: segments,
        tensor: tensor,
        decomposition: decomposition,
        singular_values: count
      }
    end)
  end

  @doc "Reconstructs selected decompositions by consuming scale offsets in order."
  def reconstruct_tensors(decompositions, scale_offsets, opts \\ [])
      when is_list(decompositions) do
    opts = Keyword.validate!(opts, progress: nil)
    progress = opts[:progress]
    total = length(decompositions)

    {reconstructed, offset} =
      decompositions
      |> Enum.with_index(1)
      |> Enum.map_reduce(0, fn {item, index}, offset ->
        count = item.singular_values
        next_offset = offset + count

        emit_progress(progress, %{
          event: :reconstruct_started,
          index: index,
          total: total,
          path: item.path,
          offset_start: offset,
          offset_end: next_offset,
          singular_values: count
        })

        offsets =
          scale_offsets
          |> Nx.slice([offset], [count])
          |> Nx.as_type(Nx.type(item.decomposition.s))

        {elapsed_us, tensor} =
          :timer.tc(fn ->
            item.decomposition
            |> reconstruct(offsets)
            |> Nx.as_type(Nx.type(item.tensor))
          end)

        emit_progress(progress, %{
          event: :reconstruct_finished,
          index: index,
          total: total,
          path: item.path,
          tensor_backend: Backend.label(tensor),
          elapsed_ms: div(elapsed_us, 1000)
        })

        {%{path: item.path, segments: Map.get(item, :segments), tensor: tensor}, next_offset}
      end)

    if offset != Nx.size(scale_offsets) do
      raise ArgumentError,
            "scale offset size mismatch: consumed #{offset}, got #{Nx.size(scale_offsets)}"
    end

    %{tensors: reconstructed, offset_count: offset}
  end

  @doc "Decomposes and reconstructs selected tensors with scale offsets."
  def adapt_tensors(tensors, scale_offsets, opts \\ []) when is_list(tensors) do
    opts = Keyword.validate!(opts, progress: nil, svd_compute_type: :source)

    tensors
    |> decompose_tensors(progress: opts[:progress], svd_compute_type: opts[:svd_compute_type])
    |> reconstruct_tensors(scale_offsets, progress: opts[:progress])
  end

  @doc "Puts adapted tensor entries back into a nested params container."
  def put_tensor_entries(%{data: data} = state, tensor_entries)
      when is_list(tensor_entries) and is_map(data) do
    %{state | data: put_tensor_entries(data, tensor_entries)}
  end

  def put_tensor_entries(container, tensor_entries) when is_list(tensor_entries) do
    Enum.reduce(tensor_entries, container, fn entry, acc ->
      segments = Map.get(entry, :segments)

      if not is_list(segments) or segments == [] do
        raise ArgumentError,
              "adapted tensor entry #{inspect(entry.path)} does not include path segments"
      end

      put_tensor_at_segments(acc, segments, entry.tensor)
    end)
  end

  @doc "Loads the raw router vector from safetensors."
  def load_router_vector!(path, tensor_name \\ @router_vector_key) do
    path
    |> Safetensors.read!()
    |> Map.fetch!(tensor_name)
  end

  @doc "Splits a raw vector into SVF scale offsets and linear head weights."
  def split_router_vector(vector, scale_count, hidden_size, output_count)
      when is_integer(scale_count) and scale_count >= 0 and
             is_integer(hidden_size) and hidden_size > 0 and
             is_integer(output_count) and output_count > 0 do
    size = Nx.size(vector)
    head_count = hidden_size * output_count
    expected_size = scale_count + head_count

    if size != expected_size do
      raise ArgumentError,
            "router vector size mismatch: expected #{expected_size}, got #{size}"
    end

    scale_offsets = Nx.slice(vector, [0], [scale_count])

    head_weights =
      vector
      |> Nx.slice([scale_count], [head_count])
      |> Nx.reshape({output_count, hidden_size})

    %{
      scale_offsets: scale_offsets,
      head_weights: head_weights,
      scale_count: scale_count,
      head_count: head_count
    }
  end

  @doc "Loads linear head weights into a dense-head-like params state."
  def put_linear_head_weights(%{data: data} = params, head_weights, layer_name \\ "routing_head") do
    {output_count, hidden_size} = Nx.shape(head_weights)

    kernel =
      head_weights
      |> Nx.transpose()
      |> Nx.as_type(:f32)

    bias = Nx.broadcast(0.0, {output_count})
    expected_kernel_shape = {hidden_size, output_count}

    case get_in(data, [layer_name, "kernel"]) do
      %Nx.Tensor{} = existing ->
        if Nx.shape(existing) == expected_kernel_shape do
          %{params | data: Map.put(data, layer_name, %{"kernel" => kernel, "bias" => bias})}
        else
          raise ArgumentError,
                "head kernel shape mismatch: expected #{inspect(expected_kernel_shape)}, got #{inspect(Nx.shape(existing))}"
        end

      _ ->
        raise ArgumentError, "missing dense layer #{inspect(layer_name)} kernel"
    end
  end

  defp maybe_transfer_backend(tensor, nil), do: tensor
  defp maybe_transfer_backend(tensor, backend), do: Nx.backend_transfer(tensor, backend)

  defp backend_label_from_option(nil, tensor), do: Backend.label(tensor)
  defp backend_label_from_option(backend, _tensor), do: Backend.label(backend)

  defp normalize_rank!(nil, shape), do: min_rank!(shape)

  defp normalize_rank!(rank, shape) when is_integer(rank) and rank > 0 do
    max_rank = min_rank!(shape)

    if rank <= max_rank do
      rank
    else
      raise Errors, "rank #{rank} exceeds maximum rank #{max_rank} for shape #{inspect(shape)}"
    end
  end

  defp normalize_rank!(rank, _shape),
    do: raise(Errors, "rank must be a positive integer, got #{inspect(rank)}")

  defp min_rank!(shape) when is_tuple(shape) and tuple_size(shape) == 2 do
    shape |> Tuple.to_list() |> Enum.min()
  end

  defp min_rank!(shape),
    do: raise(Errors, "expected rank-2 tensor for SVD, got shape #{inspect(shape)}")

  defp slice_rank!({u, s, v}, rank) do
    {rows, _} = Nx.shape(u)
    {_, cols} = Nx.shape(v)

    {Nx.slice(u, [0, 0], [rows, rank]), Nx.slice(s, [0], [rank]),
     Nx.slice(v, [0, 0], [rank, cols])}
  end

  defp do_flatten(%Nx.Tensor{} = tensor, path) do
    [%{path: format_path(path), segments: path, tensor: tensor}]
  end

  defp do_flatten(%{data: data}, path) when is_map(data), do: do_flatten(data, path)

  defp do_flatten(map, path) when is_map(map) do
    map
    |> Enum.flat_map(fn {key, value} -> do_flatten(value, path ++ [key]) end)
  end

  defp do_flatten(list, path) when is_list(list) do
    list
    |> Enum.with_index()
    |> Enum.flat_map(fn {value, index} -> do_flatten(value, path ++ [index]) end)
  end

  defp do_flatten(tuple, path) when is_tuple(tuple) do
    tuple
    |> Tuple.to_list()
    |> Enum.with_index()
    |> Enum.flat_map(fn {value, index} -> do_flatten(value, path ++ [index]) end)
  end

  defp do_flatten(_other, _path), do: []

  defp tensor_order_key(path) when is_binary(path) do
    cond do
      String.starts_with?(path, "embedder.") ->
        {0, 0, 0, path}

      String.starts_with?(path, "decoder.blocks.") ->
        decoder_block_order(path)

      String.starts_with?(path, "language_modeling_head.") ->
        {2, 0, 0, path}

      true ->
        {3, 0, 0, path}
    end
  end

  defp decoder_block_order(path) do
    block_index = decoder_block_index(path)

    cond do
      String.contains?(path, ".self_attention.query.") -> {1, block_index, 0, path}
      String.contains?(path, ".self_attention.key.") -> {1, block_index, 1, path}
      String.contains?(path, ".self_attention.value.") -> {1, block_index, 2, path}
      String.contains?(path, ".self_attention.output.") -> {1, block_index, 3, path}
      String.contains?(path, ".ffn.gate.") -> {1, block_index, 4, path}
      String.contains?(path, ".ffn.intermediate.") -> {1, block_index, 5, path}
      String.contains?(path, ".ffn.output.") -> {1, block_index, 6, path}
      true -> {1, block_index, 9, path}
    end
  end

  defp decoder_block_index(path) do
    case :binary.split(path, "decoder.blocks.") do
      [_prefix, rest] ->
        rest
        |> :binary.split(".")
        |> List.first()
        |> parse_block_index()

      _ ->
        0
    end
  end

  defp parse_block_index(value) do
    case Integer.parse(value) do
      {index, ""} -> index
      _ -> 0
    end
  end

  defp format_path(path) do
    path
    |> Enum.map_join(".", fn
      key when is_atom(key) -> Atom.to_string(key)
      key -> to_string(key)
    end)
  end

  defp path_matches?(_path, nil), do: true
  defp path_matches?(path, filter) when is_function(filter, 1), do: filter.(path)
  defp path_matches?(path, pattern) when is_binary(pattern), do: String.contains?(path, pattern)

  defp emit_progress(nil, _event), do: :ok
  defp emit_progress(fun, event) when is_function(fun, 1), do: fun.(event)

  defp svd_source_tensor!(tensor, :source), do: tensor
  defp svd_source_tensor!(tensor, :f32), do: Nx.as_type(tensor, :f32)

  defp svd_source_tensor!(_tensor, compute_type) do
    raise ArgumentError, "unsupported SVD compute_type #{inspect(compute_type)}"
  end

  defp v_for_reconstruction!(%Nx.Tensor{} = v, layout) when layout in [:nx, :vh], do: v
  defp v_for_reconstruction!(%Nx.Tensor{} = v, :torch_v), do: Nx.transpose(v)

  defp v_for_reconstruction!(_v, other) do
    raise ArgumentError,
          "unsupported SVD v_layout #{inspect(other)}; expected :nx, :vh, or :torch_v"
  end

  defp svd_tuple!(tensor, opts) do
    svd = Function.capture(Nx.LinAlg, :svd, 2)
    result = svd.(tensor, full_matrices?: opts[:full_matrices?])

    case result do
      {%Nx.Tensor{} = u, %Nx.Tensor{} = s, %Nx.Tensor{} = v} ->
        {u, s, v}

      other ->
        raise ArgumentError,
              "expected Nx.LinAlg.svd/2 to return {u, s, v}, got: #{inspect(other)}"
    end
  end

  defp tensor_from_item!({_path, %Nx.Tensor{} = tensor}), do: tensor
  defp tensor_from_item!(%{tensor: %Nx.Tensor{} = tensor}), do: tensor

  defp path_and_tensor_from_item!({path, %Nx.Tensor{} = tensor}), do: {path, tensor}

  defp path_and_tensor_from_item!(%{path: path, tensor: %Nx.Tensor{} = tensor}),
    do: {path, tensor}

  defp path_segments_and_tensor_from_item!({path, %Nx.Tensor{} = tensor}), do: {path, nil, tensor}

  defp path_segments_and_tensor_from_item!(%{
         path: path,
         segments: segments,
         tensor: %Nx.Tensor{} = tensor
       }),
       do: {path, segments, tensor}

  defp put_tensor_at_segments(_container, [], %Nx.Tensor{} = tensor), do: tensor

  defp put_tensor_at_segments(container, [segment | rest], %Nx.Tensor{} = tensor)
       when is_map(container) do
    if Map.has_key?(container, segment) do
      Map.update!(container, segment, &put_tensor_at_segments(&1, rest, tensor))
    else
      raise ArgumentError, "cannot put tensor at missing map segment #{inspect(segment)}"
    end
  end

  defp put_tensor_at_segments(container, [segment | rest], %Nx.Tensor{} = tensor)
       when is_list(container) and is_integer(segment) do
    if segment >= 0 and segment < length(container) do
      List.update_at(container, segment, &put_tensor_at_segments(&1, rest, tensor))
    else
      raise ArgumentError, "cannot put tensor at missing list index #{segment}"
    end
  end

  defp put_tensor_at_segments(container, [segment | rest], %Nx.Tensor{} = tensor)
       when is_tuple(container) and is_integer(segment) do
    if segment >= 0 and segment < tuple_size(container) do
      put_elem(container, segment, put_tensor_at_segments(elem(container, segment), rest, tensor))
    else
      raise ArgumentError, "cannot put tensor at missing tuple index #{segment}"
    end
  end

  defp put_tensor_at_segments(_container, [segment | _rest], %Nx.Tensor{}) do
    raise ArgumentError, "cannot descend into path segment #{inspect(segment)}"
  end
end
