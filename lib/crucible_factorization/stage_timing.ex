defmodule CrucibleFactorization.StageTiming do
  @moduledoc "Small timing helper that can include a backend synchronization point."

  alias CrucibleFactorization.Backend

  @type t :: %{
          required(:elapsed_ms) => non_neg_integer(),
          required(:force_sync_timing_ms) => non_neg_integer() | nil
        }

  @doc "Runs a function and optionally forces its result before returning."
  @spec measure((-> term()), keyword()) :: {term(), t()}
  def measure(fun, opts \\ []) when is_function(fun, 0) do
    opts = Keyword.validate!(opts, force_sync?: false, sync_fun: &Backend.force/1)
    {elapsed_us, result} = :timer.tc(fun)

    {result, sync_ms} =
      if opts[:force_sync?] do
        {sync_us, synced} = :timer.tc(fn -> opts[:sync_fun].(result) end)
        {synced, us_to_ms(sync_us)}
      else
        {result, nil}
      end

    {result, %{elapsed_ms: us_to_ms(elapsed_us), force_sync_timing_ms: sync_ms}}
  end

  defp us_to_ms(us), do: div(us, 1000)
end
