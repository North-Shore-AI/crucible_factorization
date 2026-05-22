defmodule CrucibleFactorization.StageCheckTest do
  use ExUnit.Case, async: true

  alias CrucibleFactorization.StageCheck

  test "compares matching stage tensors" do
    tensor = Nx.tensor([1.0, 2.0], type: :f32)

    assert [check] =
             StageCheck.compare_stage_tensors(
               %{"stage.source_f32" => tensor},
               %{"stage.source_f32" => tensor},
               include_tensor_summaries: false
             )

    assert check["functional_passed"]
    assert check["byte_match"]
    assert StageCheck.checks_passed?([check])
  end
end
