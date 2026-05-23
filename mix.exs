defmodule CrucibleFactorization.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/North-Shore-AI/crucible_factorization"

  def project do
    [
      app: :crucible_factorization,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),
      dialyzer: [plt_add_deps: :apps_direct],
      name: "CrucibleFactorization",
      description: "Nx SVD/SVF factorization primitives for model surgery",
      source_url: @source_url,
      homepage_url: @source_url,
      docs: docs(),
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  def cli do
    [
      preferred_envs: [
        ci: :test,
        credo: :test,
        dialyzer: :test,
        docs: :dev
      ]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.12", override: true},
      {:exla, "~> 0.12", override: true},
      {:safetensors, "~> 0.1.3"},
      {:jason, "~> 1.4"},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.40.1", only: [:dev, :test], runtime: false}
    ]
  end

  defp aliases do
    [
      ci: [
        "deps.get",
        "format --check-formatted",
        "compile --warnings-as-errors",
        "test",
        "credo --strict",
        "dialyzer --format short",
        "docs"
      ]
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "MIGRATION.md"],
      source_ref: "main",
      source_url: @source_url,
      homepage_url: @source_url
    ]
  end

  defp package do
    [
      name: "crucible_factorization",
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib mix.exs README.md LICENSE MIGRATION.md)
    ]
  end
end
