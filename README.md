# CrucibleFactorization

Nx SVD/SVF factorization primitives for model surgery and TRINITY artifact
export.

This package intentionally owns the temporary Nx/EXLA git pin required for the
thin-SVD memory behavior used by the coordinator. Contract packages should not
inherit that pin directly.

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `crucible_factorization` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:crucible_factorization, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at <https://hexdocs.pm/crucible_factorization>.

## CI

```sh
mix ci
```

CUDA is opt-in:

```sh
XLA_TARGET=cuda12 mix test --only cuda
```
