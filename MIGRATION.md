# Migration Notes

This repo was scaffolded during TRINITY decomposition Phase 1 so
`trinity_framework` could resolve local path dependencies before the public
GitHub repos existed.

Source material for the Phase 3 implementation:

- `nshkrdotcom/trinity_coordinator` tag `v0.1.0-monolith`
- source commit `64144a2983950e5fc9f2db2d26323a576c7379a1`
- `lib/trinity_coordinator/sakana/svd.ex`
- math portions of `lib/trinity_coordinator/sakana/stage_check.ex`
- math/reporting portions of `lib/trinity_coordinator/sakana/parity_trace.ex`
- sync-timing pattern from `lib/trinity_coordinator/sakana/exporter.ex`

The implementation keeps provider, orchestration, tracing, and product runtime
dependencies out of the factorization package. Compatibility functions that
previously accepted model-state structs now operate on generic maps or structs
with a `:data` field.
