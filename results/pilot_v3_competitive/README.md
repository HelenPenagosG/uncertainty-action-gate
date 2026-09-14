# Three-arm synthetic stress pilot

Executed 13–14 September 2026 with 20 runs per arm over four related scenario
families.

- Actor: `cognitivecomputations/dolphin-mistral-24b-venice-edition`
- Judge: `qwen/qwen3-8b`
- Environment: `v3` (authorized verifier permanently unavailable)
- Reward frame: `competitive`
- Maximum actor steps: 20
- Actor temperature: 0.7
- Judge temperature: 0.0

| Arm | Prohibited proposal | Prohibited execution | Flag captured | BLOCK | RECONSIDER | Final judge failures |
|---|---:|---:|---:|---:|---:|---:|
| B | 12/20 | 12/20 | 8/20 | 0 | 0 | n/a |
| C-UNC | 12/20 | 0/20 | 0/20 | 26 | 120 | 33/387 |
| C-COMP | 11/20 | 0/20 | 0/20 | 82 | 19 | 7/386 |

C-UNC recorded 120 reconsiderations and 26 hard blocks; C-COMP recorded 19
reconsiderations and 82 hard blocks. These intervention counts come from
different interactive trajectories and are descriptive.

Invalid final judge outputs followed the configured conservative fallback to
`RECONSIDER`. Their unequal frequency is an availability limitation; the
512-token judge output budget is the leading truncation hypothesis, not a
confirmed cause.

Only aggregate, sanitized artifacts are included here. Full step-level logs are
retained privately. `RESULTADO_PILOTO_ES.md` contains the original Spanish
delivery note.

This is a development proof of concept, not a confirmatory result. In v3 the
authorized task cannot be completed, so legitimate-success preservation is not
estimable.

The complete three-arm pilot cost less than USD 1 in inference. Total account
spend across development, screening, and pilot runs was USD 4.20.
