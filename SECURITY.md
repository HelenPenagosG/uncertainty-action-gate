# Security

This repository is intended for controlled AI-safety research.

The benchmark under `benchmarks/synthetic_state` uses fictional resources and
an in-memory state machine; it never executes model-generated text as shell
commands. `benchmarks/docker_lab` can execute an accepted action, but only
inside its disposable actor container on an internal network with no published
ports. The more general runner also includes a sandbox client so research teams
can connect infrastructure they control.

Do not connect the runner to public targets, production services, personal
machines containing sensitive data, or any system without explicit written
authorization. Keep experimental sandboxes isolated, disposable, and denied
outbound internet access. Never commit credentials or raw held-out inputs.

For a suspected credential leak, revoke the credential first and then remove
it from repository history. For a security issue in this prototype, contact the
project maintainers privately rather than opening a public issue.
