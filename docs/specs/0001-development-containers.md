# Development Containers

> **Status:** Draft **Author:** Martin Hess **Created:** 2026-01-12 **Updated:** N/A

## Summary

Speedup new developers by eliminating friction and failures when attempting to create a build and runtime environment
for metta, and to more closely replicate the linux x86 deployment environment using development containers.

## Problem

Setting up the metta development environment is fragile and platform-dependent. Developers on ARM Macs face a different
runtime than the Linux x86 production environment, leading to "works on my machine" issues. The setup process has
multiple failure modes that can leave developers stuck with a broken environment.

- **Fragile setup process**: A misstep during configuration can leave developers with an inoperable environment that's
  difficult to diagnose and fix
- **Platform mismatch**: ARM Mac development differs from Linux x86 production, causing behavior discrepancies that only
  surface in CI/CD or production
- **Dependency conflicts**: Native dependencies (nim, bazel, system libraries) can conflict with existing system
  installations or require specific versions
- **Onboarding friction**: New developers spend significant time troubleshooting environment setup instead of
  contributing code
- **Inconsistent environments**: Each developer's machine accumulates unique state over time, making bugs harder to
  reproduce

### Goals

#### 1. At a top level:

- [ ] **One-click onboarding**: New developers can start contributing within minutes using a pre-configured container
- [ ] **Reproducible debugging**: "Works in the devcontainer" becomes a reliable baseline for reproducing issues
- [ ] **CI/CD parity**: Local development environment matches the build pipeline, catching issues before they hit CI
- [ ] **Remote dev containers**: enables remote container development when different hardware is necessary e.g. GPU

#### 2. All metta commands work:

- [ ] configure - Configure Metta settings
- [ ] install - Install or update components
- [ ] status - Show status of components
- [ ] run - Run component-specific commands
- [ ] clean - Clean build artifacts and temporary files
- [ ] tool - Run a tool from the tools/ directory
- [ ] shell - Start an IPython shell with Metta imports
- [ ] go - Navigate to a Softmax Home shortcut
- [ ] pr-feed - Show PRs that touch a specific path
- [ ] build-dockerfiles - Build all repository Dockerfiles
- [ ] report-env-details - Report environment details including UV project directory
- [ ] clip - Copy codebase to clipboard. Pass through any codeclip flags
- [ ] gridworks - Start the Gridworks web UI
- [ ] run-monitor - Monitor training runs
- [ ] ci - Run CI checks locally
- [ ] publish - Create and push a release tag for a package
- [ ] observatory - Observatory local development
- [ ] book - Interactive marimo notebook commands
- [ ] codebase - Codebase management tools
- [ ] pytest - Python test runner
- [ ] cpptest - MettaGrid C++ test runner
- [ ] nimtest - MettaGrid Nim test runner
- [ ] lint - Code formatters

#### 3. Preserve what works

- [ ] Existing dev setup tools aren't impacted.

## Non-Goals

What is explicitly out of scope:

- devcontainer for other non-metta projects
- keeping non container dev setup tools at parity when/if devcontainer evolves

## Design

### Phase 1: prototype

- make dev container work
- make metta command work, but not sub commands

### Phase 2: the rest of it

- make each metta sub command work starting with the most used

## Open Questions

1. Not everything specified is important for developers, so need to prioritize, and prune
2. Some of these are already working, need to test
3. Some can be made to work easily, and others are more challenging
4. Some will require multiple containers, and there is some question on best way to approach
5. We want to make devcontainers part of the CI/CD verification process so that we know that devcontainers stay
   functioning - unclear what the minimal test to verify so that we aren't greatly increasing the checks time

## References

- [devcontainers](https://containers.dev)
- [devpod](https://devpod.sh)
