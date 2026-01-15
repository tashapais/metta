# Policy Arguments

> **Status:** Draft **Author:** Rhys Hiltner **Created:** 2026-01-14

## Summary

Define how runtime arguments are passed to policies across URI, storage, and serving layers.

## Problem

Today we pass args to policies as Python `**kwargs`. This creates ambiguity between policy-specific args and general
ones like `device=cpu`. Additionally, when decomposing policy usage into multiple steps (upload, download, run as
server, use in evaluation), it's unclear how args should be represented in URIs (string-only values, repeated keys) and
how overrides work across layers.

## Solution

Standardize policy args as an explicit `dict[str, str]` with clear override semantics at each layer.

## Design

### Data Model

1. Args are a map of string keys to string values (`dict[str, str]`).
2. Each key can have only a single value.
3. Args are passed to the policy as an explicit `args: dict[str, str]` parameter, not `**kwargs` expanded.

### Override Semantics

When decomposing "upload a policy, grab a policy from storage, use it for this episode" into multiple steps, each step
may override keys in the arg map. The most recent step wins. Layers may include:

1. **Upload**: Upload the policy to storage.
2. **Download**: Download the policy and store it as a zip.
3. **Serve**: Run the policy as a process exposing an HTTP server for the Policy Protocol.
4. **Use in episode**: Game engine makes an HTTP call telling the server to prepare for a particular episode.

Each layer can upsert keys; the most recent value wins.

### URI Encoding

Within a single layer, if you repeat a key like `policy=metta://policy/demo?k=1&k=2`, the ordering is undefined. This
won't come up in practice; listing for completeness.

## Open Questions

- Naming: should the parameter be called `args`, `params`, or something else?
