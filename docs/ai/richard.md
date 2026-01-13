# Richard Higgins - Codex usage analysis

This document summarizes local Codex CLI usage across Metta workspaces and provides a deep analysis of session patterns,
intent distribution, prompt structure, and assistant response characteristics. It is intended to capture how I (Richard
Higgins) actually use Codex in day-to-day work.

## Scope and data sources

- Data source: local Codex CLI session logs under `~/.codex/sessions/2025` and `~/.codex/sessions/2026`.
- Target workspaces (treated as one combined Metta workspace):
  - `/Users/relh/Code/workspace/metta`
  - `/Users/relh/Code/dummyspace/metta`
  - `/Users/relh/Code/metta`
  - `/Users/relh/Code/thirdspace/metta`
  - `/Users/relh/Code/fourthspace/metta`
- Sessions scanned: 997 session files (all found under those years).
- Sessions matched to these workspaces: 870
- Time span: 2025-11-05 to 2026-01-13 (UTC timestamps from logs).
- Filtering: removed system-injected items (AGENTS instructions, environment context blocks, and internal <user_action>
  payloads).
- Repeated prompts are retained; no de-duplication is applied.
- Character set: non-ASCII characters were stripped to keep this file ASCII-only.

## Executive summary

- Total prompts: 5975 across 870 sessions (2025: 5429, 2026: 546).
- Questions vs. imperatives: 3375 questions (56.5%) vs 2600 imperatives (43.5%).
- Top intent categories by volume: Implementation/refactor (1238, 20.7%), Build/run/install (1128, 18.9%),
  How-to/questions (1003, 16.8%).
- Multi-step directives (has 'and/then' + action verb): 1626 (27.2%).
- Prompts with file paths: 2227 (37.3%); with explicit commands: 1875 (31.4%).
- Assistant responses analyzed: 5593 (median 103 words; p90 245).

## Dataset overview

- Total user prompts (after filtering): 5975
- Prompt length: median 22 words (p25 12, p75 44, p90 109)
- Session length: median 1 prompts (mean 6.9, p75 6, p90 19, max 124)
- Short follow-ups (<= 6 words): 446 (7.5%)
- Very short follow-ups (<= 3 words): 88 (1.5%)
- Long prompts (> 200 words): 268 (4.5%)

## Session structure

Distribution of prompts per session:

| Prompts per session | Sessions |
| ------------------- | -------: |
| <= 1                |      450 |
| <= 2                |       67 |
| <= 3                |       42 |
| <= 5                |       72 |
| <= 8                |       62 |
| <= 13               |       52 |
| <= 21               |       48 |
| <= 34               |       35 |
| > 34                |       42 |

- Interpretation: the modal session is a single prompt, but there is a long tail of multi-step sessions (p90 = 19
  prompts).
- Review mode used in 114 sessions; context compaction in 35 sessions; aborted turns in 191 sessions.

## Intent taxonomy (fine-grained)

The categories below are assigned by keyword heuristics. A prompt is assigned to the first matching category in this
order:

- PR review/audit
- PR management/metadata
- Git/branch operations
- Implementation/refactor
- Build/run/install
- Tests/CI
- Lint/format
- Debugging/triage
- Docs/copy
- Analysis/research
- How-to/questions
- Ops/workflow
- Other

## Category distribution (combined)

| Category                | Count | Share |
| ----------------------- | ----: | ----: |
| PR review/audit         |   221 |  3.7% |
| PR management/metadata  |   388 |  6.5% |
| Git/branch operations   |   890 | 14.9% |
| Implementation/refactor |  1238 | 20.7% |
| Build/run/install       |  1128 | 18.9% |
| Tests/CI                |    75 |  1.3% |
| Lint/format             |    12 |  0.2% |
| Debugging/triage        |    62 |  1.0% |
| Docs/copy               |     8 |  0.1% |
| Analysis/research       |   106 |  1.8% |
| How-to/questions        |  1003 | 16.8% |
| Ops/workflow            |     8 |  0.1% |
| Other                   |   836 | 14.0% |

## Semantic groupings

These groupings roll up the fine-grained categories into broader intent clusters:

| Group                        | Included categories                                            | Count | Share |
| ---------------------------- | -------------------------------------------------------------- | ----: | ----: |
| Coordination and code review | PR review/audit, PR management/metadata, Git/branch operations |  1499 | 25.1% |
| Change-making                | Implementation/refactor, Docs/copy                             |  1246 | 20.9% |
| Execution and validation     | Build/run/install, Tests/CI, Lint/format                       |  1215 | 20.3% |
| Debugging and investigation  | Debugging/triage, Analysis/research                            |   168 |  2.8% |
| How-to and workflow          | How-to/questions, Ops/workflow                                 |  1011 | 16.9% |
| Other/uncategorized          | Other                                                          |   836 | 14.0% |

## Dominant session archetypes

Dominant category per session (based on most frequent prompt category within the session):

| Dominant category       | Sessions | Share |
| ----------------------- | -------: | ----: |
| PR review/audit         |      196 | 22.5% |
| Implementation/refactor |      148 | 17.0% |
| Git/branch operations   |      106 | 12.2% |
| Build/run/install       |      105 | 12.1% |
| How-to/questions        |       83 |  9.5% |
| PR management/metadata  |       48 |  5.5% |
| Other                   |       40 |  4.6% |
| Analysis/research       |        4 |  0.5% |
| Debugging/triage        |        3 |  0.3% |
| Ops/workflow            |        2 |  0.2% |
| Tests/CI                |        1 |  0.1% |

## Workflow flows (category sequences)

Common category sequences within multi-prompt sessions (consecutive duplicates collapsed).

Top 2-step flows:

- impl_change -> howto_question (127), git_branch -> impl_change (121), build_run -> howto_question (119), build_run ->
  impl_change (116), howto_question -> build_run (112), howto_question -> impl_change (112), impl_change -> build_run
  (109), impl_change -> other (106), impl_change -> git_branch (102), howto_question -> other (92), other ->
  howto_question (90), git_branch -> build_run (88)

Top 3-step flows:

- impl_change -> howto_question -> impl_change (49), howto_question -> build_run -> howto_question (43), build_run ->
  howto_question -> build_run (43), impl_change -> build_run -> impl_change (42), build_run -> impl_change -> build_run
  (41), git_branch -> impl_change -> git_branch (41), impl_change -> howto_question -> build_run (37), howto_question ->
  impl_change -> howto_question (37), build_run -> howto_question -> impl_change (36), build_run -> impl_change -> other
  (35), howto_question -> impl_change -> build_run (35), impl_change -> git_branch -> impl_change (35)

Group-level flows (intent clusters):

- Coordination and code review -> Change-making -> Coordination and code review (61), Change-making -> Coordination and
  code review -> Change-making (51), Change-making -> How-to and workflow -> Change-making (50), Execution and
  validation -> How-to and workflow -> Execution and validation (48), Execution and validation -> Change-making ->
  Execution and validation (48), Change-making -> Execution and validation -> Change-making (47), How-to and workflow ->
  Execution and validation -> How-to and workflow (44), How-to and workflow -> Change-making -> Execution and validation
  (43), Coordination and code review -> Execution and validation -> Coordination and code review (43), Coordination and
  code review -> Other/uncategorized -> Coordination and code review (43)

## Codebase focus (paths referenced in prompts)

This is a proxy for where editing attention concentrates. It is based on file-path mentions inside prompts (repeats
counted), not on actual diffs.

- Prompts with at least one path: 732 (12.2%)
- Total path mentions: 1822

Top-level directories referenced:

| Directory   | Mentions | Share |
| ----------- | -------: | ----: |
| packages    |      875 | 48.0% |
| metta       |      369 | 20.3% |
| tests       |      294 | 16.1% |
| agent       |       78 |  4.3% |
| recipes     |       78 |  4.3% |
| common      |       41 |  2.3% |
| experiments |       28 |  1.5% |
| tools       |       16 |  0.9% |
| app_backend |       15 |  0.8% |
| mettagrid   |       10 |  0.5% |
| docs        |        6 |  0.3% |
| cogames     |        4 |  0.2% |
| devops      |        3 |  0.2% |
| scripts     |        2 |  0.1% |
| mettascope  |        2 |  0.1% |

Within `metta/` (subfolders, counts >= 3):

| Subfolder | Mentions | Share |
| --------- | -------: | ----: |
| rl        |      238 | 68.2% |
| cogworks  |       37 | 10.6% |
| tools     |       32 |  9.2% |
| sim       |       27 |  7.7% |
| setup     |        5 |  1.4% |
| packages  |        4 |  1.1% |
| utils     |        3 |  0.9% |
| metta     |        3 |  0.9% |

Within `packages/` (package names, counts >= 2):

| Package        | Mentions | Share |
| -------------- | -------: | ----: |
| cogames        |      427 | 49.8% |
| mettagrid      |      343 | 40.0% |
| tribal_village |       34 |  4.0% |
| cortex         |       27 |  3.1% |
| pufferlib-core |       25 |  2.9% |
| gitta          |        2 |  0.2% |

## Flow diagrams (ASCII)

Text-only views of the most common flow patterns:

Top 2-step transitions (tree view):

```
impl_change
├─ howto_question (127)
├─ build_run (109)
├─ other (106)
└─ git_branch (102)
howto_question
├─ build_run (112)
├─ impl_change (112)
└─ other (92)
build_run
├─ howto_question (119)
└─ impl_change (116)
git_branch
├─ impl_change (121)
└─ build_run (88)
other
└─ howto_question (90)
```

Top 3-step transitions (path list):

```
impl_change -> howto_question -> impl_change (49)
howto_question -> build_run -> howto_question (43)
build_run -> howto_question -> build_run (43)
impl_change -> build_run -> impl_change (42)
build_run -> impl_change -> build_run (41)
git_branch -> impl_change -> git_branch (41)
impl_change -> howto_question -> build_run (37)
howto_question -> impl_change -> howto_question (37)
build_run -> howto_question -> impl_change (36)
build_run -> impl_change -> other (35)
howto_question -> impl_change -> build_run (35)
impl_change -> git_branch -> impl_change (35)
```

Group-level transitions (path list):

```
Coordination and code review -> Change-making -> Coordination and code review (61)
Change-making -> Coordination and code review -> Change-making (51)
Change-making -> How-to and workflow -> Change-making (50)
Execution and validation -> How-to and workflow -> Execution and validation (48)
Execution and validation -> Change-making -> Execution and validation (48)
Change-making -> Execution and validation -> Change-making (47)
How-to and workflow -> Execution and validation -> How-to and workflow (44)
How-to and workflow -> Change-making -> Execution and validation (43)
Coordination and code review -> Execution and validation -> Coordination and code review (43)
Coordination and code review -> Other/uncategorized -> Coordination and code review (43)
```

## Prompt composition and specificity

Heuristic indicators of prompt structure and specificity:

- Multi-step directives (has 'and/then' + action verb): 1626 (27.2%)
- Starts with 'can you': 836 (14.0%)
- Starts with 'please': 176 (2.9%)
- Mentions review/audit: 529 (8.9%)
- Mentions diff/compare/vs: 479 (8.0%)
- Mentions run/execute: 1216 (20.4%)
- Mentions tests/CI: 419 (7.0%)
- Mentions lint/format: 187 (3.1%)
- Mentions fix: 582 (9.7%)
- Mentions refactor/simplify/cleanup: 497 (8.3%)
- Includes file path or filename: 2227 (37.3%)
- Includes explicit command/tool reference: 1875 (31.4%)
- Includes URL: 157 (2.6%)
- Includes PR reference (#NNNN): 2 (0.0%)
- Includes backticked code/commands: 285 (4.8%)
- Contains error/trace indicators: 385 (6.4%)

## Assistant response analysis

- Total assistant responses: 5593
- Response length: median 103 words (p25 61, p75 161, p90 245)

Response length buckets:

| Response length (words) | Count | Share |
| ----------------------- | ----: | ----: |
| <= 10                   |    23 |  0.4% |
| <= 25                   |   186 |  3.3% |
| <= 50                   |   834 | 14.9% |
| <= 100                  |  1699 | 30.4% |
| <= 200                  |  1953 | 34.9% |
| <= 400                  |   768 | 13.7% |
| > 400                   |   130 |  2.3% |

Assistant response content signals:

- Contains code fences: 668 (11.9%)
- Contains diff/patch markers: 2 (0.0%)
- Contains bullet lists: 561 (10.0%)
- Mentions tests not run: 321 (5.7%)
- Mentions tests run/passed: 737 (13.2%)

Change size proxy (based on number of file paths mentioned in a response):

- Small (0-1 files): 1206 (21.6%)
- Medium (2-4 files): 2151 (38.5%)
- Large (5+ files): 2236 (40.0%)

Large change responses by dominant session category (top categories):

| Dominant category       | Large responses | Share of large |
| ----------------------- | --------------: | -------------: |
| Implementation/refactor |             683 |          30.5% |
| Build/run/install       |             559 |          25.0% |
| Git/branch operations   |             411 |          18.4% |
| How-to/questions        |             254 |          11.4% |
| Other                   |             180 |           8.1% |
| PR management/metadata  |              77 |           3.4% |

## Assistant failure modes and oversight signals

These counts capture explicit signals in assistant responses (self-reported uncertainty, inability, or incomplete
execution).

- Blocked/permission/sandbox mentions: 261 (4.7%)
- Tool/command failure mentions: 63 (1.1%)
- Uncertainty language: 412 (7.4%)
- Explicit mistake signals: 3 (0.1%)
- Apologies: 7 (0.1%)
- Tests not run: 321 (5.7%)
- Clarification requests: 1 (0.0%)

Interpretation: these are lower-bound indicators of oversight or friction because they rely on explicit self-reporting
in text. Actual misses are likely higher than these counts suggest.

## Category transitions within sessions

Most common transitions between prompt categories in multi-prompt sessions:

- impl_change -> howto_question (127), git_branch -> impl_change (121), build_run -> howto_question (119), build_run ->
  impl_change (116), howto_question -> build_run (112), howto_question -> impl_change (112), impl_change -> build_run
  (109), impl_change -> other (106), impl_change -> git_branch (102), howto_question -> other (92), other ->
  howto_question (90), git_branch -> build_run (88)

## Temporal trends

Prompt and session volume by month:

| Month   | Prompts | Sessions |
| ------- | ------: | -------: |
| 2025-11 |    2001 |      256 |
| 2025-12 |    3428 |      499 |
| 2026-01 |     546 |      115 |

- Note: 2026 data currently covers January only, so year-over-year comparisons are not like-for-like.

## Lexical themes (filtered prompts only)

Lexical analysis was run on prompts <= 200 words (5707 prompts) to reduce skew from log dumps and large pasted outputs.

Top unigrams:

- run (1285), branch (1239), what (989), file (986), just (968), make (897), changes (888), main (865), want (790),
  think (782), code (766), not (684), then (650), have (647), see (625), line (592), but (538), all (535), great (500),
  metta (484), now (483), need (481), policy (474), merge (471), use (464)

Top bigrams:

- branch main (312), make sure (277), code changes (198), review code (194), changes against (194), against base (194),
  base branch (194), provide prioritized (194), prioritized actionable (194), actionable findings. (194), see what
  (143), current branch (122), then run (122), cogames play (119), more concise (114)

## Tool and command mentions

Commands/tools referenced in prompts (keyword match):

- git: 439, python: 363, uv: 344, gh: 65, wandb: 61, gt: 61, docker: 27, bazel: 12, tmux: 7, pip: 4, npm: 1

## Common prompt openers

Top first words across prompts (alphabetic only):

- can (1123), i (421), great! (288), review (195), please (176), we (149), is (128), this (100), what (95), hi! (88), do
  (80), lets (80)

## Observed prompt archetypes

These describe common prompt shapes and the workflow intent behind them:

- Review/audit prompts: ask for prioritized findings against `main` or a merge base, typically with `git diff` or PR
  references. These cluster under Coordination and code review.
- PR metadata maintenance: requests to update titles/descriptions, check CI status, or manage Graphite stacks. These are
  short, directive prompts.
- Implementation/refactor asks: direct changes with conciseness constraints, often to reduce indirection or inline
  helpers, sometimes accompanied by follow-up checks.
- Execution and validation: run a command, test a checkpoint, or reproduce a failure. These often chain actions (run +
  verify + summarize).
- Debugging/triage: include error messages, logs, or CI failures and ask for investigation or fixes.
- How-to queries: short questions about usage, behavior, or differences between versions, frequently used for quick
  clarification.
- Meta-planning prompts: process or workflow guidance, such as "how should we structure this" or "what should we do
  next".
- Branch hygiene prompts: audit against `origin/main`, clean up diffs, or check for unintended changes before merge.
- Compatibility/safety prompts: validate backwards compatibility, optional dependency behavior, or safe defaults before
  release.
- Tooling configuration prompts: set up linters, formatters, or CLI flags to keep workflows consistent and repeatable.

## Investigation patterns (more in-depth)

Common investigation flows that show up across sessions:

- Repro-first: run the command or reproduce the failure, then collect the minimal failing output.
- Diff-first: compare branch vs `main` or a merge base, then focus on changes that touch the failing surface area.
- Read-the-errors: parse the traceback/logs and trace to the first actionable frame.
- Narrow the scope: ask for the smallest file set or command to reduce uncertainty before making changes.
- Verify the fix: re-run the command, lint, or targeted test to confirm the change.
- Summarize impact: provide a short explanation of the root cause, fix, and any follow-up work.

## Prompt templates that match my usage

- "Can you [task] in [file/path], then [run command], and summarize changes?"
- "Please update [file] to [behavior], keep it concise, and run [lint/test]."
- "Review the diff vs main and list prioritized findings."
- "How do I [action] in this repo? Provide exact commands."
- "Refactor [module] to [goal], avoid unrelated changes."
- "Fix [error/traceback] and explain the root cause."
- "Audit our branch for unintended changes vs main and suggest cleanups."
- "Run [command], verify output, and tell me what changed."
- "Check CI status for PR [#] and summarize failures."
- "Make this more concise without changing behavior; keep diff minimal."
- "Find existing helpers or patterns before adding new code."
- "Summarize what this block does and why it matters."

## Claude Code usage (Metta workspaces)

A small set of Claude Code sessions is present locally. This likely reflects partial retention or a reset/migration at
some point; older sessions may not be available on disk.

- Project session files scanned: 10
- Sessions detected: 7
- User prompts: 37
- Time span: 2026-01-06 to 2026-01-07
- Questions vs imperatives: 11 questions (29.7%) vs 26 imperatives (70.3%)
- Prompt length: median 11 words (p25 3, p75 31, p90 33)
- Prompts with explicit commands: 14 (37.8%)
- Prompts with file paths: 3 (8.1%)

Category distribution (Claude Code, same heuristic labels):

| Category                | Count | Share |
| ----------------------- | ----: | ----: |
| PR review/audit         |     0 |  0.0% |
| PR management/metadata  |     3 |  8.1% |
| Git/branch operations   |     7 | 18.9% |
| Implementation/refactor |     4 | 10.8% |
| Build/run/install       |     1 |  2.7% |
| Tests/CI                |     1 |  2.7% |
| Lint/format             |     0 |  0.0% |
| Debugging/triage        |     0 |  0.0% |
| Docs/copy               |     0 |  0.0% |
| Analysis/research       |     0 |  0.0% |
| How-to/questions        |     0 |  0.0% |
| Ops/workflow            |     0 |  0.0% |
| Other                   |    21 | 56.8% |

Interpretation: Claude usage in these workspaces is light and very recent (early January 2026), so it should be treated
as a partial slice rather than a full history. If older sessions existed, they are not present in the current
`~/.claude/projects` directories.

## Limitations

- All categorizations are heuristic (keyword-based), so categories are directional rather than exact.
- Some prompts include large log dumps or CI output, which can skew lexical counts even after filtering.
- Sessions are attributed to workspaces based on `cwd` from session metadata; missing metadata can exclude some prompts.
- Assistant change-size is inferred from file path mentions, which is a proxy, not a direct diff size measurement.
- Claude Code data appears sparse; conclusions about Claude usage should be treated as incomplete unless older logs are
  recovered.
