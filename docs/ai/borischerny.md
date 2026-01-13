Here is a distilled, implementation‑oriented set of instructions from the video, grouped so you can drop them into your
own workflow.[1]

---

## Core workflow philosophy

- Treat Claude Code as a **junior developer/orchestrator**, not a magic one-shot tool; expect to iterate, verify, and
  steer.[1]
- Always give Claude explicit ways to **verify its work** (tests, linters, commands, UI checks) instead of trusting raw
  output.[1]

---

## Verification‑led development

- For any coding task, tell Claude explicitly to:
  - Write or run tests along with implementation.
  - Use those tests as the main feedback loop.[1]
- If you do not want to write tests yourself, instruct Claude to:
  - Propose a test suite.
  - Implement a test for every function/feature it adds.[1]
- Use **test‑driven development** with Claude:
  - Ask it to write tests for a new feature first.
  - Only then implement the feature to satisfy those tests.[1]
- Domain‑specific verification:
  - CLI/backend: run bash commands, integration checks, etc.[1]
  - UI: use browser automation, screenshots, or MCP simulators (iOS/Android) to validate behavior and layout.[1]

---

## Designing `claude.md`

- Maintain a **project‑specific `claude.md`** file that is sent into every Claude Code session.[1]
- Keep it relatively small (Boris’s is about **2.5k tokens**). Avoid bloat.[1]
- Content to include:
  - Tech stack and key dependencies.[1]
  - Basic project structure, important directories, and key entry points.[1]
  - Code style, formatting rules, and conventions the team follows.[1]
  - “Do not do this” section (common failure modes, past mistakes).[1]
- Process around `claude.md`:
  - Make the file **unique per repo**, not global.[1]
  - For full‑stack/microservices:
    - Give each microservice its own `claude.md` (e.g., frontend vs backend).[1]
  - Ask team members to update `claude.md` multiple times per week:
    - Whenever Claude makes a mistake, add a brief note so it does not repeat it.[1]
  - Assign ownership:
    - Each team member is responsible for maintaining specific files/areas.[1]
- Optional meta‑instruction in `claude.md`:
  - Before doing any work, Claude must state how it will verify that work.[1]

---

## Modes and permissions

- **Plan mode**:
  - Always start a feature/task in plan mode.
  - Clearly describe the goal; iterate with Claude until the plan looks correct.
  - Only start execution once the plan is validated.[1]
- **Auto‑accept edits mode**:
  - After confirming the plan, run in auto‑accept mode so file edits are applied automatically.[1]
- **Dangerously skip permissions**:
  - Avoid this mode; Boris does not use it, especially in production.[1]
- **Command permissions**:
  - Configure per‑project rules for terminal commands:
    - Allowed without prompt.
    - Allowed but must ask first.
    - Fully denied.[1]
  - Store these rules in a `settings.json` under the Claude folder so they can be shared with the team.[1]

---

## Orchestration and MCP usage

- Use Claude Code as a **workflow orchestrator**, not just a code editor.[1]
- Integrate via MCP and CLIs:
  - Slack, BigQuery, Sentry, Notion, etc., accessible through their CLIs inside Claude Code.[1]
  - In Notion, let Claude:
    - Create and populate databases.
    - Use existing pages as context (e.g., video ideas, project specs).[1]
- For mobile or browser‑based verification:
  - Use browser extension or simulator MCPs (iOS/Android) to run the app and verify behavior.[1]

---

## Parallel sessions and background work

- Run **multiple Claude Code sessions in parallel** (Boris runs around five).[1]
  - Number your terminal/web tabs so you can map notifications to the right agent.[1]
- Use **web (cloud) sessions**:
  - Connect web Claude Code to GitHub.
  - Give it access to a repo so it can work in the cloud.[1]
- For long‑running tasks:
  - Use **background agents** to offload tasks:
    - Kick off work in the cloud; Claude runs in the background, then:
      - Pushes changes into a new branch for review.[1]
  - Use the **teleport command** to pull the cloud session state back into a local terminal session.[1]

---

## Long‑running verification strategies

For big or slow jobs, Boris uses three patterns:[1]

1. **Background tasks after tests**
   - Ask Claude to design and run verification tests.
   - Move the work into a background task; it reports back when complete.

2. **Stop‑hook triggered verification**
   - Configure a **stop hook** that automatically runs verification when Claude stops outputting.

3. **External verification agents (Ralph Wiggum plugin)**
   - Use an external plugin/agent to:
     - Drive and verify UI flows.
     - Run automated checks, effectively removing the human from the verification loop.

---

## Model choice and quality

- Default to **Opus 4.5 with “Thinking” enabled** for most work.[1]
- Rationale:
  - Slower and heavier than Sonnet/Haiku, but significantly fewer errors.
  - Net engineering time is lower because less steering and correction is required.[1]

---

## GitHub integration and CI hygiene

- Use the **Claude Code GitHub Action**:
  - Call it on authorized repos to review PRs.[1]
- During PR review:
  - When you notice a recurrent issue, ask Claude to add that pattern to `claude.md` so the system keeps learning.[1]
- Always run **linters and formatters**:
  - Use Claude to integrate and call language‑specific formatters and linters.
  - Treat them as verification for the last ~10% of polish to avoid CI failures.[1]

---

## Inner‑loop tooling: slash commands and sub‑agents

- **Slash commands**:
  - Encode repetitive workflows (inner loop) as custom `/commands`.[1]
  - Example: a `/github` command that runs a common GitHub flow.[1]
  - Commands live in a `commands` subfolder under your Claude folder.
  - Commit them to git to share across the team.[1]
- **Sub‑agents**:
  - Keep them focused and simple; avoid over‑engineering.[1]
  - Boris uses sub‑agents mainly for:
    - Architecture verification.
    - Refactoring existing code.
    - Validating that final builds work.[1]

---

If you want, a follow‑up can be a concrete template `claude.md` and a minimal `settings.json` plus a couple of example
`/commands` wired for your stack (e.g., Python + Docker + React).

[1](https://www.youtube.com/watch?v=B-UXpneKw6M)
