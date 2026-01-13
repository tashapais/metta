Claude Code is presented as a three-level skill stack with 36 specific lessons across foundations, workflow
enhancements, and advanced techniques, centered around using claude.md, planning modes, and parallel/subagent workflows
to turn Claude into a structured “team” rather than a chatty assistant.[1][2]

## Level 1: Foundations (Beginner)

- Install Claude Code where you actually work: locally for day-to-day coding, or on a remote server (AWS, DO, Hetzner)
  so you can drive it from anywhere, including a phone SSH client like Termius.[1]
- Run Claude Code inside other tools (Cursor, Windsurf, VS Code) if you prefer an IDE interface rather than pure
  terminal.[1]
- Rely on to-do lists before code: have Claude create and maintain a checklist so it can tackle large tasks without
  loops or overwriting prior work.[2][1]
- Use bash mode so Claude can read/write files, search, and handle git directly without you dropping back to a separate
  shell.[1]
- Offload documentation: ask Claude to explore the repo and generate architecture docs (for example `architecture.md`)
  to bootstrap understanding for you and for future Claude sessions.[2][1]
- Turn on auto-accept when appropriate so Claude can apply changes repeatedly without constant confirmation, then
  disable it when you need tighter review.[1]
- Switch models via `/model`: use Opus for deep analysis, Sonnet for cheaper/faster routine work, and “Opus plan mode”
  to plan with Opus while executing with Sonnet.[2][1]
- Interrupt aggressively: press Escape once to stop a bad trajectory and twice to step back to the previous prompt
  rather than letting it burn tokens.[1]
- Debug with screenshots: feed UI screenshots when something looks wrong so Claude can see the full state, or use images
  for UI designs/diagrams it should implement.[1]
- Let Claude write tests, and practice test-driven development by having it author end‑to‑end tests first, then
  implement code to satisfy those tests.[2][1]
- Create and evolve a **claude.md** file as project memory: record workflows, branching rules, build/test commands,
  analytics conventions, and documentation expectations so every task respects your standards automatically.[2][1]
- Continuously refine claude.md by asking Claude to update it whenever you discover a new preference or pattern you want
  enforced.[1]
- Use the message queue to stack instructions while Claude is still working, letting it process your queued tasks
  sequentially without idle time.[1]
- Move long prompts into markdown files and reference them with `@file` in the terminal instead of pasting massive
  prompts directly.[1]

## Level 2: Workflow Enhancements (Intermediate)

- Always start with planning mode: force Claude to produce a plan before it writes code so you can review and correct
  the approach early.[2][1]
- Combine planning mode with Opus-plan/Sonnet-execute to maximize quality during planning while keeping execution cheap
  and fast.[1]
- Use subagents during planning to generate multiple solution strategies in parallel, compare them, and adopt the best
  one for implementation.[2][1]
- Control “thinking depth” with think keywords: `think`, `think hard`, and `ultra think` to allocate more reasoning for
  tougher problems.[2][1]
- Leverage web search + fetch directly from Claude Code for research tasks, from small API questions (e.g., Stripe API
  usage) to larger tool/architecture evaluations.[2][1]
- Combine PDFs with web search so Claude can use local documents (e.g., ChatGPT Deep Research PDFs) alongside online
  sources in a single research workflow.[1]
- Use Claude Code to generate non-code artifacts: PRDs, UX guides, API docs, and technical design docs that are tightly
  grounded in your actual repo.[2][1]
- Track changes automatically by having Claude maintain changelogs, feature lists for the marketing site, and decision
  docs explaining why certain choices were made.[1]
- Integrate with GitHub Actions: run `/install gh actions`, then tag Claude on issues and PRs so it can propose fixes,
  open PRs, and review PRs via CI rather than your local machine.[2][1]
- Adopt a **product manager** mindset:
  - Over-specify context and constraints instead of tossing vague prompts.
  - Stop reviewing every line of generated code and instead verify behavior, tests, and user experience at higher
    abstraction levels.[2][1]

## Level 3: Advanced Techniques (Master)

- Use parallel subagents to explore multiple solutions at once for complex features or gnarly bugs, then choose or merge
  the best ideas.[2][1]
- Run multiple Claude instances concurrently via git worktrees: create separate worktrees per feature in a hidden folder
  (e.g. `.trees`), run a Claude session per worktree, and later ask Claude to merge branches and resolve
  conflicts.[1][2]
- Keep git history clean and reviewable by isolating each Claude instance’s work to its own branch/worktree before
  merging.[1]
- Define custom slash commands as reusable prompt macros: create `.claude/commands/*.md` files with description, allowed
  tools, and the command prompt, especially for repetitive tasks like changelog generation.[2][1]
- Decide what belongs in claude.md vs commands: persistent context and rules go into claude.md (always loaded), while
  reusable one-shot behaviors live in custom commands (not auto-added to context).[1]
- Create specialized subagents for focused roles (UX design, API design, security review, test running, DB
  administration, analytics, etc.) with their own tools and system prompts.[2][1]
- Use the `/agents` command and built-in wizard to set up and manage subagents rather than hand-authoring everything,
  letting Claude scaffold the agent definitions.[1]
- Let main Claude automatically delegate to subagents when their usage descriptions match the task, or explicitly route
  tasks by referencing a particular agent in your prompt.[1]
- Extend Claude Code with MCP servers to act in external systems, especially:
  - Database MCPs (MongoDB, Postgres, Supabase) for direct DB operations.
  - Playwright MCP for browser automation and UI testing.
  - Figma MCP for design-to-code workflows.[2][1]
- Recognize that MCP capabilities will grow and become a central way to connect Claude Code to a broader tool
  ecosystem.[2][1]

## Cost and Value Lessons

- Claude Code has no free tier; access it via Claude paid plans (e.g., Pro at about 20 USD/month, Max tiers at roughly
  100 and 200 USD/month for higher rate limits) or via the Anthropic API.[2][1]
- For individual builders, the video recommends picking a Pro or Max plan instead of paying directly for heavy Anthropic
  API usage, since API costs can be high outside a company setting.[1][2]
- Anthropic announced additional weekly rate limits for Claude Max users starting around late August, but the creator
  still considers Claude Code good value despite this.[2][1]

[1](https://www.youtube.com/watch?v=rfDvkSkelhg)
[2](https://lilys.ai/notes/en/claude-code-20251021/6-months-claude-code-lessons-summary)
[3](https://www.linkedin.com/posts/vladimirgorej_6-months-of-claude-code-lessons-in-27-minutes-activity-7407401446477873152-cMic)
[4](https://www.reddit.com/r/ClaudeAI/comments/1n7jut7/6_months_of_claude_code_lessons_in_27_minutes/)
[5](https://www.linkedin.com/posts/eric-riddoch_6-months-of-claude-code-lessons-in-27-minutes-activity-7407225535946223616-e71G)
[6](https://creatoreconomy.so/p/20-tips-to-master-claude-code-in-35-min-build-an-app)
[7](https://www.youtube.com/watch?v=jWlAvdR8HG0) [8](https://www.youtube.com/watch?v=NmKdYlODC24)
[9](https://github.com/ykdojo/claude-code-tips) [10](https://maven.com/alpha-clarity/ship-with-claude-code-replit)
[11](https://www.facebook.com/groups/BigDataPakistan/posts/24496569870017045/)
[12](https://www.youtube.com/watch?v=vPpb_0Ie-QU) [13](https://x.com/avthars?lang=en)
[14](https://www.youtube.com/watch?v=SUysp3sJHbA) [15](https://www.youtube.com/watch?v=MnRzxzOMuA0)
