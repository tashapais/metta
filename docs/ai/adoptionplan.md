# AI-Assisted Development Adoption Plan

Our AI-assisted development plan aims to achieve 10x productivity without compromising code quality.

## Overview

We're running structured AI adoption sprints where team members push themselves to discover what works, what doesn't,
and develop shared practices. The aim is experiential learning—finding the rough edges, developing intuition, and
building team-wide knowledge.

## Phase 1: Learn from Others

Before diving in, study what has worked for others.

### Internal Show and Tell Sessions

Team members share their AI workflows, demonstrating what works and what doesn't:

| Presenter       | Topic/Focus |
| --------------- | ----------- |
| Richard Higgins | TBD         |
| Yatharth        | TBD         |
| Kyle Herndon    | TBD         |

### Reference Documentation

Curated summaries of external practitioners' experiences (all in `docs/ai/`):

| Document                         | Description                                |
| -------------------------------- | ------------------------------------------ |
| [aicoding.md](aicoding.md)       | Core best practices for AI-assisted coding |
| [avthar.md](avthar.md)           | Avthar's workflow and recommendations      |
| [borischerny.md](borischerny.md) | Boris Cherny's approach                    |
| [enoreyes.md](enoreyes.md)       | Eno Reyes' insights                        |

## Phase 2: AI Sprint

A focused sprint where everyone commits to creating 80%+ of their code through AI.

### Sprint Goals

1. **Primary Goal**: Develop proficiency with AI-assisted development
2. **Secondary Goal**: Identify patterns, anti-patterns, and rough edges
3. **Tertiary Goal**: Build shared vocabulary and practices

### Sprint Rules

- **80% AI-generated code target**: Push yourself to use AI for everything possible
- **Choose your tools**: Claude Code (recommended), OpenAI Codex, or others
- **Experiment freely**: Try different techniques, prompting styles, and workflows
- **Share learnings in real-time**: Post to `#softmax/ai-automation` Discord channel

### Recommended Tools

| Tool                          | Notes                             |
| ----------------------------- | --------------------------------- |
| **Claude Code with Opus 4.5** | Recommended primary tool          |
| **OpenAI Codex**              | Good alternative, worth comparing |
| **Cursor/Continue**           | IDE integrations to consider      |

### What to Share During the Sprint

Post to the Discord channel about:

- **The Good**: What's working well, time savings, surprising capabilities
- **The Bad**: Frustrations, failures, where AI falls short
- **The Ugly**: Weird edge cases, unexpected behaviors, things that almost worked

### Key Principles (from aicoding.md)

1. **This is NOT vibe coding** — Understand every line. Quality matches manual development.
2. **Small tasks > big tasks** — Break work into pieces AI can succeed at 99% of the time.
3. **Context management is everything** — Treat context as a budget. Reset sessions frequently.
4. **Models learn by mimicry** — Point to examples instead of writing complex rules.
5. **Invest in local testing** — AI can't help if it can't verify its work.

## Phase 3: Reflect and Plan

At the end of the sprint, hold a retrospective to consolidate learnings.

### Retrospective Agenda

1. **Successes**: What worked well? What should we do more of?
2. **Challenges**: What was difficult? Where did AI fall short?
3. **Standardization vs. Flexibility**: What practices should be team-wide? What's personal preference?
4. **Tooling**: What tools/configurations should we standardize?
5. **Next Steps**: What's the plan for continued improvement?

### Outputs

- Updated best practices documentation
- Standardized tool configurations (e.g., CLAUDE.md, .cursorrules)
- Training materials for new team members
- Plan for ongoing learning and tool evaluation

## Ongoing: Continuous Improvement

AI tooling evolves rapidly. We need processes to stay current.

### Staying Current

- **Tool Monitoring**: Assign someone to track major AI coding tool releases
- **Monthly Check-ins**: Brief sync on new capabilities and practices
- **Capability Calibration**: Run periodic "meta task" exercises to update our view of what tools can do
- **Abstraction Guardrails**: Document the level of abstraction that produces reliable results
- **Documentation Updates**: Keep `docs/ai/` current as we learn

### Metrics to Track (Optional)

Consider tracking to measure progress:

- Time to complete common tasks
- Code review feedback on AI-generated code
- Developer satisfaction/confidence surveys
- Session hooks for Claude Code/Codex to track branch creation through merge

## Quick Start for New Team Members

1. Read [aicoding.md](aicoding.md) for core principles
2. Set up Claude Code or your preferred tool
3. Review example workflows in this directory
4. Join `#softmax/ai-automation` Discord channel
5. Start with small, well-defined tasks and expand from there

## FAQ

**Q: What if I can't hit the 80% target?**

A: The goal is learning, not perfection. Document what you tried and why it didn't work—that's valuable data.

**Q: Should I use AI for everything?**

A: Use judgment. Some tasks (security-sensitive code, complex algorithms) may benefit from more human oversight.

**Q: What about code quality?**

A: AI-generated code should meet the same standards as manually written code. Review everything.

**Q: How do I reset context when it gets too large?**

A: Start a new session. Summarize what you've accomplished and what's next in the new session's first message.
