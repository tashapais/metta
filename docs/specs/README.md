# Specs

This directory contains specification documents for proposed features, architectural changes, and significant
modifications to the codebase.

## Purpose

Specs provide a structured way to:

- Document design decisions before implementation
- Enable asynchronous review and discussion
- Create a historical record of why things were built a certain way
- Align the team on approach before investing in code

## When to Write a Spec

Write a spec when:

- Introducing a new system or major feature
- Making breaking changes to existing APIs
- Changes that affect multiple teams or components
- Architectural decisions with long-term implications
- You want feedback before investing significant time in implementation

Skip the spec when:

- Bug fixes
- Small, self-contained changes
- Refactoring that doesn't change behavior
- Changes with obvious implementation

## Process

1. Copy `TEMPLATE.md` to a new file named `NNNN-short-title.md`
2. Fill in what you know, leave the rest for later
3. Get feedback and iterate
4. Update the spec as you learn during implementation

## How Complete Should a Spec Be?

Specs help us go faster by avoiding wasted work on the wrong thing. The priority is product decisions that have the
biggest impact on implementation.

**Nail down early:**

- The problem and why it matters
- What "done" looks like
- What you're explicitly not building

**Figure out as you go:**

- Implementation details
- Edge cases
- Technical approach

Start with open questions—that's normal. Update the spec as you learn. It's a thinking tool, not a contract.

## Template Sections

- **Summary**: One paragraph describing what this feature does and who benefits
- **Problem**: What's broken, missing, or painful today
- **Solution**: How the feature solves the problem
- **Goals**: What must be true for this to be considered complete
- **Non-Goals**: What's explicitly out of scope
- **Design**: How it will be built (can be sparse initially)
- **Open Questions**: Unresolved decisions—it's fine to have many of these

## Status Values

| Status      | Meaning                                |
| ----------- | -------------------------------------- |
| Draft       | Work in progress, not ready for review |
| In Review   | Ready for feedback                     |
| Approved    | Accepted, ready for implementation     |
| Implemented | Completed and in production            |

## File Naming

Use the format: `NNNN-short-descriptive-title.md`

- `NNNN` is a zero-padded sequential number
- Use lowercase with hyphens for the title
- Example: `0001-event-sourcing-for-replays.md`
