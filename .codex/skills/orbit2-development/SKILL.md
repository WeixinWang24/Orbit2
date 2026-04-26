---
name: orbit2-development
description: Project-specific development rules for the Orbit2 repository. Use when Codex is editing, testing, debugging, running scripts, or otherwise developing inside the Orbit2 project.
---

# Orbit2 Development

## Core Rules

- Prefer the `Orbit` conda environment for Orbit2 project development.
- Run Python, tests, formatters, and project scripts through the conda environment unless the user explicitly requests another environment.
- Prefer commands like:

```bash
conda run -n Orbit python ...
conda run -n Orbit pytest ...
```

- If `conda run -n Orbit ...` fails because the environment is missing or misconfigured, report that clearly and avoid silently falling back to another Python environment for project work.

## Code / KB Discipline

- For semantically central code changes, read `25_Code_Semantics/00_Orbit2_Code_Sitemap.md` and the matching semantic note before or during implementation.
- If a change alters a file's purpose, semantic role, design intent, interface expectations, or boundary meaning, update the relevant KB-side semantic note in the same slice.
- If the change alters a sitemap row, package role, public boundary, or newly mirrored file, update `25_Code_Semantics/00_Orbit2_Code_Sitemap.md` in the same slice, or explicitly state why no sitemap change was needed.
- Keep Orbit2 code files lean: durable explanation, design rationale, and interface narrative belong in KB semantic notes rather than inline explanatory comments.
