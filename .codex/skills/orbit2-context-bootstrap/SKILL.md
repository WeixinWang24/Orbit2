---
name: orbit2-context-bootstrap
description: Bootstrap Orbit2 development context from the Obsidian knowledge base. Use when Codex starts, resumes, scopes, audits, or implements Orbit2 work and needs the right KB notes, handoffs, ADRs, code semantics, governance rules, or collaboration workflow before acting.
---

# Orbit2 Context Bootstrap

## Purpose

Load only the Orbit2 KB context needed for the current task, then act with the repo and KB aligned. Treat the KB as the project memory and routing layer, not as background decoration.

Use the Obsidian MCP tools when available. The configured Orbit2 KB path is:

`08_Agent_Workspace/Orbit2_dev`

If the KB mentions an older repo path, prefer the active workspace/cwd for code operations and treat the KB path as historical context unless the user explicitly asks otherwise.

## Default Bootstrap

At the start of Orbit2 work, read these notes first:

1. `08_Agent_Workspace/Orbit2_dev/00_Overview/00_Orbit2_Agent_First_Read.md`
2. `08_Agent_Workspace/Orbit2_dev/10_Principles/01_Agent_Development_Constitution.md`
3. `08_Agent_Workspace/Orbit2_dev/40_Workflow/02_Orbit2_Collaboration_Execution_Mode.md`
4. `08_Agent_Workspace/Orbit2_dev/25_Code_Semantics/00_Orbit2_Code_Sitemap.md`

Use summaries/headings first. Read raw/full content only when implementation, audit, or handoff details require it.

## Execution Posture

- Keep Orbit2 work knowledge-first, creation-minimizing, scope-bounded, and human+agent-operable.
- Prefer salvage mapping and existing structure over eager rebuilding.
- Keep code changes narrow and avoid cosmetic churn in code, notes, filenames, or directories.
- Use the active workspace as the repo target.
- For Python, tests, formatters, and project scripts, prefer the `Orbit` conda environment:

```bash
conda run -n Orbit python ...
conda run -n Orbit pytest ...
```

- If the `Orbit` environment is unavailable, report that clearly before using another Python environment for project work.

## Task Routing

After the default bootstrap, load task-specific notes by lane:

- **Architecture or ADR work**: read `00_Overview/03_Orbit2_Phase_1_Architecture_Spine_Summary.md`, then `50_ADRs/ADR-INDEX.md`, then the relevant ADR.
- **Code edits**: read `25_Code_Semantics/00_Orbit2_Code_Sitemap.md`, then the semantic note matching the touched package or file.
- **CLI/operator surface**: read `25_Code_Semantics/07_Orbit2_CLI_Harness_Semantic.md` and relevant handoffs under `45_Collaboration/20_Handoffs`.
- **Knowledge surface/context assembly**: read `45_Collaboration/10_Tasks/05_Knowledge_Surface_Context_Assembly_Extraction_And_Implementation_Task.md`, `25_Code_Semantics/08_Orbit2_Knowledge_Assembly_Semantic.md`, and ADR-0002.
- **Capability surface**: read `25_Code_Semantics/09_Orbit2_Capability_Surface_Semantic.md` and ADR-0003.
- **Governance surface**: read `25_Code_Semantics/11_Orbit2_Governance_Surface_Semantic.md` and ADR-0004.
- **Operation inspector/web surface**: read `25_Code_Semantics/10_Orbit2_Web_Inspector_Semantic.md` and ADR-0005.
- **Config/runtime roots**: read ADR-0010 and the `src/config/` section of the code sitemap.
- **Continuation or handoff work**: list `45_Collaboration/20_Handoffs`, read the latest active handoff relevant to the user request, then read its linked task/ADR notes.

## KB Write Discipline

Before creating or editing KB notes, read:

1. `30_Governance/02_Knowledge_Base_Naming_Convention.md`
2. `30_Governance/03_Knowledge_Base_Management_Rules.md`
3. `40_Workflow/01_Orbit2_Obsidian_KB_Management_Workflow.md`

Apply these rules:

- Prefer updating an existing note over creating a new one.
- Ensure each active note directory has exactly one governing template before adding notes there.
- Keep note names stable and locally legible; do not rename for cosmetic symmetry.
- Mirror durable code semantics into KB notes when the change affects important design intent or interface understanding.
- When semantically central code changes alter a file role, package role, public boundary, or code-semantic note, update `25_Code_Semantics/00_Orbit2_Code_Sitemap.md` in the same slice, or explicitly state why the sitemap did not need a change.
- Keep Orbit2 code files lean; put durable explanation in KB-side semantic notes rather than inline code narration.

## Current KB Map

Use these top-level Orbit2 KB areas:

- `00_Overview`: first-read, project purpose, starting posture, architecture spine.
- `10_Principles`: development constitution and core principles.
- `20_Architecture`: architecture notes that may precede or explain ADRs.
- `25_Code_Semantics`: code sitemap and file/package semantic mirrors.
- `30_Governance`: naming, KB management, mirroring, modification discipline.
- `40_Workflow`: collaboration, task lanes, handoff lanes, audit and acceptance.
- `45_Collaboration/10_Tasks`: bounded task notes.
- `45_Collaboration/20_Handoffs`: active and archived handoffs.
- `50_ADRs`: architecture decisions and ADR index.
- `70_Absorbed_Reference`: absorbed source/reference material.

## Closeout

Before final response on implementation work:

- Run focused verification through the `Orbit` conda environment when feasible.
- Mention any KB notes used or updated when they materially shaped the work.
- Confirm sitemap follow-through for semantically central code edits: updated, or not needed with a short reason.
- If a code change should alter KB semantics but no KB update was made, call that out as a follow-up instead of silently dropping it.
