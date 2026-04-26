# Orbit2 project rules

Orbit2 is a governance-oriented personal agent workbench — not a generic product. The canonical knowledge base is authoritative; prefer reading KB notes over inferring from code.

- KB root: `/Volumes/2TB/MAS/vio_vault/08_Agent_Workspace/Orbit2_dev/`
- Collaboration area: `Orbit2_dev/45_Collaboration/` (task lane `10_Tasks/`, handoff lane `20_Handoffs/`)

Global OpenClaw rules in `~/.claude/CLAUDE.md` (audit policy, agent separation, coding discipline) apply here too. This file adds Orbit2-specific discipline on top.

## Before non-trivial work

1. Read `Orbit2_dev/00_Overview/00_Orbit2_Agent_First_Read.md` and `Orbit2_dev/10_Principles/01_Agent_Development_Constitution.md`.
2. For collaboration slices, invoke `/orbit-collab` to load current task/handoff state before proposing changes.
3. Classify the work (decision / implementation / reference absorption / emergent idea). State in-scope / out-of-scope boundaries before expanding the artifact set.

## Code comments vs KB-side semantics (applies to every code change)

Orbit2 is **stricter than the ambient "default to no comments" rule**. The governing rules live in `Orbit2_dev/30_Governance/04_Code_Documentation_And_KB_Mirroring_Rules.md` and `Orbit2_dev/30_Governance/05_Code_File_KB_Mirroring_Scope_Rules.md`; the workflow lives in `Orbit2_dev/40_Workflow/08_Code_Semantic_Mirroring_Workflow.md`.

- **Do not write inline explanatory comments in Orbit2 code files.** Narrative, design intent, interface rationale, and "why this shape" all belong in `Orbit2_dev/25_Code_Semantics/` semantic notes — not in the source.
- The only comments allowed are genuinely load-bearing annotations: hidden constraints, subtle invariants, workarounds tied to a specific upstream bug, behavior that would surprise a reader. If removing the comment wouldn't confuse a future reader, delete it.
- **KB-mirror scope is selective, not exhaustive.** Only semantically-central files get a semantic note: runtime nucleus, core interfaces / contracts, public execution pathways, and files whose role is hard to infer from code alone. Out-of-scope files do NOT get a semantic note by default.
- Check `Orbit2_dev/25_Code_Semantics/00_Orbit2_Code_Sitemap.md` before deciding whether a file you are editing is in scope.
- When you modify an in-scope code file and its **purpose, semantic role, design intent, interface expectations, or boundary meaning** change materially, create or update the matching semantic note in the same slice. Stale semantic notes are a bug.
- If the file is brand-new and clearly in-scope (e.g., a new runtime-adjacent module, a new surface seam), create its semantic note from `Orbit2_dev/25_Code_Semantics/00_Code_Semantic_Note_Template.md`.
- Do not expand mirroring performatively. Do not create a semantic note for every file touched. Do not write speculative interface notes for files that are not in scope.
- Do not migrate narrative back into code comments when it belongs in the KB note. The KB is the home for explanation; the code is the home for execution.

## Constitutional rules (always apply)

- **Deletion-level caution for creation** — new files/notes/abstractions are not cheap. Prefer augmenting existing same-kind artifacts over creating parallel ones.
- **Knowledge-first** — migrate validated decisions before implementation bulk.
- **Scope-bounded** — make boundaries explicit before expanding notes or code.
- **Coherent rewrites over patch stacks** — when behavior/interface/role changes materially, rewrite at whole-function / whole-module granularity rather than narrow local patching that preserves structural drift.
- **Plug-and-play seams for replaceable subsystems** — persistence, context assembly, knowledge-source aggregation, governance/disclosure policies should expose interfaces that allow swapping implementations without smearing logic across unrelated layers. Do not overbuild speculative abstractions, but leave clean seams where replacement pressure is already foreseeable.
- **Migration discipline** — treat v1 material as one of `retain | reference-only | discard`. Do not silently carry forward implementation, structure, or note sprawl.
- **Human+agent workability as a first-class constraint** — common tasks map to a small number of files with short validation paths. Site-map coverage and KB-side semantic documentation stay aligned with semantically central code files.
- **No cosmetic churn** — do not rename notes / files / directories for tidiness alone.

## Task / handoff lane discipline

- Durable state lives in **task notes** at `Orbit2_dev/45_Collaboration/10_Tasks/`.
- Bounded baton-pass slices go in **handoff notes** at `Orbit2_dev/45_Collaboration/20_Handoffs/`, always linked back to the task note.
- Handoffs are auxiliary, not the spine — return durable state to the task note after acceptance; do not let a handoff become the long-running record.
- **Do not pre-fill acceptance fields.** `Handoff Status: accepted-archived`, `Audit Status: accepted`, `Commit Status: committed`, `Commit Hash: ...` must reflect events that actually occurred. "Archived" means the file was physically moved to `20_Handoffs/Archive/` after real audit acceptance.
- Accepted slices with repo changes require commit closeout as part of acceptance — not as an optional afterthought.

## Verification norms

- Handoff publishers name the exact pytest file(s) or command(s) the receiver must run. Do not expect the receiver to invent the test contract.
- For CLI / capability triggering / streaming / transcript / runtime changes, include at least one **real runtime verification step** (CLI run, session inspection) — synthetic tests alone are insufficient.
- Audit findings at CRIT / HIGH are blocking unless explicitly overridden. By-design or documentation-only deferrals must be called out with rationale, not silently dropped.

## Promotion discipline

Stable reusable rules emerge during execution. Promote them — do not leave them buried in collaboration notes.

| Rule kind | Target directory |
|-----------|------------------|
| Design principle / constitutional | `10_Principles/` |
| Governance / naming / management | `30_Governance/` |
| Architecture decision | `50_ADRs/` |
| Workflow protocol | `40_Workflow/` |
| KB-side semantic mirror of a code file | `25_Code_Semantics/` |

## Reference paths

- Orbit2 repo (this cwd): `/Volumes/2TB/MAS/openclaw-core/Orbit2`
- Orbit1 (v1 reference, read-only): `/Volumes/2TB/MAS/openclaw-core/ORBIT`
- OpenClaw reference fork: `/Users/visen24/MAS/openclaw_fork`
- Claude Code src reference: `/Volumes/2TB/MAS/openclaw-core/claude_code_src`
- Default conda env: `Orbit`
