---
name: orbit2-mcp-native-benchmark
description: Use when comparing Orbit2 MCP tools against Codex/coding-agent native tools, Codex CLI MCP baselines, or Orbit2 family MCP servers; especially when measuring tool latency, bridge exposure, prompt-level benchmark behavior, or read-only external profile safety. Ensures timings are measured inside the evaluated tool path, not from outer Codex session wall-clock time.
---

# Orbit2 MCP Native Benchmark

## Core Rule

Do not treat the outer Codex tool call, `codex exec` session, MCP client process startup, or model response wall-clock time as the benchmark. Measure the actual tool operation.

For Orbit2-owned MCP tools, prefer the server-side result metadata at `metadata.tool_call.elapsed_ms`. That value is recorded inside the MCP server around the registered tool function and excludes `StdioMcpClient` process startup, MCP handshake, transport setup, and outer Codex wall-clock time.

For Codex CLI comparisons, prompt the CLI to run an internal timing script and return only the script's measured MCP/tool timings.

## Default Workflow

1. Load `orbit2-development` and follow its environment rule.
2. Decide the comparison lanes:
   - Orbit2 external aggregate MCP read-only profile.
   - Orbit2 family MCP server baseline.
   - Codex CLI or coding-agent native path, if requested.
3. Choose the benchmark shape by capability level:
   - L0/L1: compare counterpart tools when a near-equivalent native/family tool exists.
   - L2/L3: compare against a Codex CLI or coding-agent orchestration baseline, because high-level Orbit2 Toolchain/Workflow tools usually do not have a single native MCP counterpart.
4. Run the bundled script for deterministic Orbit2 aggregate-vs-family MCP timing:
   ```bash
   /Users/visen24/anaconda3/envs/Orbit/bin/python .codex/skills/orbit2-mcp-native-benchmark/scripts/benchmark_mcp_calls.py --workspace /Volumes/2TB/Dev/Orbit2 --repeats 3
   ```
5. If comparing Codex CLI, make the CLI run the same script internally or run a task-specific orchestration prompt that measures its internal steps. Do not use the elapsed time of the outer `mcp__codex_cli__.codex` call.
6. Report:
   - `tool_count`
   - `mutation_visible`
   - per-case aggregate/family `metadata.tool_call` mean/min/max ms
   - transport/cold-start timings separately when useful, never as tool-call timing
   - payload equivalence or payload sanity checks
   - any benchmark caveats such as stdio cold-start cost, clearly labeled as transport overhead

## Counterpart vs Orchestration

Use direct counterpart comparison only when the task shape is genuinely similar:
- raw file read vs raw file read;
- structured file region vs structured/family region read;
- git changed files vs git changed files.

For Orbit2 L2 Toolchain and L3 Workflow tools, do not expect Codex native MCP to have a single equivalent tool. Compare these against an agent-orchestrated baseline:
- Orbit2 `repo_scout_diff_digest` vs Codex CLI using native/raw steps such as git status, git diff, file reads, and summarization;
- Orbit2 `repo_scout_changed_context` or `repo_scout_impact_scope` vs CLI-directed repository reconnaissance;
- Orbit2 `inspect_change_set_workflow` vs CLI orchestration that gathers facts and emits admissible next-step options.

For orchestration benchmarks, measure and report:
- internal tool-call count and step count;
- internal measured tool time where available;
- total number of model turns if a model-mediated baseline is used;
- output schema stability and payload completeness;
- whether intermediate state is persisted or recoverable;
- whether mutation tools stayed hidden or unused;
- whether decision-making stayed in the correct layer.

## Read-Only Discipline

When running in a read-only sandbox, set `ORBIT2_MCP_STDERR_LOG=/dev/null` in both the parent process and MCP subprocess env. `StdioMcpClient` otherwise tries to open `.runtime/mcp_stderr.log`.

Use the Orbit2 external aggregate in read-only mode:

```bash
-m src.capability.mcp_servers.orbit2.stdio_server --workspace /Volumes/2TB/Dev/Orbit2 --profile read-only
```

Verify that mutation tool names are absent. Treat any visible write, replace, move, git mutation, branch checkout, commit, or process execution tool as a failed exposure test.

## Codex CLI Prompt Pattern

Use this shape when the user asks for Codex CLI/native comparison:

```text
Do not report Codex session end-to-end time. Run the Orbit2 benchmark script inside this CLI session and return only the JSON produced by the script. Use read-only sandbox. Set ORBIT2_MCP_STDERR_LOG=/dev/null in parent and subprocess env.
```

If the CLI writes extra narration, extract the JSON result and state that the benchmark numbers are CLI-internal MCP timings.

## Interpreting Results

`StdioMcpClient` currently starts a fresh MCP subprocess for each `list_tools` and `call_tool`. Treat its outer elapsed time as transport/cold-start overhead. Do not present those numbers as tool-call latency.

The bundled script reports `mean_ms` / `min_ms` / `max_ms` from `metadata.tool_call` when available, and reports the outer client timing separately as `transport_*`.

Small differences between aggregate and family servers can be cold-start/import noise. Prefer conclusions about:
- mutation exposure safety;
- payload equivalence;
- rough wrapper overhead;
- whether the benchmark was measured at the right layer.

For L2/L3 comparisons, avoid saying "no counterpart exists, so no benchmark is possible." The benchmark target becomes "fixed Orbit2 capability vs CLI/native orchestration for the same task intent."
