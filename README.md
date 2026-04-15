![- u using quicksort or bublesort? - i using ai...](materials/ai.jpg)

[Test task](https://vlrdev.craft.me/ai_test_task/b/8B8AEE1D-3A86-4E9D-915C-1944B12B021B/%E2%9C%89%EF%B8%8F-%D0%A3%D0%B4%D0%B0%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5-%D1%81%D0%BF%D0%B0%D0%BC%D0%B0)

## Run

```bash
python main.py
```

## Current architecture

- `main.py` - single entrypoint.
- `src/agent_core.py` - runtime orchestration with two roles:
  - `Critic`: builds plan, tracks progress, detects loops/offtrack, controls restarts.
  - `Executor`: chooses and calls one browser tool per step.
- `src/agent_tools.py` - tool definitions for LLM.
- `src/entry.py` - browser wrapper (single-tab guard, SoM labels, screenshot + page text state).
- `src/prompts.py` - system prompt.

## Execution model (Critic + Executor)

1. Critic builds a sequence of subtasks from the user goal.
2. Each subtask is executed step-by-step by Executor.
3. After every tool call page state is refreshed via `get_visual_state()`.
4. Critic evaluates progress and can restart current subtask on loop/offtrack.
5. If all subtasks are done -> final `SUCCESS` report.
6. If limits are exceeded -> final `BLOCKED` report with reason.

## LangGraph node map

Runtime in `src/agent_core.py` is implemented as a LangGraph state machine:

- `init`
  - Reads user goal.
  - Critic creates initial subtask plan.
  - Initializes counters and runtime state.
- `subtask_setup`
  - Starts current subtask context.
  - Resets per-subtask counters/signatures/recent actions.
- `guard`
  - Enforces hard limits (`total_steps`, `steps_per_subtask`, `restarts_per_subtask`).
  - Decides whether to continue, restart subtask, or block run.
- `snapshot`
  - Captures current page state via `env.get_visual_state()`.
  - Routes back to `guard` if snapshot fails (retries); after too many failures in a row, run is `BLOCKED` and `finalize` runs.
- `executor_decide`
  - Executor selects exactly one tool call for current subtask.
  - Routes back to `guard` on invalid/no tool-call.
- `executor_execute`
  - Executes selected tool.
  - Logs tool result and refreshes post-action snapshot.
  - Appends action/signature history into graph state.
- `critic_evaluate`
  - Critic checks loop signal and subtask progress (`done/progress/stuck/offtrack`).
  - Decides: continue current subtask, restart, move to next subtask, or block/succeed.
- `finalize`
  - Builds final report (`SUCCESS`/`BLOCKED`) and writes run-end logs.

### Main routing flow

`init -> subtask_setup -> guard -> snapshot -> executor_decide -> executor_execute -> critic_evaluate`

Then conditional transitions:

- `critic_evaluate -> guard` (continue current subtask)
- `critic_evaluate -> subtask_setup` (next subtask)
- `critic_evaluate -> finalize` (success/block)
- `guard -> finalize` (hard-stop condition)

## .env keys used by current runtime

Required:

- `OPENROUTER_API_KEY`
- `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`)
- `AGENT_MODEL` (default `openai/gpt-4o`)

Runtime limits:

- `AGENT_MAX_TOTAL_STEPS` - hard cap for all actions in one run.
- `AGENT_MAX_STEPS_PER_SUBTASK` - max actions before subtask restart.
- `AGENT_MAX_RESTARTS_PER_SUBTASK` - max restart attempts per subtask.
- `AGENT_LOOP_REPEAT_THRESHOLD` - repeated identical action threshold for loop detection.
- `AGENT_LANGGRAPH_RECURSION_LIMIT` - LangGraph `recursion_limit` (default `200`) to avoid `GraphRecursionError` on long runs.
- `AGENT_MAX_SNAPSHOT_FAIL_STREAK` - consecutive snapshot failures before the run is `BLOCKED` (default `5`).

Snapshot tuning (optional, `src/entry.py`):

- `AGENT_SNAPSHOT_LOAD_WAIT_MS` - max wait for `load` (default `8000`).
- `AGENT_SNAPSHOT_MUTATION_WAIT_MS` - SPA idle wait timeout (default `6000`).
- `AGENT_SNAPSHOT_MUTATION_IDLE_MS` - required quiet period in ms (default `500`).
- `AGENT_SNAPSHOT_FALLBACK_SETTLE_MS` - short delay when falling back to a degraded snapshot (default `350`).

Recommended baseline:

```env
AGENT_MAX_TOTAL_STEPS=45
AGENT_MAX_STEPS_PER_SUBTASK=10
AGENT_MAX_RESTARTS_PER_SUBTASK=2
AGENT_LOOP_REPEAT_THRESHOLD=3
AGENT_LANGGRAPH_RECURSION_LIMIT=200
AGENT_MAX_SNAPSHOT_FAIL_STREAK=5
```