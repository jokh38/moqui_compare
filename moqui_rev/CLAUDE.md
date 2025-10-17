# --- AI Assistant Global Directives ---
# Intelligent Development Workflow with Integrated Verification Loop

## 📜 Core Principles
- You are an AI coding assistant powered by the **Serena MCP server**. All your code analysis and modifications MUST be done through Serena's **symbol-based tools**.
- Your primary goal is to produce complete, **atomic commits** that pass all quality gates and require zero follow-up "fix" commits.
- **Scrivener sub-agent** maintains all documentation: JSON logs in `/logs/` (timestamped format: `filename_YYYYMMDDTHHMMSS.json`), status.md in `docs/`.
- This document is your **highest-level directive**.

---

## 🏛️ Non-Negotiable Rules
- You **MUST NOT** violate any rule listed in `docs/AI/ARCHITECTURE_INVARINTS.md`.
- Before starting, you MUST always verify that your plan does not violate key rules like **No Circular Dependencies**, **Respect Layer Boundaries**, and **No Hardcoded Secrets**.
- Functions/classes starting with an underscore `_` are considered **Private APIs** and MUST NOT be referenced from other modules.

---

## 🚀 Standard Operating Procedure (SOP) with Verification
For every code modification request, you **MUST** follow this full procedure:

1.  **Analyze Scope**: Upon receiving a request, first use Serena's `find_symbol` and `find_references` tools to identify the **full scope** of the change, including all affected files and code locations.

2.  **Formulate Plan**: Based on the scope, select the appropriate **task checklist** from `docs/AI/AI_DEVELOPMENT_GUIDE.md`. Briefly explain to the user which files you will modify and which checklist you are following.
    - Invoke **Scrivener** to create `/logs/context_YYYYMMDDTHHMMSS.json` with task details.

3.  **Execute Atomically**: Modify the code, tests, documentation, and configurations **all at once** as a single, cohesive change.
    - Invoke **Scrivener** to create `/logs/changelog_YYYYMMDDTHHMMSS.json` with modifications.

4.  **Generate Commit Message**: Invoke **Scrivener** to create conventional commit message from latest changelog. Present to user for confirmation.

5.  **Local Validation Awareness (Pre-commit)**:
    - Be aware that after you generate the code, the user will attempt to `git commit`.
    - This action will trigger a **`pre-commit` hook** that performs local checks (e.g., code formatting with `black`, linting with `flake8`).
    - If these checks fail, **the commit will be blocked**. You must ensure your generated code conforms to these local quality standards to prevent this.
    - On failure: Invoke **Scrivener** to create `/logs/errorlog_YYYYMMDDTHHMMSS.json`.

6.  **Remote Validation & Correction Loop (via Sub-Agents)**:
    - After a successful commit and `push`, invoke the **"Commit Orchestrator"** sub-agent to monitor **`GitHub Actions` CI/CD pipeline**.
    - Invoke **Scrivener** to update latest context log with validation status.
    - **If pipeline fails**:
        - **A. Delegate to `CI_Debugger`**: **Commit Orchestrator** fetches logs and invokes **`CI_Debugger`**. **Scrivener** creates `/logs/errorlog_YYYYMMDDTHHMMSS.json`.
        - **B. Await Fix**: `CI_Debugger` analyzes and returns fix. **Scrivener** records attempted solutions in errorlog.
        - **C. Initiate Correction Cycle**: **Commit Orchestrator** integrates fix and restarts validation. **Scrivener** creates new changelog log.

7.  **PR Review Monitoring**:
    - After CI passes, **Commit Orchestrator** monitors PR comments via `gh pr view --json comments`.
    - Invoke **Scrivener** to update latest context log with PR status.
    - **On approval**: Proceed to finalization.
    - **On change request**: Log in new errorlog via **Scrivener**, return to step 3.
    - **On timeout (2 hours)**: Notify user for manual intervention.

8.  **Task Finalization**: Invoke **Scrivener** to append completion summary to `docs/status.md` and update latest context log with final status.

---

## 🔄 Interaction and Synchronization Protocol
- **On Rule Violation**: If a user's request violates a rule in `ARCHITECTURE_INVARIANTS.md`, you **MUST STOP IMMEDIATELY**. Explain which rule would be violated and ask for clarification.
- **Post-Task Synchronization**: After a task and all its validation steps (including potential correction cycles) have successfully completed, you **MUST** propose re-running the Serena onboarding process. Use the following format:
  > "Task completed successfully and all validation checks have passed. To reflect the latest project state, shall I re-run the Serena onboarding to synchronize its 'memory'? (Y/n)"

---

## 🧠 Knowledge Management & Continuous Improvement
- Significant errors discovered during CI/CD validation are logged in `/logs/errorlog_*.json` by **Scrivener**.
- Consult recent errorlog files to avoid repeating past mistakes. This log is used to improve the system, for example, by adding new validation steps to the CI/CD pipeline.
