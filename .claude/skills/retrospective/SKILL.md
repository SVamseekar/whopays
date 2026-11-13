---
name: retrospective
description: Use in WhoPays at the end of a working session, before context is compacted or cleared, after a GPU run that hit a new error or workaround, or when the user says "wrap up", "save what we learned" or "retrospective". Captures failed attempts, gotchas and corrections into the skills and logs so later sessions do not repeat them.
---

# Retrospective

## Overview

Lessons die with the session unless written down while fresh. Failed attempts are the most
valuable part: they say which paths to skip (Sionic AI found the "failed attempts" section is
the one people read most). This skill moves each lesson to the one place a future session will
look for it.

## Steps

1. **List what happened** from this session: errors hit and their fixes, workarounds, user
   corrections, decisions, costs, things that took longer than expected. One line each, with the
   exact error string or number.
2. **Route each item** to exactly one home:

   | Kind of lesson | Home |
   | --- | --- |
   | GPU, Kaggle, Modal or HF failure and its fix | `launch-gpu-run` known-failures table |
   | Log parsing or gate-reading mistake | `ingest-run-log` |
   | Research or literature process gap | `references/literature-protocol.md` |
   | Saving found, or a candidate not yet done | `docs/EFFICIENCY_LOG.md` |
   | Scientific or method decision | `docs/DECISIONS.md` (dated paragraph) |
   | A run, including failed or stopped | `docs/RUN_LOG.md` |
   | User preference about how to work | auto-memory (feedback type) |
   | Fact about code that the code already shows | nowhere; skip it |

3. **Write the failed attempt, not only the fix**: what was tried, why it failed, what worked.
   Use the exact error text so a future search finds it.
4. **Prune**: if a skill line is now wrong (a fixed bug, a changed command), edit or delete it.
   A stale instruction is worse than none.
5. **Check skill size**: a `SKILL.md` over about 150 lines moves scenario detail into its
   `references/` folder.
6. **Commit** by exact path, push, and tell the user in one line per changed file.

## Common mistakes

- Writing a session narrative instead of routable lessons.
- Recording the fix without the failed attempt or the error text.
- Adding a new skill when an existing skill's table is the right home.
- Recording numbers that did not come from a logged run.
