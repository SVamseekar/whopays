# Literature protocol (WhoPays)

Shared by `researching-design-choices`, `investigate-then-research` and
`efficiency-research-loop`. Goal: academic, balanced, cheap. Read it at the start of any
research step.

Why this exists: automated research systems fail in known ways. Sakana's AI Scientist did
keyword search over abstracts, cited a median of 5 papers (5 of 34 from 2020 or later) and
labelled known ideas as novel (Beel et al. 2025, arXiv 2502.14297). Deep research engines give
one-sided answers to debate-style questions in 54.7–94.8% of cases, and 40–80% of their
citations support the claim (DeepTRACE, ICLR 2026, arXiv 2509.04499). Each rule below
targets one of those failures.

## 1. Plan before searching (2 minutes)

Write in the note, before any search:
- the question and the options (the options must exist before evidence is collected);
- 3–5 concept groups with synonyms, for example {mixed precision, bit allocation},
  {multilingual, cross-lingual, low-resource}, {fairness, worst-group, minimax};
- the counter-questions: "when does option X fail?", "who argues against X?".

## 2. Search cheap first: scholarly API, then web

1. `python .claude/skills/researching-design-choices/scripts/litsearch.py search "<query>" --since 2022 -n 25`
   for each concept combination (OpenAlex: venue, year, citation count, DOI in one line each).
   Also `--sort cited_by_count:desc` for the canonical papers and `--sort publication_date:desc`
   for the newest.
2. Pick 2–3 seed papers. Snowball both ways: `litsearch.py refs <id>` (what it builds on) and
   `litsearch.py cites <id>` (who built on it, critics included).
3. WebSearch only for what the API misses: library source and issues, official docs, OpenReview
   reviews, workshop papers, very recent preprints. Send these searches in one turn.
4. Screen on title and abstract. Keep a candidate table; record why each dropped paper was
   dropped (off-topic, superseded, different setting).

## 3. Search both sides on purpose

For every option still alive, run at least one search for each:
- **support**: the method's own paper and follow-ups that use it;
- **against**: "limitations of X", "revisiting X", "X fails", "X does not generalize",
  reproducibility or negative-result papers, OpenReview reviews of the X paper;
- **alternatives**: the strongest competing method, steelmanned.

If no source argues against the chosen option after these searches, say so explicitly. Never
assume silence means agreement.

## 4. Read deep, selectively

- Open in full the 6–10 sources that decide the question. Abstract-only reading is allowed for
  screening, never for a number or claim that is used.
- For each source used, record: claim, exact number with unit, where it is (section, table,
  figure), setting (model, size, languages, hardware), and whether that setting matches ours
  (Qwen2.5 1.5B–7B, 11 languages, T4 FP16).
- Before naming a method in code or a decision ("KIVI's setting", "GPTQ defaults"), check the
  exact recipe in the paper's method section. The 2026-11-13 decision called per-token HQQ keys
  "KIVI's 4-bit setting"; KIVI quantizes keys per channel, and the per-token recipe collapsed
  MGSM on Qwen2.5 (RUN_LOG 17).

## 5. Grade the evidence

| Grade | Meaning |
| --- | --- |
| Strong | 2+ independent peer-reviewed sources, or one plus a replication, in a setting close to ours |
| Moderate | One peer-reviewed source, or several preprints that agree |
| Weak | A single preprint, blog, or a different setting (other model family, English only, A100) |

Mark preprints as preprints. Count one lab's papers as one source for agreement.

## 6. Write the balanced synthesis

The note's evidence section has exactly these parts:
1. **Evidence table**: source, type (peer-reviewed / preprint / code / docs), year, claim,
   number, grade, matches our setting (yes/partly/no).
2. **For each option: case for, case against**, each with citations. Same depth for both.
3. **Where sources disagree** and the likely reason (setting, metric, model size).
4. **Decision and confidence** (high / medium / low), and what evidence would reverse it.

## 7. Verify citations before committing

`python .claude/skills/researching-design-choices/scripts/litsearch.py check <id> <id> ...` for
every cited paper. Any `MISSING` must be resolved (wrong id, or the paper does not exist) before
the note is committed. Never cite a paper that was not opened.

## 8. Save the corpus for reuse

Append the final candidate table (id, title, grade, one-line relevance) to
`docs/literature/<topic>.md`. Next time, start from that file and only search for work newer
than its date. This is the main efficiency saving: the same literature is never searched twice.

## Budgets

| Step | Typical |
| --- | --- |
| API queries | 8–15 |
| Web searches | 4–8, in one turn |
| Candidates screened | 30–80 |
| Read in full | 6–10 |
| Cited | 12+, from 3+ source types, with at least one dissenting source if one exists |
