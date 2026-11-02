# Knowledge assessment integration — Phase A

Status: **blocked before Phase B**. Inspected on 2026-11-02.

PR [#1](https://github.com/Mohdqasem5/seniorproject/pull/1) is open and
unmerged. This assessment is based on its implementation commit
`eed664efd4766021aaca0207304153ed1787dfb8`, not an assumed merge into main.
The current main commit, `b08b178689cf655485a6d512336d95c67160cde6`, contains
only `README.md` and `ml/notebooks/Rihla_ML_Weakness_Detection.ipynb`.
The only advertised remote branches are main and PR #1's assessment branch.
The latter adds the inference module, tests, final notebook and research archive.
Neither branch contains a Spring Boot project, Java application code, database
schema/migrations/ERD, trivia persistence, authentication, or API conventions.

## Compatibility assessment

| Requirement | Existing Support | Action Required |
|---|---|---|
| Question difficulty | Yes in Python configuration; no application storage supplied | Locate administrator-managed labels and retain historical label/version snapshots. |
| Answer correctness | Yes in Python input validation; no authoritative backend records supplied | Map backend-corrected outcomes, never client-reported correctness. |
| Historical categories | Yes as caller-supplied category IDs/names; no application taxonomy supplied | Map the existing application taxonomy when its schema is available. |
| Answer chronology | Yes in Python; no persisted application chronology supplied | Supply reliable answer and feedback timestamps from stored history. |
| Completed games | Yes in Python validation; no game/session service supplied | Supply complete session records and responses; preserve five valid ten-question games. |
| Python assessment functions | Yes on PR #1's branch | Reuse `analyze_historical_knowledge`, `estimate_user_ability`, and `update_user_ability` directly. |
| Backend integration support | No | Provide the actual Spring Boot source, persistence mappings and authentication conventions before implementing an adapter or route. |

The intended PostgreSQL → Spring Boot → Python flow cannot be implemented
against this repository snapshot. There is no authenticated user context from
which to derive ownership and no database query or service to adapt. Python's
user filtering is not authentication or authorization. Creating a public Python
endpoint or inventing a Java service/schema now would introduce an unrelated
architecture rather than integrate the existing application.

## Existing reusable contract

Relevant files (paths relative to repository root):

- `ml/knowledge_assessment/fixed_difficulty_irt.py`: implementation and public
  `analyze_historical_knowledge(user_id, completed_games, interactions, configuration=None)` alias.
- `ml/knowledge_assessment/test_fixed_difficulty_irt.py`: small synthetic tests.
- `ml/notebooks/Rihla_Knowledge_Assessment_Final.ipynb`: contract explanations;
  not executed or changed for this audit.

The public function expects **game records**, not an integer completed-game count.
Games require `user_id`, `game_id`, valid `status`, and `completed_at` for completed
games. Answers require `user_id`, `attempt_id`, `game_id`, `question_id`,
`category_id`, `category_name`, `difficulty`, binary `answered_correctly`, and
timezone-aware `answered_at`. The assessment derives `game_status` from its game
record and rejects supplied conflicts. Supply `question_version` and actual
`feedback_available_at`; missing feedback time defaults to game completion.
These are **Python input names, not verified database column names**.

Outputs already include eligibility, valid game count, per-category response and
correct counts, actual accuracy percentage, smoothed accuracy fraction, ability
in logits, conditional uncertainty, evidence and availability status. Five valid
completed games are mandatory; each has ten distinct questions. Abandoned,
incomplete and short games do not count. Duplicate games are rejected; identical
answer retries are deduplicated and conflicting retries rejected. Category
evidence criteria remain inherited/provisional; no Strong/Weak cutoffs are added.
Manual Easy/Medium/Hard mapping remains fixed at -1/0/+1 by default.

## Smallest path to unblock

1. **Provide the existing backend repository/branch and its schema/migrations.**
   Locate its authenticated principal, trivia/session services, repositories,
   authoritative correctness and administrator question data before coding.
2. Map those actual records to the contract above. Determine missing persistence
   fields only from the real schema; no schema change or new table is justified
   by this snapshot. Historical label/category snapshots or immutable question
   versions are needed so later administrator edits do not reinterpret old answers.
3. Follow that backend's internal-call and API conventions, deriving user identity
   from authentication and protecting any internal Python interface. Then add
   focused mapping, ownership, transport-error and end-to-end contract tests.

## Executed verification

From the repository root:

```sh
python -B -m unittest discover -s ml/knowledge_assessment -p test_fixed_difficulty_irt.py -q
```

**24 tests passed**, with no failures, errors or skips. These verify the existing
Python estimator's fixed mapping, category separation, chronology, duplicate
handling, input validation, five-game rules and state reload behavior. They do
not verify database conversion, unauthorized-access prevention, backend-to-Python
transport, or service-error handling: those application components are absent.
No integration endpoint, backend, schema, research change or placeholder was added.
