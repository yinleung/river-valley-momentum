# Code Mode (codebases/AGENTS.md)

You are reviewing code, experiment, or figure changes. Before reviewing, open and apply `./CODING.md` as your rubric. (Codex does not expand `@import`; read the file with your tools.)

Check especially:
- upstream code left **unedited** — integration lives in `adapter/` only;
- the **Integration Contract** satisfied by the adapter;
- **unified run naming** and a full config + git SHA logged for every run;
- figures are **pure** — no experiment recomputation and no live-tracker calls at render time;
- no version-suffixed files (`_v2`) and no parallel output directories.

Separate blocking issues from nits; end with PASS or FAIL.
