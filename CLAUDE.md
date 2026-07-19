# BETASCHEDULER — General Instructions (Claude.md)

## Identity
- Address the user as **Leon**.

## Workspace map
- `latex/` — paper writing. Governed by `latex/WRITING.md`.
- `codebases/` — code, experiments, figures. Governed by `codebases/CODING.md`.
- `discussions/` — research notes, idea drafts, design briefs. Freeform; no agent standard.
- `references/` — source papers (PDFs git-ignored; catalogued in `references/README.md`).

## Routing — which rules apply
- For any **paper-text** task: read `latex/CLAUDE.md` first; it imports the paper-writing standard. If you are not already working inside `latex/`, read it before touching prose.
- For any **code / experiment / figure** task: read `codebases/CLAUDE.md` first; it imports the code standard.
- For **research-note / reference** material (`discussions/`, `references/`): read for context and grounding; these are ungoverned — no standard applies, and do not restructure them unasked.
- Most reliable habit: launch from the subtree you are working in (`cd latex` or `cd codebases`), so its `CLAUDE.md` loads deterministically alongside this one.

## Claude ↔ Codex protocol
- **Claude executes** — implement, run, edit. **Codex reviews** — it never edits files.
- After any non-trivial change, run `codex review` on the diff, judged against the relevant standard: `latex/WRITING.md` for prose, `codebases/CODING.md` for code.
- Resolve blocking Codex findings, or surface them to Leon with your response. Treat Codex as an independent second opinion, not an authority — disagree, with reasons, when warranted.
