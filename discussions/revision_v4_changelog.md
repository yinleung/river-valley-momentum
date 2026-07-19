# Revision changelog — response to review_v4.md (2026-07-18)

Executed the JMLR restructuring directed by `discussions/review_v4.md`, with deviations adjudicated by Codex (all twelve ratified pre-edit; final diff review **PASS**). Compiled PDF: 34 pages (was 30). Note: the requested "GPT-5.6" is not available on this Codex account (models installed: GPT-5.4, GPT-5.4-Mini, GPT-5.5); the discussion ran on **GPT-5.5 at xhigh reasoning**.

## Verdict on the review

The review's three core asks are sound and were implemented: (1) a self-contained Background and Preliminaries section separating standard material from original results; (2) splitting the old "River-Valley Dynamics" into a spectrum-emission section and a closed-loop section; (3) a paper-wide notation audit. Its *specific* rename suggestions were adapted where they would have created new collisions (h_t vs H_β/h(·); r_s vs figure pixels; a-amplitude vs bend-a) or figure-pixel mismatches; the review also missed four real collisions that were fixed on top (𝒪 as both big-O and polar factor; the DFT bin index m vs the buffer m_t; w as iterate/weights/innovation; hat as both DFT and estimator).

## New structure

| # | Section | Contents |
|---|---------|----------|
| 3 | Background and Preliminaries | 3.1 streams + **Def 1 (EMA)** + three-spectral-axes remark; 3.2 geometric modes z₀, tones, unit circle + three-roles-of-z paragraph; 3.3 windowed DFT (orthogonality, inversion, Parseval); 3.4 tone response eq (pertone) + H_β (standard-labeled) + T_eff remark + step conventions (single home of η_HB = η(1−β)) + probe-not-model remark + miniature; 3.5 straight/curved valley, hill offset d_t, λ_loc, tube, edge of stability; 3.6 direction maps 𝒟, polar(·), sinΘ, Weyl, Wedin (versions used); 3.7 notation table (Table 1, own float page) |
| 4 | Momentum as a Temporal Filter | Prop 1 (exact windowed transfer), Thm 1 (band filtering), tradeoff + boundary remarks — first original results |
| 5 | The Spectrum the River Valley Emits | Thm 2 (frequency separation), Thm 3 (stationary spectrum), Cor 1 (white-noise ceiling), Lem 1 (confinement) |
| 6 | Closed-Loop Consequences | open/closed-loop remark, eq (7), Prop 2 (stability), Prop 3 (variance), Cor 2 (hill loss), burn-in, Prop 4 (forced response), Prop 5 (curved reduction), β-window remark |
| 7 | Filter-First for Nonlinear Direction Maps | unchanged order; opening menu moved to 3.6 |
| 8–12 | Experiments / Real-Network / Optimizer / Scale / Discussion | internal order unchanged |

Corollary numbers swapped (ceiling is now Cor 1, hill loss Cor 2); every cross-reference is `\Cref`-label-based and was verified to resolve (source-level scan: zero dangling).

## Symbol-renaming table (old → new)

| Old | New | Scope |
|-----|-----|-------|
| generic stream x_t, x̂(ω) | s_t, ŝ(ω) | preliminaries (x is river-only now) |
| geometric ratio z | z₀ (fixed ratio) vs z (free variable) vs e^{iω} (unit circle) | 3.2 |
| e = η(1−β)λ | χ = η(1−β)λ | closed loop |
| ρ = ηλ−1 (Thm 2) | q_h | ρ-family now filter-only (ρ_high, ρ_Ω, ρ̄) |
| a = 1−ηλ (Thm 3) | a₀ | |
| δ_t hill offset (Prop 5) | d_t (matches Measurements + d_rms pixels) | δ_t now uniquely the singular-value gap |
| forcing s_t (Prop 4) | u_t (transform U(z)) | |
| Cor-3 Markov fraction θ | τ | θ_β untouched |
| EMA weights w_k | c_k | w is the iterate only |
| AR(2) innovation w_t; slow-root δ | inlined −η(1−β)ξ_t; proof-local ς | Prop 3/Prop 2 proofs |
| bend amplitude a | a_b | Thm-5 band amplitudes (a, α) untouched |
| river speed v | v_riv | tone amplitude keeps v |
| Prop-6 signal value σ | σ_S | bare σ = noise std only |
| matrices G_t,S,A,A_t,M_t,X_t,P_t,C_ω,D_ω,Ξ,U,Σ,V,u,v,u′,v′ | bold 𝐆_t,𝐒,𝐀,… | filter-first + E5; ℍ-generic g_t,m_t,m̃_t stay italic (registered exception) |
| polar factor 𝒪(·) | polar(·); 𝒪 is big-O only (bare O unified too) | fixes a collision the review missed |
| E5 hill symbol H_t | eliminated: (−1)^t𝐀 inline | H_β unambiguous |
| ĝ^LB, ξ̂ (E12) | ḡ^LB, ξ^res | hat reserved for the windowed DFT |
| dims m×n | d₁×d₂ | m is the buffer |
| indices u, r; bin index m | j, n; ℓ′ | u freed for forcing; bin index no longer shadows the buffer |
| Li et al.'s T (2T−1, T^{−1/4}) | eliminated: window = T_eff, rate = T_eff^{−1/4} exactly | incl. the intro's misleading T^{−1/4} |
| Rem-boundary/Cor-ceiling generic PSD S(ω), S | S_g(ω) (subscript names the stream) | |
| Lemma-1 second direction v; D_T kept | u′ | |
| Spearman ρ, E4's R(ω) | symbols dropped from prose | ρ_high / tube radius R own the letters |

Declined review items, with reasons: ℍ keeps blackboard (already distinct from H_β); rank r kept (σ_{r+1} in figure pixels; the true offenders were the index reuses); Δ^gap/h_t/r_s/𝒳 superseded by the above; full z-transform/ROC subsection replaced by the three-roles paragraph (the z-transform appears once, in Prop 4's proof).

## Figures re-rendered (labels only, from cached run records — no experiment re-runs)

fig_e10_bendwindow (river speed v_riv axis), fig_e5_matrix (title (−1)^t𝐀), fig_e5_wedin (σ_{r+1}(𝐌_t) legends), fig_e11_scale (HFER(𝐆, β=0) legend), fig_e12_decomp (ḡ^LB / ξ^res legends; style.py DECOMP_STYLE).

## Issues discovered (beyond the review)

1. 𝒪 meant both big-O and the polar factor, sometimes on the same page — fixed (polar(·)).
2. The DFT orthogonality display used m as a bin index next to the buffer m_t — fixed (ℓ′).
3. w carried three meanings (iterate, EMA weights, AR-innovation) — fixed (w, c_k, inlined).
4. Intro/related-work quoted Li et al.'s rate as T^{−1/4} with T ambiguous against our window length; their window constant 2T−1 with T = 1/(1−β) equals T_eff exactly, so all three sites now read T_eff^{−1/4} — an identity, not an approximation.
5. The hat was overloaded (windowed DFT vs large-batch estimator) — split (bar/tag).
6. E10 kept one bare "kv"; E9/Prop-2/Related-Work restated the η_HB conversion (single-sourced to 3.4) — both caught in the Codex re-review round.

## Added assumptions

None. No theorem statement, constant, or experimental claim changed; the edits are relocations, renames, and new standard-background prose. All 416 decimal-valued numeric tokens in main.tex are identical to the pre-revision snapshot (verified by multiset comparison), so every measured value retains its lineage to the cached run records.

## Compile status

tectonic (make): zero errors, zero unresolved references; 34 pages; notation table floats to its own page in §3. CRITERION.md canonical table, register rules (bold-matrix rule + exception, hat/PSD conventions, fixed scalar names), and the dead-label log updated in the same batch; `sec:river` split into `sec:spectrum` + `sec:loop` with every reference retargeted.
