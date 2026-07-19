# Theory note: T1 (finite-window momentum filtering) and T3 (river-valley frequency separation)

Staging note for the theorems the experiments E1/E4 witness. Rigorous; to be ported into `latex/`
under `WRITING.md` once stable. Notation follows `idea_v1.md`. Numerically verified identities are
flagged [✓ verified] with the checking code in `codebases/` history.

---

## 0. Setup

Let `g_1,…,g_T` be a sequence in a real Hilbert space `H` (e.g. `R^d`, or `R^{m×n}` with the
Frobenius inner product). The EMA momentum buffer is

    m_t = β m_{t-1} + (1-β) g_t,    m_0 = 0,    β ∈ [0,1),

equivalently `m_t = (1-β) Σ_{u=1}^{t} β^{t-u} g_u`. On the window `t = 1,…,T` define the discrete
Fourier transform at `ω_ℓ = 2πℓ/T`, `ℓ = 0,…,T-1`,

    ĝ(ω) = Σ_{t=1}^{T} g_t e^{-iωt},   m̂(ω) = Σ_{t=1}^{T} m_t e^{-iωt},

and the EMA transfer function and its magnitude response

    H_β(ω) = (1-β)/(1 - β e^{-iω}),   |H_β(ω)| = (1-β)/√(1 - 2β cos ω + β²).

`|H_β|` is `2π`-periodic and even about `0` and `π`, and for `β>0` is **strictly decreasing on
[0,π]** (its denominator `1-2β cosω+β²` is strictly increasing there), with `|H_β(0)| = 1` and
`|H_β(π)| = (1-β)/(1+β)`; for `β=0`, `H_0 ≡ 1` and `m=g`.

**Folded frequency.** The grid `ω_ℓ = 2πℓ/T` runs over `[0,2π)`, but the `g_t` are real, so
`m̂(ω_{T-ℓ}) = conj(m̂(ω_ℓ))` and `|H_β(ω_{T-ℓ})| = |H_β(ω_ℓ)|`. We therefore index every band by
the **folded frequency** `f_ℓ := min(ω_ℓ, 2π-ω_ℓ) ∈ [0,π]` (equivalently the one-sided rFFT grid
used in `codebases/core/metrics.py`), on which `|H_β|` is monotone. Without folding, `|H_β|` rises
back to `1` as `ω_ℓ → 2π`, so a band must never be taken as a one-sided cut on the raw `[0,2π)`
grid.

---

## 1. Proposition 1 — exact windowed transfer identity  [✓ verified, err ≤ 1e-13]

For every grid frequency `ω_ℓ = 2πℓ/T`,

    m̂(ω_ℓ) = H_β(ω_ℓ) · ( ĝ(ω_ℓ) − e^{-iω_ℓ(T+1)} B ),    B := (β/(1-β)) m_T = Σ_{s=1}^{T} β^{T-s+1} g_s.

The boundary vector `B` is frequency-independent and equals `β/(1-β)` times the **terminal filter
state** `m_T`; it weights the most recent samples geometrically (`g_T` by `β`, `g_{T-1}` by `β²`, …).

**Proof.** Insert `m_t = (1-β) Σ_{u=1}^{t} β^{t-u} g_u` and exchange the order of summation:

    m̂(ω) = (1-β) Σ_{u=1}^{T} g_u e^{-iωu} Σ_{r=0}^{T-u} (β e^{-iω})^r
          = (1-β) Σ_{u=1}^{T} g_u e^{-iωu} · (1 - (β e^{-iω})^{T-u+1}) / (1 - β e^{-iω})
          = H_β(ω) [ ĝ(ω) − Σ_{u=1}^{T} g_u e^{-iωu} (β e^{-iω})^{T-u+1} ].

In the subtracted sum the `u`-dependent phase cancels: `e^{-iωu}(β e^{-iω})^{T-u+1} =
β^{T-u+1} e^{-iω(T+1)}`. Hence the sum equals `e^{-iω(T+1)} Σ_u β^{T-u+1} g_u = e^{-iω(T+1)} B`.
Using `Σ_u β^{T-u+1} g_u = β·(1-β)^{-1}·(1-β)Σ_u β^{T-u} g_u = β m_T/(1-β)` gives the stated form. ∎

The only departure from the ideal LTI relation `m̂ = H_β ĝ` is the single rank-one term
`−H_β(ω) e^{-iω(T+1)} B`: the finite window does not wrap around, and the correction is exactly the
filter's end-of-window state. This is the rigorous content of "boundary/transient error."

---

## 2. Theorem T1 — finite-window band filtering

Fix a continuous folded-frequency interval `I ⊆ [0,π]` and let `Ω` be the grid points with folded
frequency in `I` (§0). Define the band constants as **continuous suprema** over `I`,
`ρ_Ω(β) := sup_{ω∈I} |H_β(ω)|`; since `|H_β(ω_ℓ)| ≤ ρ_Ω(β)` at every grid point `ω_ℓ ∈ Ω`, the
bounds below control the grid-band norms. Write `‖x̂‖_Ω := (Σ_{ω∈Ω} |x̂(ω)|²)^{1/2}` and `|Ω|` for
the number of grid points in `Ω`.

**(a) High-band attenuation.** For any band `Ω`,

    ‖m̂‖_Ω ≤ ρ_Ω(β) · ‖ĝ‖_Ω + ρ_Ω(β) √|Ω| · |B|,    |B| = β |m_T| / (1-β).

In particular for `I = [ω_c, π]`, monotone decrease of `|H_β|` on `[0,π]` gives the closed form
`ρ_high(β) = |H_β(ω_c)|`; `ρ_high(β) < 1` for `β>0, ω_c>0`, it decreases in β, with the floor
`|H_β(π)| = (1-β)/(1+β)` at the Nyquist edge.

**(b) Low-band fidelity.** For `I = [0, ω_0]`, with `ε_low(β) := sup_{ω∈[0,ω_0]} |H_β(ω) − 1|`,

    ‖m̂ − ĝ‖_{Ω_low} ≤ ε_low(β) · ‖ĝ‖_{Ω_low} + √|Ω_low| · |B|.

**Proof.** By Proposition 1, `m̂(ω) = H_β(ω) ĝ(ω) − H_β(ω) e^{-iω(T+1)} B`. For (a), Minkowski's
inequality on `Ω` gives `‖m̂‖_Ω ≤ ‖H_β ĝ‖_Ω + ‖H_β e^{-i·(T+1)} B‖_Ω`. Bound each grid factor by
`ρ_Ω(β) = sup_I|H_β|`: `‖H_β ĝ‖_Ω ≤ ρ_Ω(β)‖ĝ‖_Ω` and `‖H_β e^{-i·(T+1)}B‖_Ω ≤ ρ_Ω(β)√|Ω|·|B|` since
`|B|` is constant over `Ω`. For (b), `m̂(ω)−ĝ(ω) = (H_β(ω)−1) ĝ(ω) − H_β(ω) e^{-iω(T+1)} B`;
Minkowski with `|H_β|≤1` and `|H_β−1|≤ε_low(β)` on `[0,ω_0]` gives the claim. ∎

**Closed forms for the constants.** `|H_β|` is monotone decreasing and `|H_β−1| = β·2 sin(ω/2)/
√(1−2β cos ω+β²)` monotone increasing on `[0,π]` [✓ verified], so the suprema sit at the interval
edges:

    ρ_high(β) = (1-β)/√(1 - 2β cos ω_c + β²),
    ε_low(β)  = β · 2 sin(ω_0/2) / √(1 - 2β cos ω_0 + β²)  ≈  β ω_0 / (1-β)   (small ω_0).

**`ε_low` increases in β while `ρ_high` decreases in β** — the filtering–lag tradeoff (T6), here with explicit constants in `β, ω_0, ω_c, T`. The
boundary term scales with the terminal state `|m_T|`, which is small once the filter has settled
(e.g. near a valley floor); it vanishes relative to the signal as the window lengthens whenever
`g`'s band energy is not concentrated in the last `O(1/(1-β))` samples.

**Remark (sample-size reading).** At Nyquist the attenuation `(1-β)/(1+β) = 1/N_eff` with
`N_eff = (1+β)/(1-β)`, matching the effective-averaging length used for the stochastic bound T2.

**Remark (river-alignment condition, T4).** Combining (a) and (b): write `L := ‖ĝ‖_{Ω_low}` (river
band energy) and `Hh := ‖ĝ‖_{Ω_high}` (hill band energy). Momentum improves the low-to-high energy
ratio, `‖m̂‖_{Ω_low}/‖m̂‖_{Ω_high} > L/Hh`, whenever

    (1 − ε_low(β)) L − √|Ω_low|·|B|  >  (L/Hh)·( ρ_high(β) Hh + ρ_high(β) √|Ω_high|·|B| ),

which in the **boundary-free / long-window limit** (`|B|` negligible, §2) reduces to the clean
condition `ρ_high(β) < 1 − ε_low(β)`: high-frequency attenuation must dominate low-frequency
distortion. By the closed forms this holds for an interval of `β` once `ω_c` is bounded away from
`ω_0`; it fails as `β → 1` (where `ε_low → 1`), explaining why maximal momentum is not optimal. The
boundary term `|B| = β|m_T|/(1-β)` only sharpens the threshold and is small once the filter has
settled.

**Empirical witness.** E4 measures `R(ω) = |m̂|/|ĝ|` on fixed streams and matches `|H_β(ω)|` to
relative error 0.001 (synthetic/curved) and ≤0.115 (straight, boundary-affected) — Proposition 1
with the boundary term made visible.

---

## 3. Theorem T3 — the straight valley separates river and hill in frequency

Take `L(x,y) = φ(x) + (λ/2) y²` with `φ(x) = (μ/2)(x − x*)²`, `0 < μ ≪ λ`, and gradient descent
`w_{t+1} = w_t − η ∇L(w_t)` (the `β=0` baseline). The coordinates decouple:

    y_{t+1} = (1 − ηλ) y_t,    x_{t+1} − x* = (1 − ημ)(x_t − x*).

Assume the **hill is in the oscillatory regime** `1 < ηλ < 2` and the **river is contractive**
`0 < ημ < 1`. Set `ρ := ηλ − 1 ∈ (0,1)`.

**(i) Hill is a damped Nyquist tone.** `y_t = (−ρ)^t y_0`, so `g^{hill}_t = λ y_t = λ y_0 (−1)^t ρ^t`.
For the length-`T` window transform `ĝ^{hill}(ω) = λ y_0 Σ_{t=1}^{T} (−ρ)^t e^{-iωt}`, the triangle
inequality gives `|ĝ^{hill}(ω)| ≤ λ|y_0| Σ_{t=1}^{T} ρ^t`, with **equality exactly at `ω = π`**,
where each term `(−ρ)^t e^{-iπt} = ρ^t` is real and positive. So for every nondegenerate window
(`T ≥ 2`, `y_0 ≠ 0`) the hill spectrum is maximized at the Nyquist frequency, uniquely for `ρ>0`
[✓ verified]. Its `T→∞` shape is `∝ 1/√(1 + 2ρ cos ω + ρ²)`, increasing on `[0,π]`.

**(ii) River is low-frequency.** `g^{river}_t = μ(x_t − x*) = μ(x_0−x*)(1−ημ)^t` with
`1−ημ ∈ (0,1)`. The coefficients are positive, so the same triangle inequality gives
`|ĝ^{river}(ω)| ≤ μ|x_0−x*| Σ_{t=1}^{T} (1−ημ)^t` with **equality exactly at `ω = 0`** for every
nondegenerate window: the river spectrum is maximized at DC. Its `T→∞` shape
`∝ 1/√(1 − 2(1−ημ) cos ω + (1−ημ)²)` is decreasing on `[0,π]`.

**(iii) Momentum separates them.** Applying the EMA and Proposition 1, the hill component is
attenuated by `|H_β(π)| = (1−β)/(1+β)` (up to the boundary term), while the river component near
`ω=0` is preserved (`|H_β(0)| = 1`). Hence the momentum river-to-hill energy ratio improves over the
raw ratio by a factor **up to** `((1+β)/(1−β))²`, attained in the pure-tone limit (`ρ→1` Nyquist
hill, DC river); a damped hill tone (`ρ<1`) has spectral spread and is attenuated less, giving a
smaller factor of the same order. [✓ verified: at `β=0.9` the realized factor is 99.5 against the
pure-tone bound 361; spectral peaks land at `ω/π = 0.000` (river) and `1.000` (hill).]

**Proof.** `y_t=(−ρ)^t y_0` and `x_t−x*=(1−ημ)^t(x_0−x*)` are the closed-form solutions of the two
scalar recursions; the exact finite-window argmax claims are the triangle-inequality arguments in
(i),(ii), and the `T→∞` magnitude shapes are the geometric sums `Σ_{t≥0} r^t e^{-iωt} =
(1−r e^{-iω})^{-1}`, whose moduli `(1−2r cos ω+r²)^{-1/2}` are monotone (numerator constant,
denominator monotone in `cos ω`). (iii) is Theorem T1 applied to a high band `I_high = [ω_c, π]`
containing the hill peak at `π` and a low band `I_low = [0, ω_0]` containing the river peak at `0`:
the hill is attenuated by `ρ_high(β) = |H_β(ω_c)|`, and since its mass concentrates at `π` the
realized attenuation approaches the Nyquist value `|H_β(π)| = (1−β)/(1+β)`, while the river near
`ω=0` is preserved up to `ε_low(β)`. ∎

**Curved valley (local approximation, as `idea_v1.md` permits).** For `L = φ(x) + (λ/2)(y−f(x))²`
the hill offset `δ_t := y_t − f(x_t)` obeys, to first order in the river step and for `|f''|`
bounded, `δ_{t+1} ≈ (1 − ηλ) δ_t + O(η ẋ f')`; the homogeneous part keeps the Nyquist oscillation
of (i), while the forcing `O(η ẋ f')` injects energy at the (low) river frequency. So the
separation of T3 persists locally provided the river curvature is mild relative to `1/(1−β)`; this
is exactly the lag regime E2 exhibits, where `β=0.99` begins to track the bend with error.

**Empirical witness.** E1 shows 72 hill-gradient sign changes at `β=0` and HSR/MSR tracking
`|H_β(π)|²`; E4(b) shows the straight-valley hill stream carries energy only at `ω ≳ 0.5π`.

---

## 4. Status against Minimal Success (idea_v1.md §5)

(1) toy demos show hill filtering — E1, E2; (2) frequency plots match `|H_β(ω)|` — E4;
(3) **finite-window filtering theorem — T1 above (Proposition 1 + Theorem T1)**; (4) bridge old
frequency paper → Muon — mechanism via E4→E5, theorem T5 (filter-first for the polar map) pending.
Remaining theory: T2 (stochastic, E3 already hits the `1/N_eff` floor), T4 (folded into the T1
remark), T5 (Wedin-type bound for `O(·)`), T6 (the explicit `ρ_high`/`ε_low` tradeoff above).
