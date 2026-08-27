# 3. Preliminaries

## 3.1 Team-competitive partially observable Markov game

We consider a two-team partially observable Markov game

\[
\mathcal{G}=\langle \mathcal{S},\mathcal{I}^{+},\mathcal{I}^{-},
\{\mathcal{O}_i\},\{\mathcal{A}_i\},P,R,\gamma\rangle,
\]

where `+` denotes the controlled team and `-` its opponent. At time `t`, active populations are `N_t=|I_t^+|` and `M_t=|I_t^-|`. Controlled agent `i` receives an entity-factored local observation

\[
o_i^t=(u_i^t,E_{i,+}^t,E_{i,-}^t),
\]

where `u_i^t` contains self features and each set contains relative state, team relation, availability, and optional type attributes. Entity number, visibility, and enumeration may vary. Controlled agents share policy parameters and execute without centralized observations; training may use CTDE.

Actions comprise fixed actions such as movement and entity-related actions such as attacking a visible opponent. The latter change in number with `|E_{i,-}^t|`.

## 3.2 Permutation invariance and equivariance

For an entity permutation matrix `P`, invariant fixed-action logits satisfy

\[
q_{\mathrm{fix}}(u_i,PE_i)=q_{\mathrm{fix}}(u_i,E_i),
\]

and entity-action logits satisfy equivariance

\[
q_{\mathrm{ent}}(u_i,PE_i)=Pq_{\mathrm{ent}}(u_i,E_i).
\]

An HPN-style baseline realizes these properties with shared entity-conditioned transformations, a symmetric masked reduction, and a shared scorer for entity targets [@hao2023hpn]. PI/PE guarantees structural consistency under reordering and variable set length; it does not guarantee performance at unseen scales or identify latent opponent strategy.

## 3.3 Composition and strategy

We distinguish two time-varying opponent variables.

The observable or estimable **composition** is

\[
c_t=(M_t,\nu_t,\eta_t,\kappa_t),
\]

where `M_t` is opponent count, `nu_t` a type distribution, `eta_t` a health/survival summary, and `kappa_t` a visibility or threat summary. A composition event occurs when these quantities change due to elimination, reinforcement, substitution, or visibility manipulation.

The latent **strategy** `p_t` parameterizes the opponent's state-conditional joint policy

\[
\pi_{p_t}^{-}(a_t^{-}\mid o_t^{-},c_t).
\]

A strategy event changes this conditional mapping, such as switching from nearest-target pursuit to focus fire, while a composition event can leave the mapping unchanged. This distinction is crucial: different compositions induce different state and action marginals even under the same conditional policy.

## 3.4 Composition--strategy confounding

Let a generic history encoder produce `z_t=f(tau_{0:t}^-)`. We call the representation **composition-confounded** over an intervention family when it changes sufficiently to indicate a strategy switch after an intervention on `c_t` that holds `p_t` fixed. Conversely, a representation that ignores all changes is unhelpful because it also misses interventions on `p_t`.

We operationalize the distinction with two binary event indicators:

- `C=1` if composition is intervened on at event time `t*`;
- `S=1` if the opponent controller/policy is switched at `t*`.

The four evaluation cells are `C0S0`, `C1S0`, `C0S1`, and `C1S1`. The primary false-positive metric is

\[
\mathrm{FPR}_{comp}=\Pr(d_t>\delta\text{ within the event window}\mid C=1,S=0),
\]

where `d_t` is a strategy-change score and `delta` is frozen on validation opponents. Switch sensitivity is measured by AUROC over `S`, detection delay, and post-switch decision regret.

## 3.5 Generalization setting

Training uses a set of compositions `C_train` and opponent policy implementations `Pi_train^-`; primary testing holds out both selected compositions and policy implementations. Parameters and thresholds are frozen at test time. We retain the scale-ID/OOD by opponent-ID/OOD performance matrix as a secondary outcome, but it is not sufficient to establish disentanglement because it does not isolate within-episode event causes.

## 3.6 Problem statement

Using trajectories from training compositions and policies, learn one shared team policy with two opponent codes: an instantaneous composition code that may respond to `c_t`, and a recurrent conditional-behavior code that is stable under supported composition interventions with fixed `p_t` but sensitive to supported policy interventions. The objective is to reduce `FPR_comp` and post-switch regret without materially degrading switch AUROC, in-distribution win rate, or linear entity-processing complexity.

