# Pre-registration — committed before any optimization or evaluation results exist

**Date:** 2026-10-10 (AGI House "Living Internet" build day)

## Hypothesis
For cobot / robot-arm company pages, edits selected by our simulator raise the rate at which real AI answer engines recommend the page, compared with three alternatives:
- (a) the original page,
- (b) a generic LLM rewrite ("improve this page"), not using our learned rules,
- (c) a placebo edit with the same length change and amount of edits, not using our learned rules.

## Primary metric
**Selection rate**: for a held-out buyer question, the engine is shown a candidate set (the target's version + real competitor pages, in randomized order, position-swapped across trials) and answers the question. A trial counts as a selection if the target company is named among the engine's top-3 recommended brands (order of first mention in the answer text, using `agentrank/mentions.py`).

## Evaluation design
- Questions: `data/queries.json` → `heldout` (15 questions), never used for learning or optimization.
- Engines: Claude (`claude-opus-5-5` + web fetch of the hosted versions), Perplexity (Agent API), OpenAI (`gpt-5.5`) if account credit allows. Each engine reported separately.
- Targets: 15–20 cobot companies chosen before optimization from `data/targets.json`.
- Versions per target: original / generic / placebo / optimized, hosted at neutral random slugs on the same domain with identical facts.
- Trials: ≥20 per engine per target per version.

## Analysis (fixed in advance)
- Per engine: selection rate per version with 95% Wilson confidence intervals.
- Primary comparisons: optimized − original, optimized − generic, optimized − placebo, as paired differences across targets, with 95% bootstrap CIs (10,000 resamples over targets).
- Holm correction across the 3 comparisons × number of engines.
- **Pass:** all three differences > 0 with Holm-corrected 95% CI excluding zero, for at least two engines.
- **Secondary:** rounds needed by the bandit optimizer to reach a +20-point simulated gain decreases across successive targets (Spearman correlation of rounds vs. target order < 0).
- **Simulator validity (go/no-go before optimizing):** Pearson r between simulated and real recommendation rates over the candidate-pool pages; report regardless of outcome.

## Honesty constraints on edits
No claims absent from the original page; no hidden text or instructions to models; the same content is served to humans and bots.

---

## Amendment 1 — committed before any final-test call (budget-driven, decided by the author)

Real Claude Opus web-search calls cost ~$0.16 each, which makes the original design (15–20 targets × 4 versions × ≥20 trials × 3 engines) unaffordable within the $40 cap. Changes:

- **Engine:** Claude Opus 5.5 only (`claude-opus-5-5`), web search **off**. It is given the held-out question plus a pool of real competitor pages and one version of the target page, and answers from them plus its own knowledge (system prompt = simulator v2 prompt). This tests the same "choose among sources" stage with the real production model; it does not test live retrieval.
- **Pools:** for each held-out question, the pages Perplexity actually retrieved when asked that question (1 Perplexity call per question; its answers are not used for learning). The target's own domain is removed from the pool; 9 competitor pages + the target version.
- **Targets:** the first 10 of `data/targets.json` (order locked in commit 1cd568d).
- **Questions / trials:** the first 8 held-out questions, one trial each → 8 trials per target per version, 320 trials total.
- **Pairing / position:** for a given (target, question) the target occupies the same randomly drawn position for all four versions; positions vary across questions.
- **Baselines:** generic = "Improve this web page."; placebo = sentence-by-sentence rephrase keeping facts, order and structure, length-matched to the optimized version. Both use the same editor model, honesty rules and claim checker as the optimizer.
- **Analysis:** `agentrank/analyze.py`, unchanged after this commit. **Pass (single engine):** optimized beats original, generic and placebo, each paired difference > 0 with Holm-corrected bootstrap CI excluding zero. Since only one engine is tested, a pass supports the hypothesis for Claude Opus 5.5 only.
- **Secondary** and **simulator validity** as before; simulator validity was already measured on training questions for all three engines (reported in the demo, including the weak per-question correlations).
