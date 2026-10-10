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
