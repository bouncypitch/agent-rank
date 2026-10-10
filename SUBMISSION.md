# AgentRank — AGI House "Living Internet" submission

## One line
AgentRank learns how AI answer engines choose what to recommend, rewrites a company's page honestly to win those recommendations without needing any traffic, and checks the result with a pre-registered test on a real engine.

## Paragraph
Buyers increasingly ask ChatGPT, Claude or Perplexity which product to buy, and the answer names only a few brands. Today's optimizers learn from a site's own traffic, which leaves low-traffic B2B companies out. AgentRank instead learns from what the engines already recommend.

We probed Claude Opus 5.5, OpenAI gpt-5.5 and Perplexity with 40 cobot buyer questions and built a cheap simulator from the pages each engine read. The simulator matched each engine's brand rankings with a rank correlation of 0.76–0.91. A contextual bandit then picked page edits, and a claim checker rejected any edit that invented facts.

We committed the hypothesis, the target companies and the analysis to git before the test ran. On a real engine with held-out questions, the test failed. Optimized pages were recommended 32.5% of the time, against 27.5% for the originals and 35% for a generic AI rewrite, and none of the differences were significant. The failure pinned down the cause: the real engine scores the same page anywhere from 25% to 50% between runs, and our per-edit scoring was too noisy to tell real gains from luck. That is the next thing to fix.

The work shipped is a validated simulator, an honest optimizer, and an evaluation setup that reports its own failures.
