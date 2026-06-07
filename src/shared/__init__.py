"""Shared utilities between `deceptron` (real-repo pipeline) and `repro`
(Apollo reproduction harness): LLM call retry, trajectory formatting, and
AUROC/pAUROC metrics. Kept here because both sibling packages need the exact
same implementations for results to be comparable."""
