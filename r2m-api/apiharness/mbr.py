"""New direction: MBR (consensus) decoding of the answer step.

Every intervention we tried before this was *corpus-specific* — widen retrieval,
pass evidence, calibrate answer style — and each one won on the corpus whose
bottleneck it matched and lost on the other two. Routing between them failed at
both corpus and query granularity (E23-E30). So we stop trying to pick a
specialist and look for a lever that is uniform across corpora.

GAM and R²-Mem both take **one greedy sample** from the answer step. The metric
is token-F1. That is precisely the setting Minimum Bayes Risk decoding is for:
draw k candidates and return the one with the highest expected utility against
the others, using the evaluation metric itself as the utility,

    a* = argmax_{a in C}  (1/|C|) * sum_{b in C}  F1(a, b)

This is attractive here for three reasons the earlier attempts lacked:

1. **No assumption about the corpus.** It does not care whether golds are spans
   or paraphrases, whether retrieval is saturated, or how many pages there are.
   That is exactly what killed every previous intervention.
2. **It optimises the reported metric directly.** Token-F1 is the utility, so the
   selected answer is the one most likely to score well against an unseen
   reference drawn from the same distribution as the candidates.
3. **It is cheap and touches nothing upstream.** The search loop, memory and
   retrieval are untouched and fully cache-hit; only the final 256-token answer
   call is resampled k times.

Note the consensus answer is a *selection* among sampled candidates, not a
generated blend — so it can never hallucinate content absent from all samples.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, List, Optional, Sequence, Tuple


def _tokens(s: str) -> List[str]:
    return (s or "").lower().split()


def _f1(a: str, b: str) -> float:
    """Token-overlap F1, matching the shape of the evaluation metrics."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return float(ta == tb)
    ca, cb = Counter(ta), Counter(tb)
    common = sum((ca & cb).values())
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def mbr_select(
    candidates: Sequence[str],
    utility: Callable[[str, str], float] = _f1,
) -> Tuple[str, float, List[float]]:
    """Return (consensus answer, its mean utility, all mean utilities)."""
    cands = [c for c in candidates if (c or "").strip()]
    if not cands:
        return "", 0.0, []
    if len(cands) == 1:
        return cands[0], 1.0, [1.0]

    scores: List[float] = []
    for a in cands:
        # Exclude self-similarity so a lone outlier cannot win on its own vote.
        others = [b for j, b in enumerate(cands) if b is not a]
        scores.append(sum(utility(a, b) for b in others) / max(1, len(others)))
    best = int(max(range(len(cands)), key=lambda i: scores[i]))
    return cands[best], scores[best], scores


def make_mbr_answerer(
    base_answerer: Callable,
    n_samples: int = 8,
    temperature: float = 0.7,
    sampler_generator=None,
):
    """Wrap a DatasetSpec answerer with MBR decoding.

    `sampler_generator` must be a generator configured at `temperature` > 0;
    the baseline's own generator is typically at 0.3 and is reused if none is
    given. Diversity is what MBR feeds on, so a dedicated hotter sampler is
    preferable.
    """

    def answerer(category, summary, question, generator, evidence=None, **kw):
        gen = sampler_generator or generator
        cands: List[str] = []
        prev_salt = getattr(gen, "cache_salt", None)
        for i in range(n_samples):
            # Distinct cache key per draw, same prompt.
            if hasattr(gen, "cache_salt"):
                gen.cache_salt = i
            try:
                out = base_answerer(category, summary, question, gen,
                                    **({"evidence": evidence} if evidence else {}))
            except TypeError:
                out = base_answerer(category, summary, question, gen)
            if out:
                cands.append(out)
        if hasattr(gen, "cache_salt"):
            gen.cache_salt = prev_salt
        if not cands:
            return ""
        answer, _score, _all = mbr_select(cands)
        return answer

    return answerer


