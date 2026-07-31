"""Selection validity vs. measurement validity in router-generated LLM evaluation.

This package implements the controlled 2x2 validity experiment behind the TAE workshop paper:

  Factor A (assignment validity)   A0 common-population log / A1 selective-router log
  Factor D (measurement validity)  D0 construct-aligned evaluator / D1 stable construct-biased
                                   evaluator / D2 evaluator-stream mismatch (diagnostic)

Everything is an offline replay of the released LLMRouterBench performance-cost track; no new model
inference is performed. The *reference* scoring channel (the designated target outcome) is touched
only by the two labelled oracle conditions and by final scoring -- `validation.py` enforces that.
"""
