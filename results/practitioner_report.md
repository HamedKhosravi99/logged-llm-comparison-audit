# router-audit report

- models 4, request types 3, logged rows 440
- supported pairs 12 of 12, never-observed pairs 0

## 1. Should you audit at all

- logged-count imbalance 0.19
- **verdict: probably keep your money**

  A balanced log is not worth auditing. On the study corpora, auditing a balanced
  log *raised* regret, so this verdict can legitimately be to keep your money.

## 2. What to buy next

| model | request type | top-two margin | traffic | audit price |
|---|---|---|---|---|
| gpt-4o-mini | airline | 0.0056 | 0.333 | $0.1202 |
| gpt-4o | airline | 0.0056 | 0.333 | $2.1554 |
| gpt-4.1-mini | retail | 0.1716 | 0.333 | $0.3590 |
| gpt-4.1-mini | telecom | 0.2923 | 0.333 | $1.0252 |
| gpt-4o | retail | 0.1716 | 0.333 | $2.4291 |
| gpt-4o | telecom | 0.2923 | 0.333 | $4.2437 |

- total audit cost **$10.33** for 6 pairs at 30 readouts each

## 3. Money


### Decision per request type

| request type | current model | recommended model | action |
|---|---|---|---|
| airline | gpt-4o-mini | gpt-4o-mini | retain |
| retail | gpt-4o | gpt-4o | retain |
| telecom | gpt-4.1-mini | gpt-4.1-mini | retain |

- expected quality, current 0.5924; recommended 0.5924
- request types requiring validated labels: airline
- request types with insufficient support: none
- current policy $39.7171 per 1k requests
- recommended at a 0.02 tolerance $39.7171 per 1k
- gross savings at 100,000 requests **$0.00**
- net of the audit **$-10.33**
- **no finite break-even**, the recommended policy is not cheaper

## 4. What this does not fix

A randomized audit repairs the assignment half of the problem only. If the
automatic score measures something other than the outcome the decision is about,
no audit budget removes that error, and a comparison corrected this way can still
name the wrong model. Buy gold labels for that half, not more reruns.

## 5. Warnings

  [W-VERSION] no model_version column, so results cannot be pinned to a model build