# router-audit report

- models 5, request types 8, logged rows 5,644
- supported pairs 37 of 40, never-observed pairs 0

## 1. Should you audit at all

- logged-count imbalance 1.22
- **verdict: audit**

  A balanced log is not worth auditing. On the study corpora, auditing a balanced
  log *raised* regret, so this verdict can legitimately be to keep your money.

## 2. What to buy next

| model | request type | top-two margin | traffic | audit price |
|---|---|---|---|---|
| deepseek-v3-0324 | type_2 | 0.0412 | 0.325 | $0.0075 |
| deepseek-v3-0324 | type_5 | 0.0248 | 0.193 | $0.0187 |
| gpt-5-chat | type_4 | 0.0153 | 0.058 | $0.0131 |
| gpt-5-chat | type_7 | 0.0836 | 0.070 | $0.0145 |
| qwen3-235b-a22b-thinking-2507 | type_2 | 0.0412 | 0.325 | $0.1392 |
| deepseek-v3-0324 | type_0 | 0.0913 | 0.131 | $0.0285 |
| qwen3-235b-a22b-thinking-2507 | type_4 | 0.0153 | 0.058 | $0.0775 |
| claude-sonnet-4 | type_5 | 0.0248 | 0.193 | $0.3534 |
| gemini-2.5-flash | type_1 | 0.1157 | 0.041 | $0.0221 |
| qwen3-235b-a22b-thinking-2507 | type_7 | 0.0836 | 0.070 | $0.0727 |

- total audit cost **$2.05** for 16 pairs at 30 readouts each

## 3. Money

- current policy $5.0503 per 1k requests
- recommended at a 0.05 tolerance $3.4969 per 1k
- gross savings at 1,000,000 requests **$1,553.33**
- net of the audit **$1,551.28**
- break-even **1,320 requests**

## 4. What this does not fix

A randomized audit repairs the assignment half of the problem only. If the
automatic score measures something other than the outcome the decision is about,
no audit budget removes that error, and a comparison corrected this way can still
name the wrong model. Buy gold labels for that half, not more reruns.

## 5. Warnings

  [W-SUPPORT] 3 model/request-type pairs have fewer than 8 rows and are excluded from every decision
  [W-VERSION] no model_version column, so results cannot be pinned to a model build