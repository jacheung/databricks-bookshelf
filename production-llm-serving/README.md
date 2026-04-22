# Production LLM Serving

This blueprint helps you size and deploy a production LLM serving architecture on Databricks. The goal is to serve your workload reliably at the right cost, using two levers: **Provisioned Throughput (PT)** and **Pay-Per-Token (PPT)**.

---

## The Two Levers

### Pay-Per-Token (PPT)
Shared infrastructure — you pay only for tokens consumed, with no capacity to manage. The tradeoff is separate hard ceilings on input and output throughput.

| | GPT OSS 120B |
|---|---|
| Input cost | $0.15 / 1M tokens |
| Output cost | $0.60 / 1M tokens |
| Max input throughput | 200,000 tokens/min |
| Max output throughput | 10,000 tokens/min |

Note the 20:1 ratio between input and output limits — this reflects the underlying compute asymmetry between prefill and decode, and is why "tokens per second" as a single number is a misleading capacity metric (see below).

### Provisioned Throughput (PT)
Dedicated inference capacity measured in **model units**, billed hourly regardless of usage. You get a guaranteed throughput floor and a burst scaling mechanism that can temporarily step up to the next model unit increment (e.g., 50 → 100 units) when capacity is available.

---

## Why "Tokens Per Second" Is the Wrong Metric

A single TPS number collapses two fundamentally different constraints — input throughput and output throughput — into one, hiding which one is actually binding for your workload.

**Prefill (input):** All input tokens are processed in a single parallel forward pass. Expensive due to O(n²) attention, but done in one step — the system can absorb a large volume of input tokens quickly.

**Decode (output):** Each output token is generated one at a time, autoregressively. Every new token must attend over the full KV cache — all input tokens plus all previously generated output tokens. This is sequential, compounds as generation length grows, and is far more resource-intensive per token than prefill.

The PPT limits make this concrete: 200,000 input tokens/min vs 10,000 output tokens/min — a 20:1 ratio. **Output throughput is almost always the binding constraint.** A workload generating long responses will hit the output ceiling long before it approaches the input ceiling.

Two workloads with identical total tokens behave very differently:

| Workload | Input | Output | Binding constraint |
|---|---|---|---|
| RAG / Q&A | 2,000 | 200 | Input ceiling (high volume, short answers) |
| Code / summarization | 200 | 2,000 | Output ceiling (long generations) |

**Size against your input and output limits separately, not a combined TPS number.**

---

## How to Size for Production

### Step 1 — Profile your workload on PPT first

Before committing to PT, run your workload on PPT and measure:
- Average input tokens per request
- Average output tokens per request
- Sustained input tokens/min and output tokens/min at typical load
- Peak input tokens/min and output tokens/min

Do not guess. Real distributions are rarely what you expect — and input/output ratios vary significantly by use case.

### Step 2 — Is PPT enough?

Compare your measured sustained load against the PPT ceilings **separately for input and output**. Either limit can be the binding constraint depending on your workload.

- **Both below ceiling:** Stay on PPT. No PT needed. Done.
- **Either at or above ceiling:** PT is required — not as a cost optimization, but because PPT physically cannot serve your load.

### Step 3 — Size your PT provisioning

Use the **[Databricks GenAI Pricing Calculator](https://www.databricks.com/product/pricing/genai-pricing-calculator)** to get a model unit recommendation for your workload shape. It takes:

- Cloud provider + region
- Model
- **Average input tokens** per request
- **Average output tokens** per request
- **Queries per minute (QPM)**

Enter your **P75–P90 load profile** — not your average, not your peak.

**Then validate empirically:**

1. Deploy at the recommended model unit level
2. Load test with traffic that matches your real input/output distribution
3. Observe where you start seeing 429s — that is your actual capacity ceiling
4. Adjust model units and repeat if needed

### Step 4 — Understand your full capacity stack

For a 50-unit PT endpoint, your capacity in descending order is:

```
┌─────────────────────────────────────────────────────┐
│  PPT fallback          ← overflow + 503/429 safety  │
├─────────────────────────────────────────────────────┤
│  PT burst (100 units)  ← best-effort, not guaranteed │
├─────────────────────────────────────────────────────┤
│  PT provisioned (50 units) ← guaranteed floor        │
└─────────────────────────────────────────────────────┘
```

- **PT provisioned** is your guaranteed floor — always available
- **PT burst** steps up one model unit increment automatically when capacity exists in the region — not guaranteed
- **PPT fallback** is your safety net when burst isn't available or isn't enough; this is why the blended approach matters even after you move to PT

---

## Architecture

```
Application
    │
    ▼
query_endpoint.py / query()
    │
    ├─► PT endpoint (provisioned + burst)   ◄── primary
    │
    └─► PPT (same model)                    ◄── fallback on 429 / 503
```

Requests always go to PT first. On capacity errors the client automatically retries against the PPT model. Both paths use the same OpenAI-compatible API and support streaming.

---

## Contents

| File | Purpose |
|---|---|
| `config.yml` | All configurable parameters — endpoint names, pricing inputs, deployment settings |
| `profile_workload.ipynb` | Step 1 — Profile your workload on PPT: measure input/output tokens, TTFT, TPOT |
| `capacity_deployment.ipynb` | Step 2 — Size (via pricing calculator) and deploy your PT endpoint |
| `query_endpoint.ipynb` | Step 3 — Query PT endpoint with PPT fallback |
