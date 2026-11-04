<!-- markdownlint-disable MD033 MD022 MD032 MD041 MD023 MD003 -->

---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  :root {
    --bg1: #07122a;
    --bg2: #0d2043;
    --bg3: #12305f;
    --text: #eef4ff;
    --muted: #b8c8e6;
    --brand: #8ec5ff;
    --brand2: #7ff1cf;
    --line: rgba(190, 215, 255, 0.26);
  }

  section {
    font-family: "Manrope", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    color: var(--text);
    background:
      radial-gradient(900px 380px at 5% -10%, rgba(142, 197, 255, 0.25), transparent 60%),
      radial-gradient(820px 360px at 95% 110%, rgba(127, 241, 207, 0.16), transparent 60%),
      linear-gradient(145deg, var(--bg1), var(--bg2) 48%, var(--bg3));
    padding: 56px 72px;
    line-height: 1.5;
    font-size: 28px;
  }

  h1 {
    color: #f7fbff;
    font-size: 64px;
    margin-bottom: 10px;
    letter-spacing: -0.02em;
  }

  h2 {
    color: #f2f8ff;
    font-size: 46px;
    margin-bottom: 18px;
    letter-spacing: -0.01em;
  }

  h3 {
    color: #dff0ff;
    font-size: 34px;
    margin-bottom: 10px;
  }

  p, li {
    color: var(--text);
    font-size: 26px;
  }

  ul {
    margin-top: 10px;
  }

  strong {
    color: #ffffff;
  }

  code {
    background: rgba(255, 255, 255, 0.12);
    color: #ffffff;
    border: 1px solid var(--line);
    padding: 2px 8px;
    border-radius: 8px;
  }

  .muted {
    color: var(--muted);
    font-size: 22px;
  }

  .card {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px 18px;
    background: rgba(255, 255, 255, 0.07);
  }

  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }

  .three-col {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
  }

  .kpi {
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 10px 14px;
    background: rgba(255, 255, 255, 0.07);
    font-size: 22px;
  }

  .pill {
    display: inline-block;
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 18px;
    border: 1px solid rgba(142, 197, 255, 0.4);
    background: rgba(142, 197, 255, 0.15);
    color: #e6f2ff;
    margin-right: 8px;
  }

  .accent {
    color: var(--brand2);
  }

  .small {
    font-size: 20px;
    color: var(--muted);
  }

  .center {
    text-align: center;
  }

  .timeline li {
    margin-bottom: 8px;
  }
---

<!-- _class: center -->
# LAMP
## Lost Article Matching Platform

<span class="pill">Campus AI Product</span>
<span class="pill">Multimodal Matching</span>
<span class="pill">FastAPI + MySQL</span>

<p class="muted">Presentation length: 5-8 min</p>

---

## 1. Problem & Goal

<div class="two-col">
<div class="card">

### Problem
- Lost-and-found info is fragmented.
- Manual matching is slow and often inaccurate.
- Users miss timing-critical updates.

</div>
<div class="card">

### Goal
- Build a platform for <strong>faster matching</strong>.
- Support both <strong>finder</strong> and <strong>owner</strong> workflows.
- Trigger <strong>automatic notification</strong> when confidence is high.

</div>
</div>

<p class="small">Talk track: explain why this is a real campus pain point and why speed + confidence both matter.</p>

---

## 2. Product Overview

<div class="three-col">
<div class="card">

### Account
- Register + email verification
- Login with JWT
- Profile page

</div>
<div class="card">

### Report
- Submit found item
- Text + image embeddings
- Candidate notification

</div>
<div class="card">

### Search
- Chat-style input
- Image attachment
- Ranked match suggestions

</div>
</div>

<p class="small">Stack snapshot: FastAPI, MySQL, Jinja2, OpenAI-compatible API, ZhipuAI, multimodal ranking.</p>

---

## 3. End-to-End Flow

1. User logs in and submits query/report (text + optional image).
2. Backend generates embeddings and stores request/report vectors.
3. Coarse retrieval quickly narrows candidate set.
4. Fine ranking computes confidence and sorts final matches.
5. Frontend shows chat reply, confidence chips, and action modal.
6. For report flow, high-confidence pair can trigger email notification.

<p class="muted">Keywords: retrieval speed, confidence scoring, explainable output.</p>

---

## 4. Confidence Calculation (Search)

<div class="card">

### Stage A: Coarse Screening
- Compute similarities across 4 channels:
  - text→text
  - image→image
  - text→image
  - image→text
- Keep the max score as coarse score.
- Filter by threshold (`min_coarse_score`, default 0.2).

</div>

<div class="card" style="margin-top:12px;">

### Stage B: Fine Ranking
- Preferred: multimodal engine score (`final_score` from ranking module).
- Fallback: LLM label (`yes/maybe/no`) + weighted fusion.

</div>

---

## 5. Fine Ranking Logic (Multimodal)

<div class="two-col">
<div class="card">

### Feature Inputs
- user text embedding
- user image embedding
- candidate text embedding
- candidate image embedding (if exists)

</div>
<div class="card">

### Score Fusion
- text-text dot product
- image-text dot product (scaled)
- image-image dot product (optional)
- Final score = arithmetic mean of available scores

</div>
</div>

<p class="muted">Label mapping in current code: >=0.75 yes, >=0.45 maybe, else no.</p>

---

## 6. What Users Finally See

<div class="two-col">
<div class="card">

### Chat Reply
- Markdown assistant response
- Top candidates with confidence
- Actionable next-step suggestion

</div>
<div class="card">

### Confidence Signals
- `coarse pass`
- `refine attempted`
- `final kept`

</div>
</div>

<div class="card" style="margin-top:12px;">

### UX Action Modal
- If strong match found: show success modal + CTA button
- If unlikely match: show fallback modal + next-step CTA

</div>

---

## 7. Report-to-Owner Notification

- In report flow, backend finds best pending request candidate.
- Uses cross-field similarity and optional description exact-match boost.
- If score >= `MATCH_NOTIFY_THRESHOLD` (default 0.80), send email notification.
- Mark processed to avoid duplicate notifications.

<p class="small">This closes the loop: finder submits -> owner gets notified quickly.</p>

---

## 8. Engineering Decisions

<div class="three-col">
<div class="kpi">
<strong>Performance</strong><br/>
Two-stage retrieval keeps response practical.
</div>
<div class="kpi">
<strong>Robustness</strong><br/>
Multimodal path with fallback to LLM path.
</div>
<div class="kpi">
<strong>Usability</strong><br/>
Confidence chips + modal CTA improve decision speed.
</div>
</div>

<p class="muted">Trade-off: more model calls improve quality but increase latency/cost.</p>

---

## 9. Demo Script (1 min)

<ol class="timeline">
<li>Login to account.</li>
<li>Open Search Assistant and input item description.</li>
<li>Optionally attach image and send request.</li>
<li>Show returned candidate list and confidence chips.</li>
<li>Trigger modal and click CTA redirect.</li>
<li>Show report flow and explain email notification trigger.</li>
</ol>

<p class="small">Tip: prepare one positive case + one no-match case for contrast.</p>

---

<!-- _class: center -->
## 10. Roadmap & Closing

### Next Iterations
- Unify confidence thresholds across backend/frontend config.
- Add richer match explanation (why this item matched).
- Build reviewer/admin workflow for manual verification.
- Add monitoring dashboards for precision and turnaround time.

<p class="accent"><strong>LAMP turns fragmented lost-and-found signals into a fast, explainable recovery workflow.</strong></p>

<p class="muted">Q & A</p>
