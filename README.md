<div align="center">

# 🤖 Weekly AI Tech Blog Digest

**An automated newsletter that curates, filters, summarizes, and emails the week's best AI/ML engineering posts.**

Powered by Amazon Bedrock (GPT-6.1 Sol filters, Claude summarizes) · orchestrated on AWS, defined with the CDK.

[![CI](https://github.com/bits-bytes-nn/tech-digest/actions/workflows/ci.yml/badge.svg)](https://github.com/bits-bytes-nn/tech-digest/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK-orange)
![Bedrock](https://img.shields.io/badge/LLM-Amazon%20Bedrock-green)

🇰🇷 [한국어 README](./README.ko.md)

![Newsletter Preview](./app/assets/newsletter.png)

</div>

---

## ✨ Features

- **AI-powered curation**. On Amazon Bedrock, GPT-6.1 Sol scores each post for
  relevance and Claude writes a structured, multi-section summary.
- **Multi-source aggregation**. pulls from \~20 tech blogs via RSS and resilient
  HTML scraping (AWS, Google, Meta, Microsoft, NVIDIA, and more), with
  SSRF-guarded requests and per-source health tracking.
- **Content quality gate**. drops posts whose visible text is too thin to
  summarize *before* they reach the LLM, so the digest never ships empty write-ups.
- **Skimmable cards**. every article leads with a one-sentence takeaway and a
  reading-time estimate, and the highest-scoring piece leads the issue.
- **Clip-budget aware**. mail clients truncate a message over \~100 KB. The
  summary length budget and the template's markup weight are both tuned to stay
  under it, and the build warns if an issue would be cut.
- **Crawl-health monitoring**. tracks every source's fetch status and alerts when
  one fails *or* stops offering anything to filter — a moved feed and a broken
  scraper look identical to a quiet blog otherwise, and most sources are
  legitimately empty in a normal week. A quiet blog never pages; an allow-list
  covers sources known to fail from AWS egress IPs, so the alarm stays actionable.
- **Serverless infrastructure**. AWS Lambda *or* Batch (config-selectable),
  scheduled by EventBridge, defined as code with the AWS CDK.
- **Professional email**. responsive HTML templates with dark-mode support,
  per-source logos, score badges, captioned figures and fully localized chrome
  (KO/EN), delivered through Amazon SES.
- **Idempotent delivery**. a per-issue ledger in S3 records which recipients an
  issue has reached, so a retried run — a Spot reclaim, an OOM — serves only the
  readers who were not yet served instead of delivering the whole issue twice.
- **Measured, not guessed**. two rubric harnesses steer tuning instead of taste:
  one scores the *filter* (does the right article get picked?) and one scores the
  *output* (is the write-up any good?). Every output check is deterministic and
  maps onto a specific prompt rule, so a low score says which instruction to fix.
- **Reproducible builds**. base images pinned by digest *and* every Python
  dependency, transitive ones included, pinned in `app/requirements.lock`, which CI applies as a
  constraints file so the suite tests exactly what the image ships.
- **Attributable cost**. on-demand Bedrock carries no taggable resource, so the
  pipeline invokes through tagged application inference profiles and logs token
  usage per pipeline stage.

---

## 📐 Documentation

**AWS architecture** — infrastructure & data flow:

![AWS architecture diagram](./docs/diagrams/aws-architecture.png)

**Processing pipeline** — ingestion → delivery:

![Processing pipeline diagram](./docs/diagrams/pipeline-flow.png)

---

## 🏗️ Architecture

### Core components

| Module | Responsibility |
| --- | --- |
| `feed_parser.py` | RSS parsing + HTML scraping (BeautifulSoup4), per-source health tracking |
| `summarizer.py` | Content gate → relevance filter → rank/cap → summarize, all via Bedrock |
| `model_factory.py` | Bedrock model capability registry + LangChain chat-model construction, cost-attribution profiles |
| `quality_metrics.py` | Deterministic output-quality rubric (pure functions) |
| `eval_metrics.py` | Filtering-score metrics (pure functions) |
| `newsletter_renderer.py` | HTML generation with Jinja2 (responsive, dark-mode, localized, size-checked) |
| `aws_helpers.py` | S3, SES, SNS, SSM, and Batch operations |

### Pipeline

```
collect → gate → filter → rank → summarize → greet → render → deliver
```

### Infrastructure

- **Lambda / Batch**. execution environment selected by `lambda_or_batch`.
- **EventBridge**. scheduled execution (template default: Fridays 01:00 UTC; prod runs Saturdays).
- **S3**. config, recipients, generated newsletters, and article HTML.
- **SSM Parameter Store**. LangChain API key and Batch queue/definition names.
- **SES**. newsletter delivery. **SNS** — run/health notifications.
- **Bedrock (us-west-2)**. GPT-6.1 Sol (filter), Claude Sonnet 5.5 (summarize), Claude
  Haiku 5.5 (greeting, title casing and output repair). Any model in the `LanguageModelId`
  catalog in [`app/src/constants.py`](./app/src/constants.py) is selectable per
  stage, including OpenAI GPT-6 models; per-model request rules (thinking
  off-switches, effort, the GPT dialect) are in design.md §8.

---

## 🛠️ Tech Stack

- **Language / IaC:** Python 3.12+, AWS CDK, Docker
- **AI:** Amazon Bedrock, LangChain
- **Scraping:** Feedparser, BeautifulSoup4
- **Rendering / config:** Jinja2, Pydantic, YAML

---

## 📋 Configuration

Create `app/configs/config-{stage}.yaml` (e.g. `config-dev.yaml`). The four
top-level sections map to the Pydantic models in
[`app/configs/config.py`](./app/configs/config.py):

```yaml
resources:
  project_name: tech-digest
  stage: dev
  lambda_or_batch: batch
  cron_expression: "cron(0 1 ? * 6 *)"   # Fridays 01:00 UTC (EventBridge: 1 = SUN)

scraping:
  min_content_length: 600                # drop posts thinner than this (visible chars)
  expected_flaky_urls: []                # URL substrings whose failures are reported, not alerted
  rss_urls:
    - "https://aws.amazon.com/blogs/amazon-ai/feed/"
    - "https://www.amazon.science/index.rss"

summarization:
  filtering_model_id: openai.gpt-6.1-sol
  summarization_model_id: anthropic.claude-sonnet-5-5
  greeting_model_id: anthropic.claude-haiku-5-5
  min_score: 0.7                         # keep posts scoring >= this
  max_posts: 5                           # cap kept posts (applied before summarizing)
  max_per_source: 2                      # optional diversity cap; unfilled slots are
                                         # backfilled, so it never shrinks the issue

newsletter:
  sender: "your-verified-sender@example.com"
  header_title: "Weekly AI Tech Blog Digest"
```

> Model IDs come from the `LanguageModelId` catalog in
> [`app/src/constants.py`](./app/src/constants.py).

---

## 🚀 Usage

### Deploy infrastructure

```bash
# deploy_infra.py is the CDK app; it only synthesizes. `cdk deploy` deploys it.
# --profile must reach the same account as resources.profile_name: the app
# resolves the account and writes the SSM secret with the config's profile.
CONFIG_FILE_SUFFIX=dev cdk deploy NewsletterDevStack --profile <aws-profile>
CONFIG_FILE_SUFFIX=prod cdk deploy NewsletterProdStack --profile <aws-profile>
python scripts/put_inference_profiles.py --stage dev   # once per account/stage
```

### Bedrock cost attribution

On-demand Bedrock usage carries no taggable resource, so
`scripts/put_inference_profiles.py` creates a tagged (`Project`/`Stage`) application
inference profile per configured model, and model resolution prefers it. A missing
profile falls back to the system-defined id. Each call also logs
`LLM usage stage=... model=... input=... output=...` per pipeline stage.

- `application-inference-profile` is a different IAM resource type from
  `inference-profile`; the policy must grant both, or every call is `AccessDenied`
  once a profile exists.
- Activate the `Project` cost allocation tag under **Billing → Cost allocation tags**
  (takes up to 24h, not retroactive).

See [design.md §8](./docs/design.md#8-bedrock-모델-팩토리-appsrcmodel_factorypy--배치-처리-appsrcutilspy) for details.

### Run locally

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Configure environment
cp .env.template .env        # then edit .env

# Generate and send a digest for a given week
python app/main.py --end-date 2026-06-03 --recipients you@example.com

# Or submit it as a Batch job
python app/run_batch.py --end-date 2026-06-03 --language ko --recipients you@example.com
```

`--recipients` sends only to the addresses given and neither reads nor writes the
delivery ledger. If none of them is a valid address, `run_batch.py` exits with an
error rather than falling back to the full recipients list.

### Test & quality gates

```bash
# Install dev tooling (ruff, mypy, pytest)
pip install -e ".[dev]"

# Lint, format-check, type-check, and test
ruff check .
ruff format --check .
cd app && mypy .             # run from app/ so the dual import layout resolves;
                             # `.` (not `src`) also checks main.py / run_batch.py
pytest                       # fast, offline unit/integration suite

# Output-quality rubric over a generated run. Scoring is deterministic, offline
# and free; --regenerate is the exception and makes real Bedrock calls.
# Each dimension maps onto one summarization-prompt rule, so a low score says
# which instruction to change. The overall score averages the graded dimensions;
# contract checks (structure, lede, captions...) are listed as violations. Two
# graded ones — translationese and cliche — score a rate rather than presence, since the phrasing they catch is
# ordinary Korean at low frequency. --compare diffs two runs to verify an edit.
python scripts/eval_summary_quality.py --date 2026-08-19
python scripts/eval_summary_quality.py --compare 2026-08-18 2026-08-19

# A/B a prompt edit on identical input: re-summarize a past run's stored articles
# with the current prompt and diff the rubric. COSTS MONEY (one call per article).
python scripts/eval_summary_quality.py --regenerate 2026-08-19

# Judge a filter-rubric edit on real candidates, without labels: freeze recent
# weeks' candidates, score them before and after the edit (COSTS MONEY), and list
# the posts whose line-up changed. The full recipe is in the script's docstring.
python scripts/replay_filter.py collect --stage prod --weeks 6
python scripts/replay_filter.py score --stage prod --label before   # then edit
python scripts/replay_filter.py score --stage prod --label after
python scripts/replay_filter.py compare --stage prod before after
```

CI runs the lint, format, type and test gates on pushes and pull requests to `main` ([`.github/workflows/ci.yml`](./.github/workflows/ci.yml)).
The output-quality rubric needs a generated run, so it is run by hand.

---

## 📄 License

MIT
