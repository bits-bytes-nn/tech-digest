<div align="center">

# 🤖 주간 AI 기술 블로그 다이제스트

**선도적인 기술 블로그에서 그 주의 가장 좋은 AI/ML 엔지니어링 글을 모아 큐레이션·필터링·요약한 뒤 이메일로 보내 주는 자동화 뉴스레터.**

Amazon Bedrock (Claude) 기반 · AWS 위에서 CDK로 정의·오케스트레이션.

[![CI](https://github.com/bits-bytes-nn/tech-digest/actions/workflows/ci.yml/badge.svg)](https://github.com/bits-bytes-nn/tech-digest/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![AWS CDK](https://img.shields.io/badge/IaC-AWS%20CDK-orange)
![Bedrock](https://img.shields.io/badge/LLM-Amazon%20Bedrock%20(Claude)-green)

🇺🇸 [English README](./README.md)

![뉴스레터 미리보기](./app/assets/newsletter.png)

</div>

---

## ✨ 주요 기능

- **AI 기반 큐레이션** — Claude(Amazon Bedrock)가 글마다 관련성을 점수로 매기고,
  여러 섹션으로 구조화된 요약을 직접 작성합니다.
- **다중 소스 집계** — RSS와 견고한 HTML 스크레이핑으로 약 20개 기술 블로그(AWS,
  Google, Meta, OpenAI, Anthropic, NVIDIA 등)에서 글을 모읍니다. 모든 요청에는
  SSRF 가드와 소스별 헬스 추적이 따라붙습니다.
- **콘텐츠 품질 게이트** — 본문이 너무 빈약한 글은 LLM에 닿기 *전에* 걸러내, 빈
  껍데기 글이 다이제스트에 실리지 않게 합니다.
- **크롤 헬스 모니터링** — 모든 소스의 페치 상태를 추적하다가 소스가 실패하면 SNS
  알림을 띄워, 조용히 지나칠 뻔한 고장을 곧바로 드러냅니다.
- **서버리스 인프라** — 설정으로 고르는 AWS Lambda *또는* Batch 위에서 돌아가며,
  EventBridge로 스케줄링하고 AWS CDK로 코드화합니다.
- **완성도 높은 이메일** — 다크모드, 소스별 로고, 점수 배지를 갖춘 반응형 HTML
  템플릿을 Amazon SES로 전달합니다.

---

## 📐 문서

**AWS 아키텍처** — 인프라 & 데이터 흐름:

![AWS 아키텍처 다이어그램](./docs/diagrams/aws-architecture.png)

**처리 파이프라인** — 수집 → 전달:

![처리 파이프라인 다이어그램](./docs/diagrams/pipeline-flow.png)

---

## 🏗️ 아키텍처

### 핵심 구성 요소

| 모듈 | 책임 |
| --- | --- |
| `feed_parser.py` | RSS 파싱 + 견고한 HTML 스크레이핑(BeautifulSoup4 / Selenium), 소스별 헬스 추적 |
| `summarizer.py` | 콘텐츠 게이트 → 관련성 필터 → 랭킹/캡 → 요약까지, 모두 Bedrock으로 |
| `newsletter_renderer.py` | Jinja2 기반 HTML 생성(반응형, 다크모드, 점수 배지) |
| `aws_helpers.py` | S3, SES, SNS, SSM, Batch 연동 |

### 파이프라인

```
collect → gate → filter → rank → summarize → greet → render → deliver
```

### 인프라

- **Lambda / Batch** — `lambda_or_batch` 설정으로 실행 환경을 고릅니다.
- **EventBridge** — 정해진 일정에 실행합니다(기본: 토요일 01:00 UTC).
- **S3** — 설정, 수신자 목록, 생성된 뉴스레터, 글 HTML을 보관합니다.
- **SSM Parameter Store** — LangChain API 키와 Batch 큐/정의 이름을 저장합니다.
- **SES** — 뉴스레터를 전달합니다. **SNS** — 실행/헬스 알림을 보냅니다.
- **Bedrock(us-west-2)** — Claude Sonnet 5(필터 + 요약), Claude Haiku 4.5(인사말).

---

## 🛠️ 기술 스택

- **언어 / IaC:** Python 3.12+, AWS CDK, Docker
- **AI:** Amazon Bedrock, LangChain
- **스크레이핑:** Feedparser, BeautifulSoup4, Selenium
- **렌더링 / 설정:** Jinja2, Pydantic, YAML

---

## 📋 설정

`app/configs/config-{stage}.yaml`(예: `config-dev.yaml`)을 만드세요. 최상위 네 개
섹션은 [`app/configs/config.py`](./app/configs/config.py)의 Pydantic 모델에 그대로
대응합니다.

```yaml
resources:
  project_name: tech-digest
  stage: dev
  lambda_or_batch: batch
  cron_expression: "cron(0 1 ? * 6 *)"   # 토요일 01:00 UTC

scraping:
  min_content_length: 600                # 이보다 빈약한 글은 제외(가시 텍스트 문자 수)
  rss_urls:
    - "https://aws.amazon.com/blogs/amazon-ai/feed/"
    - "https://www.amazon.science/index.rss"

summarization:
  filtering_model_id: anthropic.claude-sonnet-5
  summarization_model_id: anthropic.claude-sonnet-5
  greeting_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
  min_score: 0.7                         # 이 점수 이상인 글만 유지
  max_posts: 5                           # 남길 글 수 상한(요약 전에 적용)

newsletter:
  sender: "your-verified-sender@example.com"
  header_title: "Weekly AI Tech Blog Digest"
```

> 모델 ID는 [`app/src/constants.py`](./app/src/constants.py)의 `LanguageModelId`
> 카탈로그에서 가져다 씁니다.

---

## 🚀 사용법

### 인프라 배포

```bash
python scripts/deploy_infra.py
```

### 로컬 실행

```bash
# 런타임 의존성 설치
pip install -r requirements.txt

# 환경 구성
cp .env.template .env        # 복사한 뒤 .env 편집

# 특정 주차의 다이제스트 생성·발송
python app/main.py --end-date 2026-06-03 --recipients you@example.com

# 또는 Batch 잡으로 제출
python app/run_batch.py --end-date 2026-06-03 --language ko --recipients you@example.com
```

### 테스트 & 품질 게이트

```bash
# 개발 도구 설치(ruff, mypy, pytest)
pip install -e ".[dev]"

# 린트, 포맷 검사, 타입 검사, 테스트
ruff check .
ruff format --check .
cd app && mypy .             # 듀얼 임포트 레이아웃을 해석하려면 app/에서 실행.
                             # `.`(=`src` 아님)로 main.py / run_batch.py도 검사
pytest                       # 빠른 오프라인 단위/통합 스위트(324개 테스트)
```

이 검사들은 모든 푸시와 풀 리퀘스트에서 CI로도 똑같이 실행됩니다
([`.github/workflows/ci.yml`](./.github/workflows/ci.yml)).

---

## 📄 라이선스

MIT
