# 🤖 주간 AI 기술 블로그 다이제스트

> [English](./README.md) · **한국어**

선도적인 기술 블로그에서 그 주의 가장 좋은 AI/ML 엔지니어링 글을 큐레이션·필터링·
요약해 이메일로 전달하는 자동화 뉴스레터입니다. **Amazon Bedrock(Claude)** 기반으로
AWS 위에서 오케스트레이션됩니다.

![뉴스레터 미리보기](./app/assets/newsletter.png)

---

## ✨ 주요 기능

- **AI 기반 큐레이션** — Claude(Amazon Bedrock)가 각 글의 관련성을 점수화하고
  구조화된 다중 섹션 요약을 작성합니다.
- **다중 소스 집계** — RSS와 회복력 있는 HTML 스크레이핑으로 ~20개 기술 블로그(AWS,
  Google, Meta, OpenAI, Anthropic, NVIDIA 등)에서 글을 모으며, 요청은 SSRF 가드와
  소스별 헬스 추적을 갖춥니다.
- **콘텐츠 품질 게이트** — 가시 텍스트가 너무 빈약한 글을 LLM에 도달하기 *전에*
  제외해, 다이제스트가 빈 글을 싣지 않게 합니다.
- **크롤 헬스 모니터링** — 모든 소스의 페치 상태를 추적하고, 소스가 실패하면 SNS
  알림을 발행해 조용한 고장을 빠르게 표면화합니다.
- **서버리스 인프라** — AWS Lambda *또는* Batch(설정으로 선택), EventBridge로
  스케줄링하며 AWS CDK로 코드화합니다.
- **전문적 이메일** — 다크모드, 소스별 로고, 점수 배지를 갖춘 반응형 HTML 템플릿을
  Amazon SES로 전달합니다.

---

## 📐 문서

**AWS 아키텍처** — 인프라 & 데이터 흐름:

![AWS 아키텍처 다이어그램](./assets/diagrams/aws-architecture.png)

**처리 파이프라인** — 수집 → 전달:

![처리 파이프라인 다이어그램](./assets/diagrams/pipeline-flow.png)

---

## 🏗️ 아키텍처

### 핵심 구성 요소

| 모듈 | 책임 |
| --- | --- |
| `feed_parser.py` | RSS 파싱 + 회복력 있는 HTML 스크레이핑(BeautifulSoup4 / Selenium), 소스별 헬스 추적 |
| `summarizer.py` | 콘텐츠 게이트 → 관련성 필터 → 랭킹/캡 → 요약, 모두 Bedrock으로 |
| `newsletter_renderer.py` | Jinja2 기반 HTML 생성(반응형, 다크모드, 점수 배지) |
| `aws_helpers.py` | S3, SES, SNS, SSM, Batch 연동 |

### 파이프라인

```
collect → gate → filter → rank → summarize → greet → render → deliver
```

### 인프라

- **Lambda / Batch** — `lambda_or_batch`로 실행 환경 선택.
- **EventBridge** — 스케줄 실행(기본: 토요일 01:00 UTC).
- **S3** — 설정, 수신자, 생성된 뉴스레터, 글 HTML.
- **SSM Parameter Store** — LangChain API 키, Batch 큐/정의 이름.
- **SES** — 뉴스레터 전달. **SNS** — 실행/헬스 알림.
- **Bedrock(us-west-2)** — Claude Sonnet 4.6(필터 + 요약), Claude Haiku 4.5(인사말).

---

## 🛠️ 기술 스택

- **언어 / IaC:** Python 3.12+, AWS CDK, Docker
- **AI:** Amazon Bedrock, LangChain
- **스크레이핑:** Feedparser, BeautifulSoup4, Selenium
- **렌더링 / 설정:** Jinja2, Pydantic, YAML

---

## 📋 설정

`app/configs/config-{stage}.yaml`(예: `config-dev.yaml`)을 만드세요. 네 개의 최상위
섹션은 [`app/configs/config.py`](./app/configs/config.py)의 Pydantic 모델에
대응합니다.

```yaml
resources:
  project_name: tech-digest
  stage: dev
  lambda_or_batch: batch
  cron_expression: "cron(0 1 ? * 6 *)"   # 토요일 01:00 UTC

scraping:
  min_content_length: 600                # 이보다 빈약한 글은 제외(가시 문자 수)
  rss_urls:
    - "https://aws.amazon.com/blogs/amazon-ai/feed/"
    - "https://www.amazon.science/index.rss"

summarization:
  filtering_model_id: anthropic.claude-sonnet-4-6
  summarization_model_id: anthropic.claude-sonnet-4-6
  greeting_model_id: anthropic.claude-haiku-4-5-20251001-v1:0
  min_score: 0.7                         # 이 점수 이상인 글만 유지
  max_posts: 5                           # 유지 글 상한(요약 전에 적용)

newsletter:
  sender: "your-verified-sender@example.com"
  header_title: "Weekly AI Tech Blog Digest"
```

> 모델 ID는 [`app/src/constants.py`](./app/src/constants.py)의 `LanguageModelId`
> 카탈로그에서 가져옵니다.

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
cp .env.template .env        # 이후 .env 편집

# 특정 주차의 다이제스트 생성 및 발송
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
cd app && mypy src           # 듀얼 임포트 레이아웃 해석을 위해 app/에서 실행
pytest                       # 빠른 오프라인 단위/통합 스위트(240개 테스트)
```

이 검사들은 모든 푸시·풀 리퀘스트에서 CI로도 실행됩니다
([`.github/workflows/ci.yml`](./.github/workflows/ci.yml)).

---

## 📄 라이선스

MIT
