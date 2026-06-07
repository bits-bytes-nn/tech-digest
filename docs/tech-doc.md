# Tech Digest — 기술 문서

> **주간 AI 기술 블로그 다이제스트** 뉴스레터 서비스의 기술 레퍼런스입니다.
> 코드를 한 줄씩 따라가며 동작 원리를 설명하는, 이 프로젝트의 단일 심층 출처이며
> 코드와 함께 계속 갱신됩니다. 설치와 사용법 같은 상위 수준 안내는 루트
> `README.md`를 보세요.

**최종 갱신:** 2026-06-05

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [아키텍처 다이어그램](#2-아키텍처-다이어그램)
3. [저장소 구조](#3-저장소-구조)
4. [설정 (`app/configs/`)](#4-설정-appconfigs)
5. [상수 & Enum (`app/src/constants.py`)](#5-상수--enum-appsrcconstantspy)
6. [크롤링 & 파싱 (`app/src/feed_parser.py`)](#6-크롤링--파싱-appsrcfeed_parserpy)
7. [크롤러 헬스 추적](#7-크롤러-헬스-추적)
8. [Bedrock 모델 팩토리 & 배치 처리 (`app/src/utils.py`)](#8-bedrock-모델-팩토리--배치-처리-appsrcutilspy)
9. [프롬프트 & 프롬프트 캐싱 (`app/src/prompts/prompts.py`)](#9-프롬프트--프롬프트-캐싱-appsrcpromptspromptspy)
10. [필터링 & 요약 (`app/src/summarizer.py`)](#10-필터링--요약-appsrcsummarizerpy)
11. [콘텐츠 충분성 게이트](#11-콘텐츠-충분성-게이트)
12. [인사말 생성 (`app/src/greeter.py`)](#12-인사말-생성-appsrcgreeterpy)
13. [뉴스레터 렌더링 (`app/src/newsletter_renderer.py`)](#13-뉴스레터-렌더링-appsrcnewsletter_rendererpy)
14. [이메일 템플릿 (`app/templates/`)](#14-이메일-템플릿-apptemplates)
15. [AWS 헬퍼 (`app/src/aws_helpers.py`)](#15-aws-헬퍼-appsrcaws_helperspy)
16. [오케스트레이션 (`app/main.py`)](#16-오케스트레이션-appmainpy)
17. [배치 제출 (`app/run_batch.py`)](#17-배치-제출-apprun_batchpy)
18. [인프라 코드 (`scripts/deploy_infra.py`)](#18-인프라-코드-scriptsdeploy_infrapy)
19. [컨테이너 (`app/Dockerfile-*`)](#19-컨테이너-appdockerfile-)
20. [테스트 & CI/CD](#20-테스트--cicd)
21. [로컬 vs AWS 차이](#21-로컬-vs-aws-차이)
22. [운영 런북](#22-운영-런북)
23. [알려진 한계 & 향후 과제](#23-알려진-한계--향후-과제)

---

## 1. 시스템 개요

Tech Digest는 AI/ML 엔지니어링 블로그 글을 매주 자동으로 모아 큐레이션하고,
이메일 뉴스레터로 전달하는 서버리스 파이프라인입니다. 매주 한 번 **Amazon
EventBridge** 규칙이 정해진 시각에 파이프라인을 깨우면, 컨테이너로 패키징된
Python 애플리케이션이 — 설정에 따라 **AWS Lambda** 또는 **AWS Batch** 위에서 —
다음 여덟 단계를 순서대로 수행합니다.

1. **수집(Collect)** — 약 20개 소스에서 후보 글을 모읍니다. RSS 피드가 있으면
   그대로 쓰고, 피드가 없는 사이트는 전용 HTML 스크레이퍼로 긁어옵니다.
2. **게이트(Gate)** — 본문이 너무 빈약해서 제대로 된 요약이 나오기 어려운 글을
   미리 걸러냅니다.
3. **필터(Filter)** — LLM(Amazon Bedrock의 Claude)이 각 글을 읽고 ML 연구
   관련성과 품질을 점수로 매깁니다.
4. **랭킹(Rank)** — 점수 순으로 정렬해 상위 *N*개만 남깁니다.
5. **요약(Summarize)** — 살아남은 글마다 Claude를 한 번 더 호출해, 태그와 참고
   링크까지 갖춘 구조화된 HTML 설명을 만듭니다.
6. **인사말(Greet)** — 짧고 친근한 "Peccy" 인사말을 생성합니다.
7. **렌더(Render)** — Jinja2로 다크모드와 점수 배지를 갖춘 반응형 HTML 이메일을
   조립합니다.
8. **전달(Deliver)** — **Amazon SES**로 이메일을 발송하고, 산출물을 **S3**에
   보관하며, 실행 요약과 크롤 헬스 리포트를 **Amazon SNS**로 발행합니다.

설계의 첫 번째 원칙은 **우아한 성능 저하**(graceful degradation)입니다. 소스
하나, 글 하나, LLM 호출 하나가 실패하더라도 전체 실행이 멈추지 않습니다. 그렇다고
실패를 조용히 삼키지도 않습니다 — 무엇이 어떻게 실패했는지는 항상 SNS 알림으로
드러납니다.

---

## 2. 아키텍처 다이어그램

**AWS 아키텍처** (인프라 & 데이터 흐름):

![AWS 아키텍처 다이어그램](./diagrams/aws-architecture.png)

**처리 파이프라인** (수집 → 전달):

![처리 파이프라인 다이어그램](./diagrams/pipeline-flow.png)

AWS 다이어그램은 전체 그림을 보여줍니다 — EventBridge 트리거, Lambda냐 Batch냐를
가르는 컴퓨트 선택, 블로그 소스까지 나가기 위한 VPC/NAT 송신 경로, 그리고
Bedrock·S3·SSM·SES·SNS 연동까지. 파이프라인 다이어그램은 여덟 단계가 어떻게
이어지는지를 따라가며, 특히 **콘텐츠 게이트**("기사가 너무 짧아 요약 불가" 문제를
바로잡은 지점)와 **소스별 헬스 추적**을 강조합니다.

---

## 3. 저장소 구조

```
tech-digest/
├── app/
│   ├── main.py                  # 오케스트레이션 / Lambda+Batch 진입점
│   ├── run_batch.py             # Batch 잡 제출 및 대기 CLI
│   ├── Dockerfile-lambda        # Lambda 런타임 이미지 (브라우저 없음)
│   ├── Dockerfile-batch         # Batch 이미지 (Python 3.12 + Chrome)
│   ├── configs/
│   │   ├── config.py            # Pydantic 설정 모델 + YAML 로더
│   │   ├── config-template.yaml # 주석 달린 템플릿 (커밋됨)
│   │   ├── config-ci.yaml       # CI synth 검증 (Lambda 경로, 커밋됨)
│   │   ├── config-ci-batch.yaml # CI synth 검증 (Batch 경로, 커밋됨)
│   │   ├── config-dev.yaml      # 개발 설정 (gitignore)
│   │   └── config-prod.yaml     # 운영 설정 (gitignore)
│   ├── src/
│   │   ├── feed_parser.py       # 수집: RSS + 맞춤 스크레이퍼 + 헬스
│   │   ├── summarizer.py        # 콘텐츠 게이트 + LLM 필터 + LLM 요약
│   │   ├── greeter.py           # LLM 인사말 생성기
│   │   ├── newsletter_renderer.py # Jinja2 렌더 + Selenium HTML→이미지
│   │   ├── aws_helpers.py       # S3 / SES / SNS / SSM / Batch 헬퍼
│   │   ├── utils.py             # Bedrock 모델 팩토리, BatchProcessor, 파서
│   │   ├── constants.py         # Enum (모델, 경로, 환경변수, 소스)
│   │   ├── logger.py            # 로깅 + AWS 환경 감지
│   │   └── prompts/prompts.py   # 필터링 / 요약 / 인사말 프롬프트
│   ├── templates/               # Jinja2 HTML 이메일 파셜
│   └── assets/                  # 런타임 로고 + 수신자 파일
├── scripts/deploy_infra.py      # AWS CDK 스택 (Lambda/Batch + IAM + 스케줄)
├── tests/                       # pytest 스위트
├── docs/                        # 기술 문서 (이 파일 + 다이어그램)
├── pyproject.toml               # 도구 설정: ruff, mypy, pytest
└── .github/workflows/ci.yml     # CI: lint, type-check, test, cdk synth
```

> **임포트 경로 주의.** 이 프로젝트에는 두 가지 임포트 레이아웃이 공존합니다.
> 저장소 루트에서 실행할 때는 `app.configs` / `app.src`로 임포트하고(`config.py`,
> `deploy_infra.py`, 테스트가 이 경로를 씁니다), 컨테이너 안에서는
> `WORKDIR=/app`이라 `configs` / `src`로 임포트합니다(`main.py`, `run_batch.py`가
> 이 경로를 씁니다). 테스트 스위트는 `app.*` 한쪽으로 통일했고,
> `tests/conftest.py`가 저장소 루트를 `sys.path`에 추가해 이를 맞춥니다.

---

## 4. 설정 (`app/configs/`)

설정은 스테이지별 YAML 파일에서 읽어 들이는, 타입이 지정되고 검증되는 Pydantic
트리입니다. `config.py`는 다섯 개의 모델을 정의합니다.

### `BaseModelWithDefaults`
나머지 모델이 상속하는 작은 베이스 클래스입니다. `set_defaults_for_none_fields`
모델 밸리데이터를 달고 있어, 값이 명시적으로 `None`인 필드를 선언된 기본값으로
되돌립니다. 덕분에 YAML에서 키만 적고 값을 비워둬도(`profile_name:`) 합리적인
기본값을 얻습니다. 순수 Pydantic은 명시적 `None`에 대해서는 기본값을 채워주지
않기 때문에 직접 처리한 것입니다. 일반 기본값(`field.default`)뿐 아니라
리스트·딕트 필드의 **`default_factory`**(예: `rss_urls`·`logos`·`included_topics`)도
함께 다루므로, `logos: null` 같은 명시적 null이 `None`으로 새지 않고 `{}`나 `[]`로
복원됩니다.

### `Resources`
배포와 런타임의 정체성을 담습니다.
- `project_name`(필수, 비어 있으면 안 됨), `stage`(`dev` 또는 `prod`).
- `profile_name` — 로컬에서 쓸 AWS 프로파일(AWS 위에서는 무시됩니다).
- `default_region_name`(기본 `ap-northeast-2`)과 `bedrock_region_name`
  (기본 `us-west-2`). Bedrock은 스택의 나머지와 다른 리전에 두는 경우가 많아
  따로 둡니다.
- `s3_bucket_name`(필수)과 `s3_prefix`(선택적 키 프리픽스).
- `vpc_id` / `subnet_ids` — 선택. 둘 다 지정하면 기존 VPC를 가져다 쓰고, 아니면
  새로 만듭니다.
- `lambda_or_batch` — 컴퓨트 모드를 고릅니다.
- `cron_expression` — 엄격한 EventBridge cron 정규식으로 형식을 검증합니다.

### `Scraping`
- `rss_urls` — 소스 목록(피드 URL과 스크레이퍼 랜딩 페이지).
- `days_back`(기본 7, 1 이상) — 며칠 전까지의 글을 볼지 정하는 조회 기간.
- `min_content_length`(기본 600, 0 이상) — **콘텐츠 충분성 임계값**(자세한 내용은
  §11). 마크업을 걷어낸 가시 텍스트의 길이가 이 값보다 짧으면 요약 전에
  제외합니다. 기본값 600은 RSS 요약 스텁(보통 300~500자)과 실제 본문을 가르는
  경험적 경계입니다. 한 가지 주의할 점은 이 기준이 토큰이 아니라 *문자 수*라는
  것입니다. 그래서 같은 정보를 더 적은 글자로 표현하는 CJK(한국어·중국어·일본어)
  소스(kakao·ncsoft·qwen 등)는 동일한 임계값에서 더 공격적으로 걸러질 수
  있습니다. CJK 위주로 크롤링한다면 300~400 정도로 단계적으로 낮춰 보세요.

### `Summarization`
- `use_filtering` — LLM 관련성 필터를 켜고 끕니다.
- `filtering_criteria` — `all` 또는 `amazon`(어느 프롬프트 변형을 쓸지 고릅니다).
- `included_topics` / `excluded_topics` — 필터의 방향을 미세 조정합니다.
- `filtering_model_id` / `summarization_model_id` / `greeting_model_id` —
  단계별 Bedrock 모델 ID(Claude). 기본값은 차례로 Sonnet 4.6 / Sonnet 4.6 /
  Haiku 4.5입니다.
- `filtering_enable_thinking` / `summarization_enable_thinking` — 해당 단계에서
  확장 사고(extended thinking)를 켭니다.
- `max_concurrency`(기본 10) — Bedrock 요청의 동시성 한도.
- `min_score`(기본 0.7, 0~1) — 관련성 컷오프 점수.
- `max_posts`(선택) — 다이제스트에 실을 글 수의 상한.

### `Newsletter`
- `send_emails`, `save_articles`, `convert_to_images` — 출력 동작을 켜고 끄는
  토글.
- `sender`(검증된 발신 이메일), `header_*` / `footer_title` 문자열.
- `logos` — 소스명 → 로고 URL 매핑(템플릿에 주입됩니다).

### `Config.load()`
환경변수 `CONFIG_FILE_SUFFIX`(`dotenv`로 로드하며, 보통 스테이지명)를 읽어
`config.py` 옆에 있는 `config-{suffix}.yaml`을 찾습니다. 그런 다음 `from_yaml`이
그 파일을 파싱해 검증된 설정 트리를 만들어 냅니다.

---

## 5. 상수 & Enum (`app/src/constants.py`)

- **`AutoNamedEnum`** — `auto()`로 선언한 멤버가 자기 이름을 소문자 값으로 갖게
  하는 `str, Enum` 베이스입니다(`Language.KO.value == "ko"`).
- **`EnvVars`** — 앱이 읽는 모든 환경변수 이름을 한곳에 모았습니다(리전명, 설정
  suffix, LangChain 키, 로그 레벨, SNS 토픽 ARN).
- **`FilteringCriteria`** — `ALL` 또는 `AMAZON`. 필터링 프롬프트를 고를 때 씁니다.
- **`Language`** — `EN` 또는 `KO`. 요약과 인사말의 언어를 정합니다.
- **`LanguageModelId`** — Bedrock Claude 모델 ID 카탈로그입니다. 최신 모델(Sonnet
  4.6, Opus 4.6/4.7/4.8, Haiku 4.5)과 레거시 모델이 함께 들어 있으며, 현재
  활성 기본값은 Sonnet 4.6(필터·요약)과 Haiku 4.5(인사말)입니다. 새 모델을 추가할
  때는 여기와 `utils.py`의 `_LANGUAGE_MODEL_INFO` 양쪽에 함께 넣어야 합니다.
- **`LocalPaths`** — 디렉터리와 파일명 상수(inputs, outputs, logs, templates,
  recipients).
- **`SSMParams`** / **`S3Paths`** — SSM 파라미터 suffix와 S3 프리픽스.
- **`AppConstants`** — `NULL_STRING = "null"`(Batch가 "값 없음"을 전달할 때 쓰는
  센티넬입니다 — Batch 파라미터는 진짜 빈 값을 가질 수 없기 때문입니다), 그리고
  스크레이퍼 라우팅에 쓰는 URL 패턴 조각을 담은 `External` enum.

---

## 6. 크롤링 & 파싱 (`app/src/feed_parser.py`)

가장 크고 가장 방어적인 모듈입니다. URL 목록을 받아, 중복을 제거하고 날짜로
거른 `Post` 객체 리스트와 헬스 리포트로 바꿉니다.

### `ScraperConfig`
튜닝 노브를 한곳에 모은 클래스입니다(모두 `ClassVar`). 콘텐츠 CSS 셀렉터, 허용
날짜 포맷, 마크다운 이미지 정규식, 요청 타임아웃을 담고 있고, 핵심은 다음
둘입니다.
- **`REQUEST_HEADERS_OPTIONS`** — 현실적인 브라우저 헤더 세트 3종. 첫 번째는
  `Sec-Fetch-*`, `Sec-Ch-Ua`, `Upgrade-Insecure-Requests`까지 갖춘 완전한
  Chrome 131 프로파일입니다. `User-Agent`만 달린 요청을 거부하는 안티봇 필터
  (Meta, Medium 등)의 `403`을 실질적으로 줄여 줍니다.
- **`SOURCE_MAPPING`** — 도메인이나 핸들을 표준 소스명으로 매핑합니다. 일반
  도메인뿐 아니라 Medium 퍼블리케이션 슬러그와 `@`핸들
  (`netflix-techblog`, `palantir`, `pinterest_engineering`)도 포함해, Medium에
  호스팅된 피드가 올바른 로고로 연결되게 합니다.

### `HeaderCache`
도메인별로 "마지막에 성공한 헤더 세트 인덱스"를 기억하는 프로세스 전역
클래스 레벨 캐시입니다. 같은 호스트에 다시 요청할 때 시도 루프를 건너뛰어
바로 성공했던 헤더를 씁니다.

### `SourceFetchError`
소스를 **아예** 가져오지 못했을 때(네트워크 오류, 안티봇 차단, HTTP 오류)
발생합니다. "가져오기는 성공했지만 기간 안에 글이 없는" 경우와는 구별됩니다.
컬렉터는 이 구분을 근거로 소스를 `FAILED`와 `EMPTY`로 나눕니다(§7).

### `_try_request` / `_make_robust_request`
`_try_request`는 GET 한 번을 보내고, *일시적* 실패(타임아웃, `429`, `5xx`)에는
1회 재시도합니다. `_make_robust_request`는 캐시된(직전에 성공한) 헤더 세트를
먼저 시도하고 안 되면 나머지로 넘어갑니다. 성공하면 그 인덱스를 캐시하고 응답을
돌려주며, 모두 실패하면 `None`을 반환합니다. 예전에 여기저기 흩어져 중복되던
캐시 경로와 루프 로직을 하나로 합친 결과입니다.

**SSRF 가드** — 요청을 보내기 전 대상 호스트를, 그리고 리다이렉트가 끝난 뒤
*최종* 랜딩 호스트를 다시 `_is_blocked_host`로 검사합니다. 사설·루프백·링크로컬·
예약 IP 리터럴(특히 클라우드 메타데이터 엔드포인트 `169.254.169.254`)은
거부하고, 리다이렉트 횟수는 `MAX_REDIRECTS`(5)로 제한합니다. 크롤 대상은
어차피 설정 화이트리스트(`rss_urls`)지만, 그 소스가 내부 주소로 리다이렉트하는
상황을 막기 위한 장치입니다(IP 리터럴까지만 막으며, 호스트명 DNS 리바인딩까지는
방어하지 않습니다).

### `Post` (Pydantic 모델)
필드는 `title`, `link`, `published_date`, `content`, `images`, `source`,
`summary`, `tags`, `urls`, `score`입니다.
- `validate_tags`는 태그를 **처음 등장한 순서를 유지**한 채 중복을 제거하고
  상한을 적용합니다(`dict.fromkeys`). 모델이 관련도가 높은 순서대로 태그를
  나열하기 때문에, 알파벳순으로 정렬하면 중요한 태그가 단지 뒤로 밀렸다는 이유로
  잘려 나갈 수 있어 그렇게 하지 않습니다.
- `from_entry`는 `feedparser` 엔트리로부터 `Post`를 만듭니다. 콘텐츠를 뽑고,
  소스를 판정하고, 날짜를 파싱하고, 이미지를 모읍니다.
- **`_extract_content_from_entry`** — 피드에 인라인 콘텐츠가 있으면 그것을 쓰되,
  `ScraperConfig.MIN_CONTENT_LENGTH`(원시 HTML 3000자)보다 짧으면 본문 페이지를
  추가로 스크레이핑해 보강합니다. (헷갈리기 쉬운데, 이건 폴백 페치를 *촉발하는*
  원시 HTML 기준이고, 빈약한 글을 다이제스트에서 실제로 *제외하는* 가시 텍스트
  기준 `min_content_length` 게이트(§11)와는 별개입니다.)
- **`text_length()`** — 글의 **가시** 텍스트 길이를 돌려줍니다(BeautifulSoup로
  마크업 제거). 콘텐츠 게이트가 보는 값이 바로 이것입니다. 그래서 내비게이션·
  스크립트·스타일 같은 보일러플레이트만 많은 큰 페이지를 빈약하다고 올바르게
  판정할 수 있습니다.
- `_determine_source` — Medium을 먼저 특별 처리하고(퍼블리케이션 슬러그나 `@`
  핸들, 대소문자 무시) 그다음 도메인 조회로 넘어갑니다.
- `_extract_images` — `<img src>`(절대 URL로 변환)와 마크다운 이미지 URL을 모으며
  `http(s)`만 남깁니다.

### 날짜 헬퍼
- `try_parse_published_date` — **실패 시 막는(fail-closed)** 파서입니다. ISO-8601을
  먼저 시도하고, 안 되면 알려진 포맷 목록을 차례로 시도하며, 끝내 못 읽으면
  `None`을 돌려줍니다. 모든 날짜 윈도우 게이트(RSS·제너릭·사이트별 스크레이퍼)가
  이 함수를 씁니다. 그래서 날짜를 못 읽은 글은 "지금"으로 간주돼 매주 다이제스트에
  슬쩍 끼는 대신 *제외*됩니다.
- `parse_published_date` — `try_parse_published_date`를 감싸되, 실패하면 "지금"으로
  폴백하는 변형입니다. 정렬 키가 항상 있어야 하는 `Post` 구성에만 쓰고, 글을
  포함할지 말지 판단하는 데는 쓰지 않습니다.
- `is_date_in_range` — UTC로 정규화한 뒤 경곗값을 포함해 비교합니다.

### 페처 (Protocol 뒤의 Strategy 패턴)
`PostFetcher`는 `source_url`과 `fetch(start, end)`를 요구하는 `Protocol`입니다.
- **`RssFetcher`** — `force_ipv4()` 안에서 `feedparser`로 피드를 파싱합니다(일부
  호스트가 AWS에서 IPv6로 붙으면 오작동하기 때문). 엔트리가 하나도 없으면서 인코딩
  문제가 아닌 bozo 오류는 진짜 실패로 보고 `SourceFetchError`를 던집니다. 단순
  인코딩 경고만은 허용합니다.
- **`BasePageScraper`** — HTML 스크레이퍼의 베이스 클래스입니다. `_fetch_page`가
  랜딩 페이지를 가져와 파싱하고, 요청이 실패하면 **`SourceFetchError`를 던집니다**
  (차단된 스크레이퍼가 조용히 빈 결과로 끝나지 않고 제대로 보고되도록).
- **`GenericPageScraper`** — 템플릿 메서드 베이스입니다. 서브클래스가
  `ITEM_SELECTOR`를 설정하고 `_parse_item`만 구현하면 됩니다.
- 사이트별 스크레이퍼 — 제너릭 기반의 `GoogleBlogScraper`, `LinkedInBlogScraper`,
  `QwenBlogScraper`와, 구조가 빈약한 사이트를 위해 링크·날짜 휴리스틱을 직접 짠
  `AnthropicBlogScraper`, `MetaAIBlogScraper`, `XAIBlogScraper`가 있습니다.

### `ScraperRegistry`
URL 패턴 조각을 스크레이퍼 클래스에 매핑합니다. `get_fetcher`는 주어진 URL에 맞는
스크레이퍼를 돌려주거나, 맞는 게 없으면 `RssFetcher`로 폴백합니다.
`create_fetchers`는 설정된 URL들로부터 페처 목록을 구성합니다.

### `PostCollector`
`from_urls`가 페처들을 만들고, `collect_posts`가 각 페처를 실행합니다. 그 과정에서
링크로 중복을 제거하고, 소스별 헬스를 기록하고, 한 줄짜리 요약을 로깅한 뒤,
최신순으로 정렬된 글 목록을 반환합니다.

---

## 7. 크롤러 헬스 추적

"이 소스가 실제로 동작하고 있나?", "왜 로컬에서는 되는데 AWS에서는 안 되지?"
같은 운영 질문에 답하려고 추가한 기능입니다.

- **`SourceStatus`** — `OK`(가져와서 글까지 생성), `EMPTY`(가져왔지만 기간 안에
  글이 없음 — 오래된 피드일 수 있음), `FAILED`(가져오기 자체가 실패).
- **`SourceHealth`** — 소스별 레코드: `url`, `fetcher`, `status`, `post_count`,
  `error`.
- **`CrawlReport`** — `SourceHealth`들을 한데 모읍니다. `failed` / `empty` / `ok`로
  나눠 보여 주고, `total_posts`, 한 줄 요약 `summary_line()`
  (`"18 ok, 2 empty, 1 failed (34 posts)"`), 그리고 실패 소스와 오류를 사람이
  읽기 좋게 정리한 `format_alert()`를 제공합니다.

이 리포트는 `PostCollector.collect_posts`가 실행하면서 채웁니다. AWS에서
실행하다 소스가 하나라도 `FAILED`가 되면, `main.py`가 곧바로 전용 SNS 알림을
발행하고(`_send_crawl_health_alert`), 성공 알림에도 `summary_line()`을 함께
실어 보냅니다. 덕분에 깨진 소스(예: 데이터센터 IP에서만 발동하는 안티봇 차단 —
§21)가 다이제스트에서 조용히 사라지는 대신, 바로 손쓸 수 있는 이메일이 됩니다.

---

## 8. Bedrock 모델 팩토리 & 배치 처리 (`app/src/utils.py`)

### `LanguageModelInfo`와 `_LANGUAGE_MODEL_INFO`
모델별 역량 메타데이터 레지스트리입니다(컨텍스트 윈도우, 최대 출력 토큰, 프롬프트
캐싱·사고·성능 최적화·1M 컨텍스트 지원 여부). 팩토리는 이걸 보고 요청을 검증하고
기능을 안전하게 켭니다.

### `BaseBedrockModelFactory` (제너릭 ABC)
적절한 타임아웃, 적응형 재시도, 커넥션 풀을 갖춘 boto3 `bedrock-runtime`
클라이언트를 만듭니다. 서비스명, info dict, 실제 모델 생성 로직은 서브클래스가
채웁니다.

### `BedrockCrossRegionModelHelper`
가능하면 모델 ID를 **크로스 리전 추론 프로파일**로 해석합니다. 먼저 `global.*`
프로파일을, 그다음 리전 프로파일(`us.*`/`apac.*`)을 시도하고, 둘 다 없으면 순수
모델 ID로 폴백합니다. 이 덕분에 같은 설정을 여러 리전에서 그대로 돌릴 수 있습니다.
프로파일 목록은 `list_inference_profiles`로 조회합니다.

### `BedrockLanguageModelFactory`
`get_model`은 (가능하면 크로스 리전인) 모델 ID를 해석하고, `ChatBedrockConverse`
(크로스 리전이거나 사고를 켤 때)와 `ChatBedrock` 중 무엇을 쓸지 정한 뒤 설정을
구성합니다 — 온도(사고를 켜면 1.0으로 강제), 검증된 `max_tokens`, 선택적 1M
컨텍스트 베타 플래그, 선택적 성능 최적화 레이턴시 모드, 선택적 사고 예산까지.
Converse 경로와 비Converse 경로는 기능을 적용하는 방식이 달라서, 그 차이를 헬퍼
메서드가 안으로 감춥니다.

### `BatchProcessor` (Pydantic 모델)
동시 LLM 호출을 **배치 우선, 순차 폴백** 전략으로 돌립니다. 작업을 청크로 나눠
제한된 동시성으로 LangChain `.batch()`를 시도하고, 청크가 실패하면 항목별로
재시도(tenacity 지수 백오프)하는 방식으로 떨어집니다. 진행률은 `tqdm`로
표시합니다. 필터링과 요약 양쪽 모두에 깔리는 회복력 계층입니다. *(예전에 쓰지
않던 async 변형은 데드코드라 제거했습니다.)*

### `HTMLTagOutputParser`
모델이 내놓은 텍스트에서 이름이 붙은 태그를 뽑아내는 LangChain 출력 파서입니다.
태그명이 하나면 그 안의 텍스트를, 리스트면 "태그명 → 내부 HTML" 딕트를
돌려줍니다. `score`·`reason`·`summary`·`tags`·`urls` 같은 구조화된 필드를 모델
출력에서 꺼내는 방식입니다.

### `RetryableBase`
미리 구성해 둔 tenacity 재시도 데코레이터를 노출하는 믹스인입니다. `Greeter`가
씁니다.

### 기타 헬퍼
정규식 기반의 `validate_email(s)` / `validate_emails(list)`, 선택적 종료일과
조회 기간으로 날짜 윈도우를 계산하는 `get_date_range`, sync/async 양쪽을 재는
타이밍 데코레이터 `measure_execution_time`.

---

## 9. 프롬프트 & 프롬프트 캐싱 (`app/src/prompts/prompts.py`)

### `BasePrompt`
system/human 템플릿과, 선언된 입력/출력 변수를 담는 frozen 데이터클래스입니다.
`__post_init__`이 "선언한 입력 변수가 정말로 템플릿 안에 등장하는지"를
검증합니다. `get_prompt(enable_prompt_cache)`가 LangChain `ChatPromptTemplate`을
만들어 냅니다.

**프롬프트 캐싱(활성화됨).** 프롬프트 캐싱은 *프리픽스 매칭*으로 동작합니다 —
매번 바뀌는 `{post}`보다 **앞에 오는 부분만** 캐시됩니다. 그런데 원래의 거대한
정적 루빅스는 human 템플릿에서 `{post}` *뒤에* 있었고, 그래서 구조적으로 캐시가
불가능했습니다. 이를 다음과 같이 풀었습니다.

- 각 프롬프트는 `cache_split_marker`를 선언합니다(필터링은
  `**EVALUATION PROCESS:**`, 요약은 `**CORE PRINCIPLES:**`).
- `enable_prompt_cache=True`이고 마커가 있으면, `get_prompt`가 마커 이후의 정적
  지시문을 캐시되는 **system 프리픽스**로 옮기고, 매번 바뀌는 데이터(`{post}`)만
  human 메시지에 남깁니다. 이때 **텍스트를 복제하지 않습니다** — human 템플릿이
  단일 출처로 유지되고, 빌드 시점에 둘로 쪼개질 뿐입니다.
- 두 Bedrock 백엔드는 **서로 호환되지 않는** 캐시 마커를 쓰므로, `get_prompt`는
  실제로 실행될 백엔드에 맞는 **하나만** 내보냅니다(`use_converse` 플래그로
  분기). `ChatBedrockConverse`는 끝에 붙는 `{"cachePoint": {"type": "default"}}`
  블록을 받고, `ChatBedrock`은 텍스트 블록의 `cache_control: {ephemeral}`을
  받습니다. 주의할 점은, `cachePoint` 블록이 `ChatBedrock` 요청의 system
  콘텐츠에 들어가면 요청을 빌드하는 순간
  `ValueError("System message content item must be type 'text'")`로 죽는다는
  것입니다. 그래서 둘을 절대 동시에 넣지 않습니다. `Summarizer`는 실제로 생성된
  LLM 인스턴스 타입을 보고(`isinstance(..., ChatBedrockConverse)`) 알맞은
  플래그를 넘깁니다.
- 그 결과, 요약 system 프리픽스 약 1,350 토큰이 모든 글에 걸쳐 바이트 단위로
  동일하게 유지되어 캐시 히트가 납니다. 필터링 프리픽스는 루빅스를 단순화(과적합
  제거)한 뒤 약 1,020~1,030 토큰이 됐는데, 이는 Bedrock 최소 캐시 단위(약 1,024
  토큰) 경계에 아슬아슬하게 걸쳐 있어 필터링 캐시 히트는 보장되지 않습니다(요약
  캐시는 확실히 동작). 어차피 한 번 실행에 글을 몇 개밖에 처리하지 않는
  워크로드라, 캐싱 효과 자체가 본질적으로 제한적입니다.
- 캐싱은 모델의 `supports_prompt_caching` 역량에 따라 켜집니다(Summarizer가 모델
  정보를 보고 판단). 정적 프리픽스 안에 매번 바뀌는 값을 두면 호출마다 캐시가
  깨지므로, 토픽 목록처럼 한 번 실행 동안 안 바뀌는 변수는 마커 *앞*(human)에
  두고, 루빅스 안의 방향 표현("listed above" 등)은 위치에 의존하지 않도록
  고쳤습니다.

### `FilteringPrompt`
`for_criteria`로 선택하는 두 변형(`ALL`, `AMAZON`)이 있습니다. 토픽 관련성을
검증하고, `score`(0.00~1.00)·`reason`·정규화된 `title`을 이름 붙은 태그로 내도록
지시합니다.

**재현성을 위한 앵커 기반 점수제(튜닝됨).** 예전 루빅스는 수십 개의 가산 미세
모디파이어(±0.02/0.03)를 쌓아 올리는 방식이라 과적합돼 있었고, LLM이 그 산술을
일관되게 재현하지 못해 점수가 흔들렸습니다. 이를 **이산 앵커 점수**(0.05 단위
그리드: 0.85/0.80/0.70/0.60/0.50/0.35/0.15/0.05)에서 하나를 고른 뒤, 최대 ±0.05
한 단계만 조정하는 방식으로 단순화했습니다. 여기에 더해 dev/prod에서
`filtering_enable_thinking`를 **False**로 둬(확장 사고는 temperature=1.0을
강제하므로) temperature=0의 결정적 점수를 보장합니다. 요약은 사고를 그대로
유지합니다(점수와 무관하므로). `min_score`가 갖는 임계값의 의미는 변하지
않았습니다.

### `SummarizationPrompt`
`for_language`로 선택하는 `EN` / `KO` 변형이 있습니다. 다섯 섹션(왜 중요한가,
아키텍처, 기술 심층 분석, 성과, 향후)으로 구성된 사실 기반의 HTML 전용 요약과
함께 `tags`, 참고 `urls`를 요청합니다. **섹션 생략 규칙**을 둬서, 원문이 뒷받침하지
않는 섹션은 아예 빼게 했습니다 — "정보 없음" 같은 빈 섹션이 생기지 않도록.

### `GreetingPrompt`
"Peccy" 페르소나를 씁니다. `for_language`로 EN/KO를 고르고, 짧은(50~70 단어)
평문 인사말을 생성합니다.

---

## 10. 필터링 & 요약 (`app/src/summarizer.py`)

### `SummaryOutput` (Pydantic)
모델의 요약 출력을 검증합니다. 마크다운을 HTML로 변환하고(스타일용 클래스 주입
포함), **HTML을 새니타이즈**하며, 태그를 정규화하고(첫 등장 순서를 유지한 채
중복 제거·상한·기본값 적용), URL을 정규화합니다(마크다운·HTML 링크·평문 URL을
모두 이스케이프된 앵커로).
- **`_sanitize_html`** — 요약은 Jinja `|safe`로 렌더되는 **신뢰 경계**입니다.
  그래서 허용 목록(구조·인라인 태그)에 없는 태그는 언랩하고, `<script>`/`<style>`은
  제거하며, 이벤트 핸들러(`on*=`)나 `style` 같은 비허용 속성과
  `javascript:`/`data:` 스킴의 href·src를 걷어냅니다. 클래스 주입
  (`code`/`pre`/`table`) *뒤*에 돌려, 허용된 클래스는 살아남게 합니다.
- **`_normalize_urls` / `_is_safe_url`** — href 스킴을 `http`/`https`/`mailto`로
  화이트리스트하고 href와 텍스트를 모두 이스케이프합니다. 마크다운 링크, 기존
  `<a>` 태그, 평문 URL을 모두 똑같이 정규화하며, 허용되지 않는 스킴은 렌더하지
  않고 버립니다.

### `Summarizer`
- `__init__`에서 LangChain 체인 두 개를 만듭니다. **필터**
  (`prompt | llm | HTMLTagOutputParser`)와 **요약기**
  (`prompt | llm | OutputFixingParser(HTMLTagOutputParser)`). 요약기 쪽은 저렴한
  Haiku "수정" 모델로 잘못된 출력을 복구합니다. 두 체인 모두 모델이 지원하면
  **프롬프트 캐싱을 켭니다**(§9).
- **`process_posts`** — 공개 진입점이자 품질을 결정하는 단일 초크 포인트입니다.
  ① **콘텐츠 게이트**(§11) → ② 선택적 관련성 필터링 → ③ **점수 랭킹 +
  `max_posts` 캡** → ④ 요약 순서로 처리합니다. 캡을 **요약 직전**에 적용하는 것이
  핵심입니다 — 요약은 LLM 단계 중 가장 비싸므로, 어차피 버려질 글을 요약하느라
  Bedrock 비용을 낭비하지 않으려는 것입니다. 거친 앵커 점수 탓에 동점이 잦아서,
  2차 정렬 키로 게시일(최신 우선)을 써 결정적으로 동점을 가릅니다.
- **`_gate_by_content_length`** — 빈약한 글을 제외하되, 사유와 함께
  `filtered_out_posts`에 기록해 실행 알림에 드러나게 합니다.
- **`_filter_posts`** — 필터 체인을 배치로 실행하고, 각 글의 `score`/`reason`을
  파싱해 `min_score` 이상인 글만 남기며, 나머지는 제외 항목으로 기록합니다. 점수
  파싱이 실패한 글도 `filtered_out_posts`에 남겨 리포트에서 누락되지 않게 합니다
  (조용한 손실 방지).
- **`_summarize_posts`** — 요약 체인을 배치로 실행하고, 각 글에
  `summary`/`tags`/`urls`를 채운 뒤 결과를 `_postprocess_summary`로 넘깁니다.
- **`_postprocess_summary`** — 결정적인 수정 두 가지를 적용합니다. ① 한글 종결
  콜론을 마침표로 교정(`_KO_TERMINAL_COLON`) — 종결 음절 뒤 콜론이 **닫는 태그나
  문자열 끝 바로 앞**에 올 때만 고쳐서, 리스트를 여는 정상 콜론(`…같습니다: 첫째`)
  이나 비율(`3:1`)은 건드리지 않습니다. ② 소스 호스트 별칭 정규화
  (`_HOST_ALIASES`). 오타 교정 휴리스틱과 호스트 별칭이라는 서로 다른 의도를
  명시적으로 분리했습니다. (요약 프롬프트에도 "문장을 콜론으로 끝내지 말 것"을
  지시해 원천에서부터 줄입니다.)

---

## 11. 콘텐츠 충분성 게이트

**문제.** 가끔 요약기까지 도달한 "글"이 정작 읽을 만한 본문은 거의 없는 경우가
있었습니다(티저, 링크만 있는 항목, 대부분이 보일러플레이트인 페이지). 그러면
요약기가 실패하거나 거의 빈 요약을 내놓았습니다 — "기사가 너무 짧아 글을 못 씀"
증상이 바로 이것입니다.

**해결.** `Summarizer.process_posts`의 맨 앞에 게이트를 둡니다.

1. `Post.text_length()`로 HTML을 걷어내고 **가시** 문자 수를 잽니다.
2. `scraping.min_content_length`(기본 **600**)보다 짧은 글을 제외합니다.
3. 제외한 글은 각각 사유와 함께 `filtered_out_posts`에 기록해 SNS 실행 요약에
   보고합니다(조용히 사라지지 않도록).

게이트가 필터와 요약기보다 *앞*에 있으므로, 필터링을 켰든 껐든 상관없이 빈약한
글은 LLM에 도달하지 못합니다. 임계값은 스테이지별로 설정할 수 있습니다. 이
로직은 `tests/test_content_gate.py`로 검증합니다.

---

## 12. 인사말 생성 (`app/src/greeter.py`)

`Greeter`는 설정된 언어로 `prompt | llm | StrOutputParser` 체인을 만들고
`greet(context)`를 노출하며, `RetryableBase`의 재시도 데코레이터로 감쌉니다.
인사말 모델은 보통 저렴한 Haiku 티어를 씁니다.

---

## 13. 뉴스레터 렌더링 (`app/src/newsletter_renderer.py`)

### 모델
`Article`, `Header`, `Section`, `Footer`, `NewsletterData` — 타입이 지정된 뷰
모델입니다. `Article`은 날짜를 검증하고 점수를 0~1로 제한하며, 접근성 alt
텍스트에 쓰는 `source_label` 프로퍼티를 노출합니다. `validate_date`는 다양한 입력
날짜 포맷을 `YYYY-MM-DD`로 강제 변환합니다.

### `NewsletterRenderer`
**autoescape를 켠** Jinja2 `Environment`를 감쌉니다(요약 HTML만 의도적으로 `|safe`
필터로 렌더합니다). `|safe`로 빠져나가는 요약과 URL은 업스트림 `SummaryOutput`에서
이미 새니타이즈와 스킴 검증을 거쳤으므로(§10의 `_sanitize_html`/`_normalize_urls`),
이 신뢰 경계는 모델이나 원문이 주입한 스크립트, `javascript:` 링크로부터
보호됩니다. 전체 뉴스레터나 단일 글을 렌더할 수 있습니다.

### `HtmlToImageConverter`
HTML을 스크린샷으로 찍는 선택적 Selenium/Chrome 렌더러입니다(단일/분할 페이지).
이제 Chrome/Chromium 바이너리가 없으면(`chrome_available()`) 모호한 WebDriver
오류 대신 바로 손쓸 수 있는 메시지와 함께 **즉시 실패**합니다 — Lambda에
브라우저가 없는 경우(§21)를 위한 가드입니다. `PATH`나 `CHROMEDRIVER_PATH`의
`chromedriver`를 우선하고 `CHROME_BINARY`를 존중하며, `webdriver-manager`
폴백은 로컬 개발에서만 씁니다.

### `NewsletterBuilder`
날짜별 글 JSON을 로드하고 소스별 로고를 붙인 뒤, 뉴스레터 HTML(과 선택적으로
글별 HTML/이미지)을 렌더해 출력 경로를 돌려줍니다. 동작은 `BuildConfiguration`이
결정합니다.

---

## 14. 이메일 템플릿 (`app/templates/`)

이메일 클라이언트 호환성을 최대로 끌어올리려고 테이블 기반 HTML을 쓰며, Jinja2
파셜을 조립합니다: `template.html`(셸 + CSS), `header_section.html`,
`first_section.html`(인사말), `main_contents.html`(글 카드),
`footer_section.html`, `article.html`(독립 글).

사용성·디자인 측면의 기능은 다음과 같습니다.
- **다크모드** — `@media (prefers-color-scheme: dark)`와 `color-scheme` 메타
  태그를 씁니다. 카드·텍스트·칩·코드 블록이 모두 색을 반전합니다.
- **관련도 점수 배지** — 날짜 옆에 알약 모양(`★ 0.84`)으로 렌더합니다. 그라디언트를
  벗겨 버리는 클라이언트(Outlook 등)를 위해 단색 `background-color` 폴백을
  넣었습니다.
- **접근성** — `<html>`의 `lang`, 이메일 본문의 `role="article"`과 `aria-label`,
  그리고 소스명에서 만든 의미 있는 이미지 alt 텍스트.
- **모바일** — `@media (max-width: 800px)`로 제목과 패딩을 줄입니다. 코드 블록은
  넘치지 않고 스크롤되거나 줄바꿈됩니다.
- **프리헤더** — 받은편지함 미리보기 줄에 인사말 인트로를 씁니다(없으면 제목).

---

## 15. AWS 헬퍼 (`app/src/aws_helpers.py`)

boto3 위에 얹은, 얇고 로깅이 잘 된 래퍼들입니다. 우아한 성능 저하에 도움이 되는
지점에서는 예외를 호출자에게 던지지 않고, 명확한 성공/실패 신호를 돌려줍니다.
- `check_and_download_from_s3`, `upload_to_s3` — S3 객체 I/O.
- `get_account_id`, `get_ssm_param_value` — STS/SSM 조회.
- `send_email` — 제대로 된 MIME 멀티파트 메시지와 생성된 `Message-ID`를 갖춘 SES
  `send_raw_email`.
- `submit_batch_job`, `wait_for_batch_job_completion` — Batch 잡 제출과 폴링.

---

## 16. 오케스트레이션 (`app/main.py`)

`handler(event, context)`가 Lambda와 Batch 양쪽의 단일 진입점입니다. 흐름은
다음과 같습니다.

1. 설정을 로드합니다. 기본 리전 boto 세션과 Bedrock 리전 boto 세션을 구성합니다
   (로컬에서는 `profile_name`을 반영).
2. `_setup_aws_env` — AWS에서는 LangChain API 키를 SSM에서 채워 넣습니다.
3. **`_fetch_and_filter_posts`** — 글을 수집한 뒤 `Summarizer.process_posts`에
   위임합니다(게이트 → 필터 → 랭킹/`max_posts` 캡 → 요약). 글 목록, 날짜 suffix,
   제외 목록, 그리고 **`CrawlReport`** 를 돌려줍니다.
4. AWS 실행 중 소스가 실패했다면 크롤 헬스 알림을 발행합니다.
5. 살아남은 글이 하나도 없으면 200으로 우아하게 종료합니다.
6. **`_process_posts_and_create_newsletter`** — 글 JSON을 영속화하고, 인사말을
   생성하고, HTML을 빌드하고, 산출물을 S3에 업로드합니다.
7. 이메일이 켜져 있으면 수신자를 해석하고(CLI 인자 또는 S3 파일), 수신자마다 약간의
   간격을 두고 발송한 뒤, 성공 알림을 발행합니다(크롤 요약과 제외 글 포함).
8. 처리되지 않은 예외가 나면 실패 알림을 발행하고 500을 반환합니다.

언어(`Language`)는 `handler` 진입부에서 **딱 한 번** 해석해 필터링·인사말·뉴스레터
빌드·파일명까지 똑같이 흘려보냅니다. 이렇게 해야 `--language en`인데 요약은
영어로 나오면서 인사말만 KO 기본값으로 남는 불일치가 생기지 않습니다.

`__main__` 블록은 `--end-date`, `--language`, `--recipients`를 파싱하고
(`"null"` 센티넬은 "값 없음"으로 취급) `handler`를 호출합니다.

---

## 17. 배치 제출 (`app/run_batch.py`)

SSM에서 Batch 잡 큐/정의 이름을 조회하고, 파라미터를 정제한 뒤(`None`이나 공백은
`"null"` 센티넬로, 수신자는 콤마로 결합) 잡을 제출하고, 완료될 때까지 폴링하는
로컬 CLI입니다. 스케줄을 기다리지 않고 온디맨드로 실행을 트리거할 때 씁니다.

---

## 18. 인프라 코드 (`scripts/deploy_infra.py`)

설정으로 완전히 파라미터화되는 단일 AWS CDK `NewsletterStack`입니다.

- **VPC** — `vpc_id`/`subnet_ids`가 주어지면 기존 VPC를 가져오고, 아니면 퍼블릭
  서브넷과 `PRIVATE_WITH_EGRESS` 서브넷, 그리고 NAT 게이트웨이 1개를 갖춘 2-AZ
  VPC를 새로 만듭니다(블로그 소스까지 나가기 위한 송신 경로).
- **IAM(최소 권한)** — 광범위한 `*FullAccess` 매니지드 정책을 붙이는 대신, 역할에
  앱이 실제로 하는 일에 딱 맞춘 **범위 지정 인라인 정책**을 부여합니다: 설정 버킷의
  S3 객체/리스트 작업, **설정된 발신자로 제한된**(`ses:FromAddress` 조건) SES
  발송, 이 스택 토픽으로의 SNS publish, 이 프로젝트 파라미터 경로의 SSM 읽기와
  (SecureString 복호화를 위한, `kms:ViaService=ssm`으로 제한된) `kms:Decrypt`,
  해당 리전의 **Anthropic 파운데이션 모델 + 추론 프로파일 ARN으로 범위 지정된**
  Bedrock invoke(프로파일 탐색만은 계정 레벨로 유지), 프로젝트 Batch 로그 그룹으로
  범위 지정된 CloudWatch Logs. 끝까지 매니지드 정책으로 남겨 둔 건 런타임 서비스
  실행 역할(Lambda basic/VPC exec, 또는 Batch용 ECS-on-EC2 인스턴스 역할)뿐입니다.
  `s3_bucket_name`이 비어 있으면 (S3를 `*`로 범위 지정하는 대신) 스택이
  **fail-closed**로 실패합니다. AWS Well-Architected 보안 기둥을 의식한 개선입니다.
- **SNS** — 토픽을 만들고, 선택적으로 운영자 이메일을 구독시킵니다.
- **컴퓨트** — `DockerImageFunction`(Lambda) 또는 EC2 온디맨드 + 스팟 컴퓨트
  환경의 Batch 잡 정의 중 하나입니다. 어느 쪽이든 EventBridge 스케줄에 연결됩니다.
- **SSM** — Batch 큐/정의 이름은 CDK `StringParameter`로, LangChain API 키는
  **SecureString**으로 저장합니다. CloudFormation은 SecureString을 만들지 못하고
  평문 `StringParameter`는 키를 synth 템플릿에 노출시키므로, 키만은 배포 시점에
  `boto3`로 따로 기록합니다(런타임은 `WithDecryption=True`로 읽습니다).

---

## 19. 컨테이너 (`app/Dockerfile-*`)

- **`Dockerfile-lambda`** — `public.ecr.aws/lambda/python:3.12-x86_64` 기반.
  **브라우저가 없어서** 이미지 변환은 여기서 지원하지 않습니다(파일에 명시했고,
  §13의 fail-fast 가드로 강제합니다). Lambda 런타임이 비루트로 실행하므로 별도
  `USER` 지시는 없습니다.
- **`Dockerfile-batch`** — `ubuntu:24.04`(Python 3.12)에 `google-chrome-stable`,
  한국어/CJK/이모지 폰트, `xvfb`를 설치합니다. `CHROME_BINARY`를 고정해, Selenium이
  불안정한 네트워크 다운로드 없이 맞는 드라이버를 찾게 합니다. `convert_to_images`를
  켰다면 Batch를 씁니다. **비루트 사용자**(`appuser`)로 실행하며(Chrome은 이미
  `--no-sandbox`/`--disable-dev-shm-usage`로 띄우므로 호환), 런타임 쓰기는 `/app`이
  아니라 `/tmp`(`ROOT_DIR`)로 갑니다.
- 두 이미지의 베이스는 **digest로 고정**해 재현 가능한 빌드를 보장합니다(태그와
  digest를 함께 갱신).

---

## 20. 테스트 & CI/CD

- **`pyproject.toml`** 이 도구 설정을 한곳에 모읍니다: `pytest`(`unit`/
  `integration`/`live` 마커와 importlib 임포트 모드), `ruff`(lint + format),
  `mypy`.
- **`tests/`** — 빠르고 네트워크를 타지 않는 pytest 스위트입니다(240개 테스트).
  커버하는 범위는 다음과 같습니다.
  - 날짜 파싱(fail-closed `try_parse_published_date` 포함), 소스/이미지 파싱.
  - **사이트별 스크레이퍼 골든 테스트** — `tests/fixtures/scrapers/`에 저장해 둔
    HTML로 Google·LinkedIn·Qwen·Meta·xAI·Anthropic의 제목/링크/날짜 윈도우/내비
    필터를 오프라인에서 고정.
  - 요청 계층(모킹 세션으로 헤더 로테이션·일시 재시도·도메인별 캐시), 그리고
    **SSRF 가드**(내부 IP 타깃·내부 리다이렉트 차단).
  - 콘텐츠 게이트와 크롤 헬스 분류.
  - **프롬프트 캐싱 분할** — 정적 프리픽스에 `{post}`가 없는지, 백엔드별로 올바른
    캐시 마커만 들어가는지, 양쪽 `langchain_aws` 변환 함수를 실제로 통과하는지.
  - 필터/요약 내부 로직(점수 경계, title 덮어쓰기, None 응답 정렬, 잘못된 출력
    처리, `max_posts` 요약-전 캡), thinking budget 클램프, Bedrock 모델 정보
    레지스트리와 크로스 리전 ID 구성.
  - 설정 검증(명시적 null → `default_factory` 복원 포함).
  - 요약 정규화/후처리(태그 순서 보존, 종결 콜론 교정, 그리고 **HTML 새니타이즈**·
    **URL 스킴 화이트리스트**의 XSS 가드).
  - **AWS 헬퍼**(SES MIME 빌드, S3 키/프리픽스, 404 분기, Batch 폴링 상태 머신 —
    페이크 boto 클라이언트로).
  - **`run_batch`** 의 `"null"` 센티넬 파라미터 처리와 날짜 검증, **`main.handler`**
    의 제어 흐름(무-게시물 200 / 예외 500).
  - 엔드투엔드 Jinja2 렌더링(점수 배지·다크모드·접근성·alt 텍스트), SNS 알림 배선.

  `conftest.py`가 `sys.path`를 바로잡고 공유 픽스처를 제공합니다.
- **필터링 점수 평가 하버스** — `scripts/eval_filtering.py`가 라벨링된 평가셋
  (`tests/eval_data/filtering_eval_set.json`)에 대해 결정성, 0.05 그리드 준수,
  기대 밴드 일치도를 측정합니다. 메트릭은 `app/src/eval_metrics.py`의 순수 함수가
  계산하며(이 함수들은 `tests/test_eval_metrics.py`로 검증), 기본은 **드라이런
  (비용 0)**, `--live`는 운영 `Summarizer`의 필터 체인을 그대로 써 실제 Bedrock으로
  점수를 매겨 분포를 출력하고, 임계치(결정성 ≥99%, 온그리드 ≥99%, 밴드 ≥80%)에
  못 미치면 비정상 종료합니다. **라이브 측정 결과(Sonnet 4.6, 사고 OFF,
  2026-06):** 결정성 100%(σ=0.000), 0.05 그리드 준수 100%, 밴드 일치 100% —
  "재현 가능한 점수"라는 목표와 앵커 점수제가 실측으로 확인됐습니다.
- **`.github/workflows/ci.yml`** — push와 PR마다 실행합니다: 의존성 설치,
  `ruff check`, `ruff format --check`, **`mypy`**(차단형이며, 컨테이너 임포트
  루트와 맞추려고 `cd app && mypy .`로 실행 — 진입점 `main.py`/`run_batch.py`까지
  포함), 커버리지 포함 `pytest`, 평가 하버스 **드라이런**. 별도 잡이 **`cdk synth`**
  (차단형)를 **Lambda 경로(`config-ci.yaml`)와 Batch 경로(`config-ci-batch.yaml`)
  양쪽**으로 돌려 인프라가 컴파일되는지 검증하며(Batch가 운영 컴퓨트 토폴로지),
  `CDK_SYNTH_DUMMY_AZS=1`로 AWS 자격증명 없이도 VPC가 synth되게 합니다.

---

## 21. 로컬 vs AWS 차이

노트북에서는 잘 되는데 AWS에서는 실패할 수 있는 것들과, 그 이유입니다.

1. **데이터센터 IP 안티봇 차단.** 일부 소스(특히 `ai.meta.com`, `x.ai`, 그리고
   공격적으로 레이트 리밋되는 Medium 피드)는 거주지 IP에서는 잘 열려도 AWS NAT
   송신 IP에는 `403`이나 챌린지 페이지를 돌려줍니다. 적용한 완화책은 완전한
   브라우저형 헤더(`Sec-Fetch-*` 포함)와 일시 재시도입니다. 최후의 수단(기본
   비활성)은 해당 소스를 거주지 프록시나 헤드리스 브라우저로 우회시키는
   것입니다. 이제 이런 소스는 조용히 사라지는 대신 크롤 헬스 알림에 `FAILED`로
   드러납니다.
2. **Lambda에는 브라우저가 없음.** Lambda 이미지에 Chrome이 없어
   `convert_to_images`가 동작하지 못합니다. 변환기가 "Batch를 쓰거나 기능을
   끄라"는 안내와 함께 즉시 실패합니다.
3. **IPv6 이슈.** 일부 호스트가 AWS에서 IPv6로 붙으면 오작동하므로 `RssFetcher`가
   IPv4를 강제합니다(`force_ipv4`).
4. **오래된 소스 URL.** 여러 피드가 이전했습니다(DeepMind → `deepmind.google`,
   Facebook → `engineering.fb.com`, MS Research → `microsoft.com/.../feed`,
   Netflix → Medium). 크로스 호스트 리디렉트 추적에 의존하지 않도록 설정을 표준
   URL로 갱신했습니다.

---

## 22. 운영 런북

- **온디맨드 실행 트리거:** `python app/run_batch.py --end-date YYYY-MM-DD
  --language ko --recipients you@example.com`(Batch), 또는 로컬에서
  `python app/main.py ...`.
- **"No posts found"가 뜰 때:** 크롤 헬스 SNS 알림이나 로그에서 `FAILED` 소스를
  확인하세요. 날짜 윈도우(`days_back`)가 최신 글을 포함하는지도 확인합니다.
- **특정 소스가 안 보일 때:** 헬스 리포트에서 `FAILED`(페치 오류/안티봇)인지
  `EMPTY`(기간 내 글 없음/오래된 피드)인지 찾으세요. 설정 URL을 갱신하거나 프록시
  뒤로 옮깁니다.
- **요약이 비거나 너무 짧을 때:** 콘텐츠 게이트가 이를 막아 주지만, 충실한 글까지
  제외된다면 `scraping.min_content_length`를 낮추세요.
- **이메일이 안 갈 때:** SES 아이덴티티·검증 상태와 발신자 주소를 확인하세요. SNS
  실패 알림에서 구체적인 오류를 볼 수 있습니다.

---

## 23. 알려진 한계 & 향후 과제

- **강한 안티봇 소스**(x.ai, 가끔 Meta)는 AWS 데이터센터 IP에서 안정적으로 쓰려면
  거주지 프록시나 헤드리스 브라우저가 필요합니다. 지금은 `FAILED`로 보고되며 SNS
  알림에 드러납니다(조용히 사라지지 않습니다).
- **프롬프트 캐싱의 효과는 제한적입니다.** 한 번 실행에 글을 몇 개밖에 처리하지
  않고, 필터링 프리픽스가 약 1,020~1,030 토큰이라 Bedrock 캐시 최소 단위 경계에
  걸칩니다(요약 약 1,350 토큰은 확실히 캐시됩니다). 비용 절감은 본질적으로
  작습니다.
- **저빈도 소스**(예: `eugeneyan.com`)는 ACTIVE이긴 하나 게시 주기가 낮아 자주
  `EMPTY`로 나타날 수 있습니다 — 제거 대상이 아니라 관찰 대상입니다.
- **커버리지.** 핵심 경로는 테스트했지만(전체 약 62%), `feed_parser`의 사이트별
  스크레이퍼 내부 휴리스틱과 `newsletter_renderer`의 Selenium 캡처 경로는 라이브
  의존성이 커서 단위 테스트가 얕습니다.
- **평가셋 규모.** 현재 라벨링된 평가셋은 합성 글 6건으로, 회귀 게이트로는
  충분하지만 통계적으로는 작습니다. 실제 기사 발췌로 확장하면 밴드 경계 보정이
  더 정밀해집니다. (점수 결정성·그리드·밴드 일치는 라이브에서 이미 100%
  검증됐습니다 — §20.)

---

*이 문서는 코드와 함께 관리됩니다. 동작을 바꿀 때는 같은 변경 안에서 여기 관련
섹션도 함께 갱신하세요.*
