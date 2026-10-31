"""End-to-end snapshot of one issue: what readers receive and what S3 keeps.

Only the outside world is faked: the crawl (a fixed set of posts), Bedrock (a
model that answers per stage from the post in its prompt) and AWS (an in-memory
session). Everything between, from filtering to the MIME message, runs for real,
so a refactor that changes any byte of the delivered issue fails here.

Regenerate the snapshots after an intended output change with
``UPDATE_GOLDEN=1 pytest tests/test_golden_output.py`` and review the diff.
"""

from __future__ import annotations

import email
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import pytest
from botocore.exceptions import ClientError
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app import main
from app.domain.constants import Language
from app.domain.crawl import CrawlReport, SourceHealth, SourceStatus
from app.domain.post import Post
from app.src.feed_parser import PostCollector
from app.src.model_factory import BedrockLanguageModelFactory

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
CI_CONFIG = Path(__file__).parent.parent / "app" / "configs" / "config-ci-batch.yaml"
BUCKET = "tech-digest-ci-placeholder-bucket"
RECIPIENTS = ["reader.one@example.com", "reader.two@example.com"]
END_DATE = "2026-10-31"


class _Article:
    def __init__(self, key, title, score, published, cased=None):
        self.key, self.title, self.score, self.published = key, title, score, published
        # The title-case model's answer: ASCII marks, as the real model writes.
        self.cased = cased or title

    @property
    def link(self) -> str:
        return f"https://aws.amazon.com/blogs/machine-learning/{self.key}/"

    @property
    def reference(self) -> str:
        return f"https://docs.aws.amazon.com/{self.key}/guide.html"

    def post(self) -> Post:
        body = " ".join(
            f"Paragraph {i} of {self.key} explains the design in detail."
            for i in range(30)
        )
        return Post(
            title=self.title,
            link=self.link,
            published_date=datetime.fromisoformat(self.published),
            content=(
                f"<article><h1>{self.title}</h1><p>marker:{self.key}</p>"
                f"<p>{body}</p><p>See <a href='{self.reference}'>the guide</a>.</p>"
                "</article>"
            ),
            source="aws",
        )


ARTICLES = [
    _Article(
        "launch-a",
        "announcing faster sagemaker ai jobs",
        0.95,
        "2026-10-28",
        "Announcing Faster SageMaker AI Jobs",
    ),
    _Article(
        "launch-b",
        "TorchNeuron, Amazon\u2019s new PyTorch backend",
        0.90,
        "2026-10-27",
        "TorchNeuron, Amazon's New PyTorch Backend",
    ),
    _Article(
        "guide-old",
        "a deep guide to agentcore gateway",
        0.75,
        "2026-10-25",
        "A Deep Guide to AgentCore Gateway",
    ),
    _Article(
        "guide-new",
        "evaluating models with Bedrock \u2013 part 2",
        0.75,
        "2026-10-29",
        "Evaluating Models with Bedrock - Part 2",
    ),
    _Article("tutorial", "Configure a desktop app", 0.70, "2026-10-26"),
    _Article("roundup", "How a user group grew", 0.05, "2026-10-30"),
]


def _summary(article: _Article, language: Language) -> str:
    if language is Language.KO:
        para = (
            f"{article.key} 글은 설계와 운영 방법을 단계별로 설명합니다. "
            "구성 요소마다 역할과 한계를 짚고, 실제 배포에서 확인한 수치를 함께 보여줍니다. "
        )
        body = "\n\n".join(para * 4 for _ in range(6))
        one_liner = f"{article.key} 글의 핵심을 한 문장으로 정리했습니다."
    else:
        para = (
            f"The {article.key} post walks through the design and its operation step "
            "by step, naming what each component does and where it stops. "
        )
        body = "\n\n".join(para * 5 for _ in range(6))
        one_liner = f"The gist of {article.key} in one sentence."
    return (
        f"<summary>\n### 📌 개요\n\n{body}\n\n- 첫째 항목\n- 둘째 항목\n</summary>\n"
        f"<one_liner>{one_liner}</one_liner>\n"
        f"<tags>SageMaker, {article.key}, MLOps</tags>\n"
        f"<urls>{article.reference}</urls>"
    )


GREETING = {
    Language.KO: (
        "안녕하세요! 이번 주에는 SageMaker AI 작업을 더 빨리 시작하는 방법과 Trainium용 "
        "새 PyTorch 백엔드, 그리고 AgentCore Gateway를 깊이 있게 다룬 안내서를 골랐습니다. "
        "바쁜 한 주였다면 위의 세 편만 읽어도 충분합니다."
    ),
    Language.EN: (
        "Hello! This week brings faster SageMaker AI job starts, a new PyTorch backend "
        "for Trainium, and a deep guide to AgentCore Gateway. If the week got away "
        "from you, these three are the ones worth reading first, in that order, with "
        "a coffee."
    ),
}


def _quoted(title: str) -> str:
    return json.dumps(title, ensure_ascii=False)


def _article_in(text: str) -> _Article:
    [match] = [a for a in ARTICLES if f"marker:{a.key}" in text]
    return match


def _fake_get_model(language: Language):
    def get_model(self, model_id, *, stage=None, **kwargs):
        def answer(prompt: Any) -> AIMessage:
            text = prompt.to_string()
            if stage == "filtering":
                article = _article_in(text)
                reply = f"<score>{article.score:.2f}</score><reason>fixed</reason>"
            elif stage == "summarization":
                reply = _summary(_article_in(text), language)
            elif stage == "greeting":
                reply = GREETING[language]
            elif stage == "title-case":
                asked = sorted(
                    (a for a in ARTICLES if _quoted(a.title) in text),
                    key=lambda a: text.index(_quoted(a.title)),
                )
                reply = json.dumps([a.cased for a in asked])
            else:
                raise AssertionError(f"unexpected model stage {stage!r}")
            return AIMessage(content=reply)

        return RunnableLambda(answer)

    return get_model


class _FakeCollector:
    def __init__(self):
        self.report = CrawlReport(
            sources=[
                SourceHealth(
                    url="https://aws.amazon.com/blogs/amazon-ai/feed/",
                    fetcher="RssFetcher",
                    status=SourceStatus.OK,
                    post_count=len(ARTICLES),
                )
            ]
        )

    def collect_posts(self, start_date, end_date):
        return [a.post() for a in ARTICLES]


def _missing(operation: str) -> ClientError:
    return ClientError({"Error": {"Code": "404"}}, operation)


class _FakeS3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _missing("HeadObject")
        return {}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])

    def upload_file(self, path, bucket, key):
        self.objects[key] = Path(path).read_bytes()

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")

        class _Body:
            def __init__(self, data):
                self.data = data

            def read(self):
                return self.data

        return {"Body": _Body(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()
        return {}


class _FakeSES:
    def __init__(self):
        self.sent: list[dict] = []

    def send_raw_email(self, **kwargs):
        self.sent.append(kwargs)
        return {"MessageId": f"m-{len(self.sent)}"}


class _FakeSESv2:
    def list_suppressed_destinations(self, **kwargs):
        return {"SuppressedDestinationSummaries": []}


class _FakeSession:
    def __init__(self, s3: _FakeS3, ses: _FakeSES):
        self.region_name = "ap-northeast-2"
        self.clients = {"s3": s3, "ses": ses, "sesv2": _FakeSESv2()}

    def client(self, service, *args, **kwargs):
        if service == "bedrock-runtime":
            return object()
        return self.clients[service]


def _run_issue(
    monkeypatch,
    tmp_path,
    language: Language,
    seed: dict[str, bytes] | None = None,
    get_model=None,
) -> dict[str, Any]:
    """Run the handler once; ``seed`` pre-populates the bucket (a prior attempt's
    ledger, say) and ``get_model`` replaces the fake model factory."""
    config = main.Config.from_yaml(str(CI_CONFIG))
    config.newsletter.send_emails = True
    s3 = _FakeS3(
        {
            "tech-digest/recipients/recipients-dev.txt": "\n".join(RECIPIENTS).encode(),
            **(seed or {}),
        }
    )
    ses = _FakeSES()
    session = _FakeSession(s3, ses)

    monkeypatch.setattr(main.Config, "load", classmethod(lambda cls: config))
    monkeypatch.setattr(boto3, "Session", lambda *a, **k: session)
    monkeypatch.setattr(
        PostCollector, "from_urls", classmethod(lambda cls, urls: _FakeCollector())
    )
    monkeypatch.setattr(
        BedrockLanguageModelFactory,
        "get_model",
        get_model or _fake_get_model(language),
    )
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setattr(main, "ROOT_DIR", tmp_path)

    result = main.handler({"END_DATE": END_DATE, "LANGUAGE": language.value}, None)
    assert result["statusCode"] == 200, result

    messages = []
    for sent in ses.sent:
        message = email.message_from_string(sent["RawMessage"]["Data"])
        [part] = [p for p in message.walk() if p.get_content_type() == "text/html"]
        messages.append(
            {
                "source": sent["Source"],
                "destinations": sent["Destinations"],
                "headers": {
                    name: str(
                        email.header.make_header(
                            email.header.decode_header(message[name])
                        )
                    )
                    for name in ("Subject", "From", "To", "List-Unsubscribe")
                },
                "html": part.get_payload(decode=True).decode("utf-8"),
            }
        )
    ledgers = {
        key: {k: v for k, v in json.loads(body).items() if k != "updated_at"}
        for key, body in s3.objects.items()
        if key.startswith("tech-digest/deliveries/")
    }
    return {
        "messages": messages,
        "s3_keys": sorted(s3.objects),
        "ledgers": ledgers,
        "objects": s3.objects,
    }


def _check(name: str, actual: str) -> None:
    path = GOLDEN_DIR / name
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
    assert path.exists(), f"missing snapshot {path}; run with UPDATE_GOLDEN=1"
    assert actual == path.read_text(encoding="utf-8"), (
        f"{name} differs from its snapshot"
    )


@pytest.mark.parametrize(
    "language", [Language.KO, Language.EN], ids=lambda lang: lang.value
)
def test_the_delivered_issue_matches_its_snapshot(monkeypatch, tmp_path, language):
    run = _run_issue(monkeypatch, tmp_path, language)

    assert [m["destinations"] for m in run["messages"]] == [[r] for r in RECIPIENTS]
    [html] = {m["html"] for m in run["messages"]}
    _check(f"issue-{language.value}.html", html)
    envelope = {
        "messages": [
            {k: v for k, v in m.items() if k != "html"} for m in run["messages"]
        ],
        "s3_keys": run["s3_keys"],
        "ledgers": run["ledgers"],
    }
    _check(
        f"delivery-{language.value}.json",
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    # The lineup itself, so a snapshot refresh cannot quietly accept a wrong pick:
    # by score, recency breaking the 0.75 tie, capped at max_posts.
    titles = re.findall(r'<h2 class="card-title"[^>]*>([^<]+)</h2>', html)
    assert titles == [
        "Announcing Faster SageMaker AI Jobs",
        "TorchNeuron, Amazon\u2019s New PyTorch Backend",
        "Evaluating Models with Bedrock \u2013 Part 2",
    ]


def test_a_retry_after_a_partial_send_serves_only_the_rest(monkeypatch, tmp_path):
    """A Batch retry of a run that reached reader.one before dying: one message,
    to reader.two, and the rebuild archived beside (not over) the delivered copy."""
    ledger_key = f"tech-digest/deliveries/delivered-dev-{END_DATE}-ko.json"
    canonical_key = f"tech-digest/newsletters/newsletter-dev-{END_DATE}-ko.html"
    delivered_copy = b"<html>the issue reader.one received</html>"
    monkeypatch.setenv("AWS_BATCH_JOB_ATTEMPT", "2")

    run = _run_issue(
        monkeypatch,
        tmp_path,
        Language.KO,
        seed={
            ledger_key: json.dumps(
                {"stage": "dev", "delivered": [RECIPIENTS[0]]}
            ).encode(),
            canonical_key: delivered_copy,
        },
    )

    assert [m["destinations"] for m in run["messages"]] == [[RECIPIENTS[1]]]
    assert run["messages"][0]["headers"]["To"] == RECIPIENTS[1]
    assert run["ledgers"] == {ledger_key: {"stage": "dev", "delivered": RECIPIENTS}}
    retry_key = f"tech-digest/newsletters/retries/2/newsletter-dev-{END_DATE}-ko.html"
    assert retry_key in run["s3_keys"]
    assert run["objects"][canonical_key] == delivered_copy


def test_short_answer_stages_cap_their_output(monkeypatch, tmp_path):
    """Filter, greeting and title-case calls pass an explicit max_tokens: left
    unset, each request reserves the model's full output maximum against the
    per-minute token quota."""
    from app.src.greeter import GREETING_MAX_TOKENS
    from app.src.summarizer import Summarizer
    from app.src.title_caser import TITLE_CASE_MAX_TOKENS

    fake = _fake_get_model(Language.KO)
    requested: dict[str, int | None] = {}

    def get_model(self, model_id, *, stage=None, **kwargs):
        requested.setdefault(stage, kwargs.get("max_tokens"))
        return fake(self, model_id, stage=stage, **kwargs)

    _run_issue(monkeypatch, tmp_path, Language.KO, get_model=get_model)
    assert requested["filtering"] == Summarizer.FILTER_MAX_TOKENS
    assert requested["greeting"] == GREETING_MAX_TOKENS
    assert requested["title-case"] == TITLE_CASE_MAX_TOKENS
    # Enough for a reasoning model to think and still answer.
    assert min(requested[s] or 0 for s in ("filtering", "greeting", "title-case")) >= (
        4096
    )
