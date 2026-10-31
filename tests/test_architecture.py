"""The dependency rule of the ports-and-adapters layout, checked on the source.

``app/domain`` holds rules with no I/O, ``app/application`` the services written
against the interfaces in ``application/ports.py``, and ``app/adapters`` the code
that talks to the outside world. Dependencies point inward only: an adapter may
know the ports and the domain, a service may know the domain, and the domain
knows neither. ``app/main.py`` is the one place that sees all three and wires
them. Nothing enforces this at import time, so a stray import would compile,
pass every other test and quietly couple the layers again.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"

# What each layer must not import, as module-name prefixes. "langchain" covers
# langchain_core, langchain_aws and langchain_classic.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "domain": (
        "app.application",
        "app.adapters",
        "app.main",
        "app.run_batch",
        "app.configs",
        "boto3",
        "botocore",
        "requests",
        "feedparser",
        "selenium",
        "webdriver_manager",
        "jinja2",
        "langchain",
        "tenacity",
        "tqdm",
    ),
    "application": (
        "app.adapters",
        "app.main",
        "app.run_batch",
        "boto3",
        "botocore",
        "requests",
        "feedparser",
        "selenium",
        "jinja2",
        "langchain_aws",
    ),
    "adapters": ("app.application", "app.main", "app.run_batch"),
}

# Exceptions to the prefixes above: (file or "<layer>/*", module prefix).
ALLOWED: tuple[tuple[str, str], ...] = (
    # The retry policy lives in one place, the services' retry decorator, and
    # deciding whether a failed model call can succeed on retry means reading
    # the SDK's own error types: langchain_aws re-raises botocore's throttling
    # and connection errors unchanged. Only the exception classes are imported;
    # the application makes no AWS call.
    ("application/llm_batching.py", "botocore.exceptions"),
    # Adapters implement the ports, so they may import them, and nothing else
    # from the application.
    ("adapters/*", "app.application.ports"),
)

# Top-level modules every layer imports, held to the domain's rule so that
# importing them cannot pull an outer layer into the domain.
SHARED_AS_DOMAIN: frozenset[str] = frozenset({"logger.py", "paths.py"})


def _layer(path: Path, root: Path = APP) -> str | None:
    top = path.relative_to(root).parts[0]
    if top in SHARED_AS_DOMAIN:
        return "domain"
    return top if top in FORBIDDEN else None


def _imports(path: Path, root: Path = APP) -> set[str]:
    """Every name ``path`` imports, as ``module`` or ``module.name``, relative
    imports resolved, at any depth (lazy imports inside functions and
    ``TYPE_CHECKING`` blocks included)."""
    parts = list(path.relative_to(root.parent).with_suffix("").parts)
    package = ".".join(parts if parts[-1] == "__init__" else parts[:-1])
    package = package.removesuffix(".__init__")
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                base = package.rsplit(".", node.level - 1)[0]
                module = f"{base}.{module}" if module else base
            # The imported names, not the bare package: ``from app.application
            # import ports`` imports the ports module, not every service.
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def _within(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _violations(path: Path, root: Path = APP) -> list[str]:
    layer = _layer(path, root)
    assert layer is not None
    relative = path.relative_to(root).as_posix()
    allowed = [prefix for where, prefix in ALLOWED if where in (relative, f"{layer}/*")]
    return sorted(
        module
        for module in _imports(path, root)
        if any(
            _within(module, p) or (p == "langchain" and module.startswith(p))
            for p in FORBIDDEN[layer]
        )
        and not any(_within(module, p) for p in allowed)
    )


SOURCES = sorted(p for p in APP.rglob("*.py") if _layer(p))


def test_every_layer_has_source_files():
    """A moved directory must not turn the rule below into a vacuous pass."""
    assert {_layer(p) for p in SOURCES} == set(FORBIDDEN)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.relative_to(APP).as_posix())
def test_imports_point_inward(path):
    violations = _violations(path)
    assert not violations, f"{path.relative_to(APP)} imports {violations}"


class TestTheCheckItself:
    """The test above is only as good as the import resolver and the matcher."""

    @pytest.fixture
    def root(self, tmp_path) -> Path:
        for layer in ("domain", "application", "adapters"):
            (tmp_path / "app" / layer).mkdir(parents=True)
        return tmp_path / "app"

    def test_relative_and_lazy_imports_are_seen(self, root):
        probe = root / "domain" / "probe.py"
        probe.write_text(
            "from .post import Post\n"
            "def f():\n"
            "    import boto3\n"
            "    from app.adapters import feeds\n",
            encoding="utf-8",
        )
        assert _imports(probe, root) == {
            "app.domain.post.Post",
            "boto3",
            "app.adapters.feeds",
        }
        assert _violations(probe, root) == ["app.adapters.feeds", "boto3"]

    def test_any_langchain_package_is_kept_out_of_the_domain(self, root):
        probe = root / "domain" / "probe.py"
        probe.write_text("from langchain_core.runnables import Runnable\n", "utf-8")
        assert _violations(probe, root) == ["langchain_core.runnables.Runnable"]

    def test_an_adapter_may_import_the_ports_but_no_service(self, root):
        probe = root / "adapters" / "probe.py"
        probe.write_text(
            "from app.application.ports import ObjectStore\n"
            "from app.application import ports\n"
            "from app.application.summarize import Summarizer\n",
            encoding="utf-8",
        )
        assert _violations(probe, root) == ["app.application.summarize.Summarizer"]

    def test_the_shared_logger_is_held_to_the_domain_rule(self, root):
        probe = root / "logger.py"
        probe.write_text("from app.adapters.aws import send_email\n", "utf-8")
        assert _violations(probe, root) == ["app.adapters.aws.send_email"]

    def test_the_botocore_exception_is_allowed_only_where_it_is_listed(self, root):
        source = "from botocore.exceptions import ClientError\n"
        listed = root / "application" / "llm_batching.py"
        other = root / "application" / "delivery.py"
        listed.write_text(source, encoding="utf-8")
        other.write_text(source, encoding="utf-8")
        assert _violations(listed, root) == []
        assert _violations(other, root) == ["botocore.exceptions.ClientError"]
