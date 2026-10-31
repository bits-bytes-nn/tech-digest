"""The container image starts the app the way the repo does: as the ``app``
package. ``cdk synth`` builds no image, so without these a wrong COPY target or
entry point would ship green. No Docker needed."""

from __future__ import annotations

import json
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"


def _instructions(dockerfile: str, keyword: str) -> list[str]:
    body = (APP / dockerfile).read_text(encoding="utf-8")
    return [
        line.removeprefix(keyword).strip()
        for line in body.splitlines()
        if line.startswith(f"{keyword} ")
    ]


def _cmd(dockerfile: str) -> list[str]:
    [cmd] = _instructions(dockerfile, "CMD")
    return json.loads(cmd)


def test_the_batch_image_runs_the_package_from_its_parent():
    assert _cmd("Dockerfile-batch") == ["python3", "-m", "app.main"]
    assert "app/" in [
        line.split()[-1] for line in _instructions("Dockerfile-batch", "COPY")
    ]
    [workdir] = _instructions("Dockerfile-batch", "WORKDIR")
    assert workdir != "/srv/app", "the package's parent must be the working dir"
