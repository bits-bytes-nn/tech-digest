"""The container images start the app the way the repo does: as the ``app``
package. ``cdk synth`` builds no image, so without these a wrong COPY target or
entry point would ship green. No Docker needed."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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


def test_the_lambda_image_imports_its_handler(tmp_path):
    # Stage exactly what the Dockerfile copies, then import the CMD's handler
    # with only that directory on the path, as the Lambda runtime does.
    for line in _instructions("Dockerfile-lambda", "COPY"):
        *sources, destination = line.split()
        for source in sources:
            if source.startswith("requirements"):
                continue
            target = tmp_path / destination
            if (APP / source).is_dir():
                shutil.copytree(APP / source, target, dirs_exist_ok=True)
            else:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy(APP / source, target / source)
    [handler] = _cmd("Dockerfile-lambda")
    module, function = handler.rsplit(".", 1)
    result = subprocess.run(
        [sys.executable, "-E", "-s", "-c", f"import {module}; {module}.{function}"],
        cwd=tmp_path,
        env={"LOG_TO_FILE": "false", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_the_batch_image_runs_the_package_from_its_parent():
    assert _cmd("Dockerfile-batch") == ["python3", "-m", "app.main"]
    assert "app/" in [
        line.split()[-1] for line in _instructions("Dockerfile-batch", "COPY")
    ]
    [workdir] = _instructions("Dockerfile-batch", "WORKDIR")
    assert workdir != "/srv/app", "the package's parent must be the working dir"
