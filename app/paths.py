"""Filesystem anchors for the ``app`` package and the repository around it.

Modules derive their paths from these instead of counting ``parent`` hops from
their own ``__file__``, so moving a module cannot silently move a directory.
"""

from pathlib import Path

#: The ``app`` package directory: templates/, configs/ and assets/ live here, and
#: it is the build context of the container image.
APP_DIR: Path = Path(__file__).resolve().parent
#: The repository root, where a local run keeps logs/, inputs/ and outputs/.
REPO_ROOT: Path = APP_DIR.parent
