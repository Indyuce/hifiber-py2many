import os
import pathlib
from unittest.mock import Mock

from .language import LanguageSettings
from .python_transformer import PythonTranspiler, RestoreMainRewriter
from .rewriters import InferredAnnAssignRewriter

CI = os.environ.get("CI", "0")
if CI in ["1", "true"]:  # pragma: no cover
    from .pyrs import settings as rust_settings
else:
    try:  # pragma: no cover
        from .pyrs import settings as rust_settings
    except ImportError:
        from pyrs import settings as rust_settings


PY2MANY_DIR = pathlib.Path(__file__).parent
ROOT_DIR = PY2MANY_DIR.parent
FAKE_ARGS = Mock(indent=4)


def python_settings(args, env=os.environ):
    return LanguageSettings(
        PythonTranspiler(),
        ".py",
        "Python",
        formatter=["black"],
        rewriters=[RestoreMainRewriter()],
        post_rewriters=[InferredAnnAssignRewriter()],
    )


ALL_SETTINGS = {
    "python": python_settings,
    "rust": rust_settings,
}


def _get_all_settings(args, env=os.environ):
    return dict((key, func(args, env=env)) for key, func in ALL_SETTINGS.items())
