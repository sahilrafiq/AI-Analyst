"""Shared test helpers: repo path, a fake Jupyter kernel and a working-directory base class.
The tests need no Ollama and no Jupyter kernel: the agent loop is driven by scripted 'model' replies."""
import contextlib
import io
import os
import shutil
import sys
import tempfile
import traceback
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")

import agent  # noqa: E402


class FakeSession:
    """exec()-based stand-in for the Jupyter kernel: persistent namespace, same run() contract as PythonSession."""

    def __init__(self, workdir, setup=agent.BASE_SETUP):
        self.setup, self.workdir = setup, Path(workdir).resolve()
        self.workdir.mkdir(exist_ok=True)
        self.restart()

    def restart(self):
        self.ns = {}
        _, self.setup_error = self._exec(self.setup)

    def _exec(self, code):
        cwd0 = os.getcwd()
        os.chdir(self.workdir)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, self.ns)
            return buf.getvalue(), None
        except Exception:
            return buf.getvalue(), "\n".join(traceback.format_exc().splitlines()[-4:])
        finally:
            os.chdir(cwd0)

    def run(self, code, timeout=120):
        return self._exec(code)

    def close(self):
        pass


def resp(text):
    """A scripted reply in the shape of an Ollama /api/chat answer."""
    return {"message": {"content": text}, "eval_count": 5, "eval_duration": 1_000_000_000}


def captured(fn, *args, **kwargs):
    """Run fn and return (result, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kwargs)
    return out, buf.getvalue()


class InTempDir(unittest.TestCase):
    """Runs every test inside a fresh temporary directory (files written with relative paths land there)."""

    def setUp(self):
        self._cwd = os.getcwd()
        self.tmp = Path(tempfile.mkdtemp(prefix="ai_analyst_test_"))
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)


class WithFakeAgent(InTempDir):
    """InTempDir plus a patched agent module: fake kernel, and `self.script(fn)` installs a scripted model."""

    def setUp(self):
        super().setUp()
        self._orig = (agent.PythonSession, agent.post)
        agent.PythonSession = FakeSession

    def tearDown(self):
        agent.PythonSession, agent.post = self._orig
        super().tearDown()

    def script(self, fn):
        agent.post = fn

    def new_agent(self, workdir=None, **kw):
        return agent.Agent("fake", workdir=workdir or self.tmp / "ws", **kw)
