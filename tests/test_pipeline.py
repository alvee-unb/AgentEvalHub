"""test_pipeline.py — Unit tests for the pipeline orchestrator."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_pipeline import StepResult, run_step, _python_in_venv


class TestStepResult:
    def test_default_status_pending(self):
        r = StepResult("test", "Test Step")
        assert r.status == "pending"
        assert r.passed is False

    def test_success_passed(self):
        r = StepResult("test", "Test Step")
        r.status = "success"
        assert r.passed is True

    def test_failed_not_passed(self):
        r = StepResult("test", "Test Step")
        r.status = "failed"
        assert r.passed is False


class TestPythonInVenv:
    def test_returns_string(self):
        result = _python_in_venv("nonexistent/venv")
        assert isinstance(result, str)

    def test_falls_back_to_current_interpreter(self):
        result = _python_in_venv("no_such_venv_xyz")
        assert "python" in result.lower()

    def test_finds_scripts_python(self, tmp_path):
        scripts = tmp_path / "Scripts"
        scripts.mkdir()
        py = scripts / "python.exe"
        py.write_text("")  # empty placeholder
        result = _python_in_venv(str(tmp_path))
        assert result == str(py)


class TestRunStep:
    def test_dry_run_returns_success(self):
        step = {"id": "test", "name": "Test", "script": "echo.py", "retry": 1}
        r = run_step(step, dry_run=True)
        assert r.status == "success"

    def test_successful_step(self, tmp_path):
        script = tmp_path / "ok.py"
        script.write_text("import sys; sys.exit(0)")
        step = {
            "id": "ok", "name": "OK Step",
            "script": str(script), "venv": "", "retry": 1,
        }
        r = run_step(step)
        assert r.status == "success"
        assert r.returncode == 0

    def test_failing_step(self, tmp_path):
        script = tmp_path / "fail.py"
        script.write_text("import sys; sys.exit(1)")
        step = {
            "id": "fail", "name": "Fail Step",
            "script": str(script), "venv": "", "retry": 1,
        }
        r = run_step(step)
        assert r.status == "failed"
        assert r.returncode == 1

    def test_step_duration_recorded(self, tmp_path):
        script = tmp_path / "fast.py"
        script.write_text("pass")
        step = {"id": "fast", "name": "Fast", "script": str(script), "venv": "", "retry": 1}
        r = run_step(step)
        assert r.duration_s >= 0.0

    def test_step_id_and_name_preserved(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text("pass")
        step = {"id": "my_id", "name": "My Name", "script": str(script), "venv": "", "retry": 1}
        r = run_step(step, dry_run=True)
        assert r.step_id == "my_id"
        assert r.name == "My Name"
