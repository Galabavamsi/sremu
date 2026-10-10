"""sremu.sysinfo (D12 platform probes, D15 numba cache directory)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from sremu import sysinfo

pytestmark = pytest.mark.unit


def test_d12_sysinfo_is_layer_0_and_does_not_import_numba():
    code = "import sys, sremu.sysinfo; print('numba' in sys.modules)"
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "False"


def test_d12_cli_uses_the_shared_probes():
    from sremu import cli

    info = cli.collect_info()
    assert set(sysinfo.platform_info()) <= set(info)
    assert info["logical_cpus"] == os.cpu_count()
    assert info["versions"]["numpy"] == sysinfo.dist_version("numpy")
    for name in ("_cpu_model", "_gpus", "_aws_instance", "_numba_state", "_power_plan_or_governor"):
        assert not hasattr(cli, name), name  # moved to sysinfo, not duplicated


def test_d12_parse_powercfg_output():
    out = "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)"
    assert sysinfo.parse_powercfg(out) == "Balanced"
    assert sysinfo.parse_powercfg("GUID: x  (Ultimate Performance) ") == "Ultimate Performance"
    assert sysinfo.parse_powercfg("no plan name") == "no plan name"


def test_d12_aws_instance_from_dmi(tmp_path):
    (tmp_path / "sys_vendor").write_text("Amazon EC2\n", encoding="utf-8")
    (tmp_path / "product_name").write_text("c6i.8xlarge\n", encoding="utf-8")
    (tmp_path / "board_asset_tag").write_text("i-0123456789abcdef0\n", encoding="utf-8")
    assert sysinfo.aws_instance(tmp_path) == {"type": "c6i.8xlarge", "id": "i-0123456789abcdef0"}
    (tmp_path / "board_asset_tag").unlink()
    assert sysinfo.aws_instance(tmp_path) == {"type": "c6i.8xlarge", "id": None}
    (tmp_path / "sys_vendor").write_text("LENOVO\n", encoding="utf-8")
    assert sysinfo.aws_instance(tmp_path) is None
    assert sysinfo.aws_instance(tmp_path / "missing") is None


def test_d12_governor_file(tmp_path):
    f = tmp_path / "scaling_governor"
    f.write_text("performance\n", encoding="utf-8")
    assert sysinfo.read_governor(f) == "performance"
    assert sysinfo.read_governor(tmp_path / "missing") is None
    assert sysinfo.LINUX_GOVERNOR_FILE.as_posix() == (
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    )


def test_d15_default_cache_dir_matches_numba_user_wide_cache(monkeypatch):
    from numba.misc.appdirs import AppDirs

    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    want = str(AppDirs(appname="numba", appauthor=False).user_cache_dir)
    assert os.path.normcase(sysinfo.default_numba_cache_dir()) == os.path.normcase(want)
    monkeypatch.setenv("NUMBA_CACHE_DIR", "/somewhere/nb")
    assert sysinfo.default_numba_cache_dir() == "/somewhere/nb"
    assert sysinfo.default_numba_cache_dir({}) == sysinfo.default_numba_cache_dir(
        {"NUMBA_CACHE_DIR": ""}
    )


def test_d15_nosvml_cache_dir_rules(monkeypatch):
    assert sysinfo.NOSVML_SUFFIX == "-nosvml"
    assert sysinfo.nosvml_cache_dir("/c/nb") == "/c/nb-nosvml"
    assert sysinfo.nosvml_cache_dir("/c/nb/") == "/c/nb-nosvml"
    assert sysinfo.nosvml_cache_dir("C:\\nb\\") == "C:\\nb-nosvml"
    assert sysinfo.nosvml_cache_dir("/c/nb-nosvml") == "/c/nb-nosvml"  # idempotent
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    assert sysinfo.nosvml_cache_dir() == sysinfo.default_numba_cache_dir() + "-nosvml"
    assert sysinfo.nosvml_cache_dir("") == sysinfo.default_numba_cache_dir() + "-nosvml"


def test_d12_numba_state_reports_svml_threads_and_cache():
    st = sysinfo.numba_state()
    import numba

    assert st["version"] == numba.__version__
    assert st["cache_dir"] == str(numba.config.CACHE_DIR)
    assert st["disable_intel_svml"] == bool(numba.config.DISABLE_INTEL_SVML)
    assert isinstance(st["threading_layers_available"], list)
    assert sysinfo.numba_svml_disabled() == bool(numba.config.DISABLE_INTEL_SVML)
    assert sysinfo.numba_threads() == numba.get_num_threads()
