"""Smoke tests of the example scripts on the CPU at tiny sizes: they parse their flags, run and write their outputs."""

import importlib.util
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_script(script, out, *args):
    """Run an example script in a fresh process; returns its summary.json."""
    cmd = [sys.executable, os.path.join(ROOT, script), *args, "--device", "cpu", "--quiet", "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with open(os.path.join(out, "summary.json")) as f:
        return json.load(f)


def test_droplet_splash_example(tmp_path):
    spec = importlib.util.spec_from_file_location("droplet_splash_example",
                                                  os.path.join(ROOT, "examples", "water", "droplet_splash.py"))
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    code = example.main(["--res", "24", "--steps", "3", "--no-render", "--device", "cpu", "--quiet",
                         "--levelset-opt", "ppc=9", "--out", str(tmp_path)])
    assert code == 0
    with open(tmp_path / "summary.json") as f:
        summary = json.load(f)
    assert summary["status"] == "completed" and summary["steps"] == 3 and summary["res"] == [24, 24]
    assert os.path.exists(tmp_path / "volume.csv") and os.path.exists(tmp_path / "frames.csv")


def test_dam_break_example(tmp_path):
    spec = importlib.util.spec_from_file_location("dam_break_example",
                                                  os.path.join(ROOT, "examples", "water", "dam_break.py"))
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    code = example.main(["--res", "24", "--steps", "3", "--no-render", "--device", "cpu", "--quiet",
                         "--levelset-opt", "ppc=9", "--out", str(tmp_path)])
    assert code == 0
    with open(tmp_path / "summary.json") as f:
        summary = json.load(f)
    assert summary["status"] == "completed" and summary["steps"] == 3 and summary["res"] == [24, 24]
    assert summary["a"] == 0.25 and summary["wall_hit_t"] is None  # the front is still far from the wall
    with open(tmp_path / "frames.csv") as f:
        assert f.readline().strip() == "step,t,front,area"


def test_zalesak_example(tmp_path):
    s = _run_script(os.path.join("examples", "benchmarks", "zalesak.py"), tmp_path, "--res", "32", "--steps", "6")
    assert s["benchmark"] == "zalesak" and s["steps"] == 6 and s["device"] == "cpu"
    assert abs(s["ratio_t0_final"] - 1.0) < 0.05
    assert os.path.exists(tmp_path / "interface.png") and os.path.exists(tmp_path / "volume.csv")


def test_sphere3d_example(tmp_path):
    s = _run_script(os.path.join("examples", "benchmarks", "sphere3d.py"), tmp_path,
                    "--res", "16", "--ppc", "27", "--steps", "4", "--mesh-steps", "0")
    assert s["benchmark"] == "sphere3d_enright" and s["steps"] == 4 and s["protocol"] == "full"
    assert abs(s["ratio_t0"]["4"] - 1.0) < 0.1
    assert os.path.exists(tmp_path / "meshes" / "mesh_0000.ply") and os.path.exists(tmp_path / "volume.csv")


def test_benchmark_protocol_settings():
    """The 2D benchmark protocols both seed with seed_min = ppc / 2; overriding it (e.g. --seed-min 0) is a
    custom run (its own output directory)."""
    spec = importlib.util.spec_from_file_location("benchmarks_common",
                                                  os.path.join(ROOT, "examples", "benchmarks", "_common.py"))
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    parser = common.make_parser("test", steps=1)

    def settings(*argv):
        return common.protocol_settings(parser.parse_args(list(argv)))

    assert settings() == (12, 6, 8, "full")
    assert settings("--protocol", "advection") == (0, 0, 8, "advection")
    assert settings("--protocol", "advection", "--ppc", "25") == (0, 0, 12, "advection")
    assert settings("--protocol", "advection", "--seed-min", "0") == (0, 0, 0, "custom")
    assert settings("--protocol", "advection", "--seed-min", "8") == (0, 0, 8, "advection")
    assert settings("--seed-min", "0") == (12, 6, 0, "custom")
