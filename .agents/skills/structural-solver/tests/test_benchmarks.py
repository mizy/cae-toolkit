"""Analytical benchmarks: tension bar and cantilever beam."""
import numpy as np

from solve import run_solve


def test_tension_bar_t10():
    """delta = F*L/(E*A), mean von Mises = F/A; rel error < 1%."""
    r = run_solve(geometry="box:0.1,0.01,0.01", units="m", material_name="steel",
                  order=2, mesh_size=0.005, fixes=["x_min"], loads=["x_max: fx=1000"],
                  scale=1.0, output_dir=None, with_benchmarks=False, name="t")
    s = r["results"]["field_summary"]
    delta_an = 1000 * 0.1 / (210e9 * 0.01 * 0.01)
    sigma_an = 1000 / (0.01 * 0.01)
    assert abs(s["load_point_delta"] - delta_an) / delta_an < 0.01
    assert abs(s["mean_von_mises"] - sigma_an) / sigma_an < 0.03


def test_cantilever_t10():
    """delta = F*L^3/(3*E*I), k = 3*E*I/L^3; rel error < 5%."""
    r = run_solve(geometry="box:0.1,0.01,0.01", units="m", material_name="steel",
                  order=2, mesh_size=0.0025, fixes=["x_min"], loads=["x_max: fz=-1000"],
                  scale=1.0, output_dir=None, with_benchmarks=False, name="c")
    s = r["results"]["field_summary"]
    I = 0.01 * 0.01**3 / 12
    delta_an = 1000 * 0.1**3 / (3 * 210e9 * I)
    k_an = 3 * 210e9 * I / 0.1**3
    assert abs(s["load_point_delta"] - delta_an) / delta_an < 0.05
    assert abs(s["stiffness"] - k_an) / k_an < 0.05


def test_t4_tension_bar():
    """T4 must be close (slightly stiff); allow 10%."""
    r = run_solve(geometry="box:0.1,0.01,0.01", units="m", material_name="steel",
                  order=1, mesh_size=0.0025, fixes=["x_min"], loads=["x_max: fx=1000"],
                  scale=1.0, output_dir=None, with_benchmarks=False, name="t4")
    s = r["results"]["field_summary"]
    delta_an = 1000 * 0.1 / (210e9 * 0.01 * 0.01)
    assert abs(s["load_point_delta"] - delta_an) / delta_an < 0.10


def test_reaction_balance_gate():
    r = run_solve(geometry="box:0.1,0.01,0.01", units="m", material_name="steel",
                  order=2, mesh_size=0.005, fixes=["x_min"], loads=["x_max: fx=1000"],
                  scale=1.0, output_dir=None, with_benchmarks=False, name="g")
    gates = r["gates"]
    assert gates["反力平衡 |ΣR+F|/|F|"]["passed"]
    assert gates["能量一致性 |SE-0.5u·f|/(0.5u·f)"]["passed"]
