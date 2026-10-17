"""The PFM-LS backend of the free-surface solver: volume guard (both reinitialisation units), NaN scrub
and checkpoints (CPU)."""

import numpy as np
import pytest
import warp as wp

import pfmls
from pfmls.fluid import FluidConfig, PFMBackendConfig
from pfmls.fluid.pfm_backend import PFMLevelSetBackend

CPU = "cpu"
N = 32


def _falling_mac(speed):
    """MAC velocity (0, -speed) inside the box, zero normal velocity on the walls."""
    ux = wp.zeros((N + 1, N), dtype=float, device=CPU)
    vy = np.full((N, N + 1), -speed, dtype=np.float32)
    vy[:, 0] = vy[:, -1] = 0.0
    uy = wp.array(vy, dtype=float, device=CPU)
    return ux, uy


@pytest.mark.parametrize("schedule", [dict(reinit_long=4, reinit_short=2),
                                      dict(reinit_long=0.04, reinit_short=0.02, reinit_unit="time")])
def test_volume_guard_rejects_and_redoes_step(schedule):
    """A rejected long reinitialisation is rolled back and the step is redone without it, giving exactly
    the state of a level set stepped with ``skip_long_reinit=True``; the guard runs for intervals in steps
    and in time units alike."""
    sdf = pfmls.sdf.circle((0.5, 0.6), 0.2)
    # a guard of 1e-9 rejects any change of the liquid area
    cfg = FluidConfig(nx=N, ny=N, pfm_levelset=PFMBackendConfig.plain(ppc=9, volume_guard=1e-9, **schedule))
    backend = PFMLevelSetBackend(cfg, sdf, device=CPU)
    ref = pfmls.PFMLevelSet2D(res=N, ppc=9, device=CPU, **schedule)
    ref.init_from_sdf(sdf)
    ux, uy = _falling_mac(2.0)
    flow = pfmls.flows.MACVelocity2D(ux, uy, cfg.dx)
    dt = 0.01
    longs = 0
    for _ in range(9):
        long_due = backend.ls.reinit_due(dt)[0]
        longs += long_due
        before = backend.rejects
        backend.advance(ux, uy, dt)
        ref.step(dt, flow, skip_long_reinit=backend.rejects > before)
        np.testing.assert_array_equal(backend.ls.phi_grid(), ref.phi_grid())
    assert longs == 2
    assert backend.rejects >= 1  # (a long step whose cell count happens not to change is accepted)
    assert backend.ls.step_index == 9


def test_scrub_and_checkpoint_roundtrip():
    """Non-finite particles/grid values are repaired and counted; state_dict() round-trips bitwise."""
    sdf = pfmls.sdf.circle((0.5, 0.6), 0.2)
    cfg = FluidConfig(nx=N, ny=N, pfm_levelset=PFMBackendConfig.plain(ppc=9, reinit_long=4, reinit_short=2))
    a = PFMLevelSetBackend(cfg, sdf, device=CPU)
    ux, uy = _falling_mac(1.0)
    for _ in range(3):
        a.advance(ux, uy, 0.01)
    k = int(np.flatnonzero(a.ls.valid.numpy())[0])
    phi = a.ls.phi.numpy().copy()
    phi[k] = np.nan
    a.ls.phi.assign(phi)
    grid = a.ls.phi_g.numpy().copy()
    grid[3, 4] = np.inf
    a.ls.phi_g.assign(grid)
    a.advance(ux, uy, 0.01)
    s = a.stats()
    assert s["nan_particles"] == 1 and s["nan_scrubs"] >= 1
    assert np.isfinite(a.ls.phi_grid()).all()

    state = a.state_dict()
    for _ in range(5):
        a.advance(ux, uy, 0.01)
    b = PFMLevelSetBackend(cfg, sdf, device=CPU)
    b.load_state_dict(state)
    assert b.stats()["nan_particles"] == 1
    for _ in range(5):
        b.advance(ux, uy, 0.01)
    np.testing.assert_array_equal(a.ls.phi_grid(), b.ls.phi_grid())
