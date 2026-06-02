import numpy as np
from devito import *
from ultrawave import Receiver


def Acoustic3DAdjointOperatorFirstOrder(model, residual, forward_fields, rec):
    """
    Prototype discrete adjoint for UltraWave first-order acoustic system.

    Forward variables:
        p, vx, vy, vz, rhox, rhoy, rhoz

    Adjoint variables:
        lp, lvx, lvy, lvz, lrhox, lrhoy, lrhoz

    This is a main-domain adjoint prototype.
    PML/damping adjoint terms are NOT fully included yet.
    """

    grid = model.grid
    x, y, z = grid.dimensions
    dt = model.critical_dt

    nt = residual.shape[0]

    # --------------------------------------------------
    # Adjoint fields
    # --------------------------------------------------

    lp = TimeFunction(
        name="lp",
        grid=grid,
        staggered=NODE,
        time_order=1,
        space_order=model.space_order,
        save=nt
    )

    lvx = TimeFunction(
        name="lvx",
        grid=grid,
        time_order=1,
        space_order=model.space_order,
        staggered=x,
        save=nt
    )

    lvy = TimeFunction(
        name="lvy",
        grid=grid,
        time_order=1,
        space_order=model.space_order,
        staggered=y,
        save=nt
    )

    lvz = TimeFunction(
        name="lvz",
        grid=grid,
        time_order=1,
        space_order=model.space_order,
        staggered=z,
        save=nt
    )

    lrhox = TimeFunction(
        name="lrhox",
        grid=grid,
        staggered=NODE,
        time_order=1,
        space_order=model.space_order,
        save=nt
    )

    lrhoy = TimeFunction(
        name="lrhoy",
        grid=grid,
        staggered=NODE,
        time_order=1,
        space_order=model.space_order,
        save=nt
    )

    lrhoz = TimeFunction(
        name="lrhoz",
        grid=grid,
        staggered=NODE,
        time_order=1,
        space_order=model.space_order,
        save=nt
    )

    # --------------------------------------------------
    # Derivatives
    # --------------------------------------------------

    lp_dx = getattr(lp, "d%s" % lp.space_dimensions[0].name)
    lp_dy = getattr(lp, "d%s" % lp.space_dimensions[1].name)
    lp_dz = getattr(lp, "d%s" % lp.space_dimensions[2].name)

    lrhox_dx = getattr(lrhox, "d%s" % lrhox.space_dimensions[0].name)
    lrhoy_dy = getattr(lrhoy, "d%s" % lrhoy.space_dimensions[1].name)
    lrhoz_dz = getattr(lrhoz, "d%s" % lrhoz.space_dimensions[2].name)

    # --------------------------------------------------
    # Adjoint equations
    #
    # Forward coupling:
    #   p -> vx,vy,vz
    #   vx -> rhox
    #   vy -> rhoy
    #   vz -> rhoz
    #   rhox,rhoy,rhoz -> p
    #
    # Adjoint reverses those couplings.
    # --------------------------------------------------

    eq_lp = Eq(
        lp.backward,
        lp
        + dt * (
            lvx.dx / model.rho
            + lvy.dy / model.rho
            + lvz.dz / model.rho
        ),
        subdomain=grid.subdomains["main"]
    )

    eq_lvx = Eq(
        lvx.backward,
        lvx
        + dt * model.rho * lrhox_dx,
        subdomain=grid.subdomains["main"]
    )

    eq_lvy = Eq(
        lvy.backward,
        lvy
        + dt * model.rho * lrhoy_dy,
        subdomain=grid.subdomains["main"]
    )

    eq_lvz = Eq(
        lvz.backward,
        lvz
        + dt * model.rho * lrhoz_dz,
        subdomain=grid.subdomains["main"]
    )

    eq_lrhox = Eq(
        lrhox.backward,
        lrhox
        + model.vp**2 * lp,
        subdomain=grid.subdomains["main"]
    )

    eq_lrhoy = Eq(
        lrhoy.backward,
        lrhoy
        + model.vp**2 * lp,
        subdomain=grid.subdomains["main"]
    )

    eq_lrhoz = Eq(
        lrhoz.backward,
        lrhoz
        + model.vp**2 * lp,
        subdomain=grid.subdomains["main"]
    )

    # --------------------------------------------------
    # Residual injection into adjoint pressure
    # --------------------------------------------------

    adj_src = Receiver(
        name="adj_src",
        grid=grid,
        npoint=rec.npoint,
        time_range=rec.time_range
    )

    adj_src.coordinates.data[:] = rec.coordinates.data[:]

    residual = np.asarray(residual, dtype=np.float32)

    if residual.ndim == 1:
        residual = residual[:, None]

    adj_src.data[:] = 0.0
    adj_src.data[:nt, :] = residual[::-1, :]

    inj = adj_src.inject(
        field=lp.backward,
        expr=adj_src
    )

    # --------------------------------------------------
    # Gradient wrt vp
    #
    # p = vp^2 * (rhox + rhoy + rhoz)
    # d p / d vp = 2 vp * (rhox + rhoy + rhoz)
    # --------------------------------------------------

    grad_vp = Function(
        name="grad_vp",
        grid=grid
    )

    rhox_fwd = forward_fields["rhox"]
    rhoy_fwd = forward_fields["rhoy"]
    rhoz_fwd = forward_fields["rhoz"]

    grad_update = Inc(
        grad_vp,
        2.0 * model.vp *
        (rhox_fwd + rhoy_fwd + rhoz_fwd) *
        lp
    )

    # --------------------------------------------------
    # Build operator
    # --------------------------------------------------

    op = Operator(
        [
            eq_lp,
            eq_lvx,
            eq_lvy,
            eq_lvz,
            eq_lrhox,
            eq_lrhoy,
            eq_lrhoz,
            grad_update
        ] + inj
    )

    return op, grad_vp
