"""OpticsBox - transfer-matrix building blocks for multi-waveguide photonic circuits.

The same physics as the OpticsBox web app (https://unixpenguin.github.io/opticsbox/), as a Python library.

Conventions
    Fields at any cross-section: [E1+, E1-, E2+, E2-, ...]; + travels forward (left to right), - backward,
    and the number is the waveguide. Every component maps the fields on its left to those on its right:
    right = M @ left. A circuit is T = Bn @ ... @ B2 @ B1, in the order light meets the components.
    Forward propagation picks up exp(-j*beta*L); the directional coupler is [[tau, -j*kappa], [-j*kappa, tau]].
    Components have no length of their own unless a length is given.
    Phase follows beta(lambda): 2*pi*n_eff/lambda by default (constant index). Pass n_g (group index) and lam0_nm to
    any component with a length (waveguide, ring, add_drop, crow, mzi, ossb, bragg, loop_phase, Circuit.y_splitter) for first-order
    dispersion, beta = 2*pi*n_eff/lam0 + (2*pi*n_g/lam0^2)*(lam0 - lambda); use the same values everywhere.
    The same components take loss_db_per_cm, the propagation loss of the waveguide.
    Mirrors: mirror(R) takes the power reflectance (reflects j*sqrt(R)); mirror(r=...) a real field reflection r.
    PLATFORMS gives typical (n_eff, n_g, loss) for common chip materials, as in the app's chip-material menu.

Quick start
    from opticsbox import *

    c = Circuit(waveguides=2)                                   # bidirectional by default
    c.add(coupler(0.7071), at=1)                                # joins waveguides 1 and 2
    c.add(lambda w: lanes(wg(1000, w), wg(1100, w)), at=1)       # anything wavelength-dependent: a function of w (nm)
    c.add(coupler(0.7071), at=1)
    r = c.outputs(1550, inputs=[1, 0])
    print(r["out_power"], r["back_power"], r["lost_power"])     # fractions of the input power

    s = c.sweep(np.linspace(1545, 1555, 2001), inputs=[1, 0])   # same keys, one row per wavelength

Y-splitters get their own radiation channel (light lost at the junction):
    c.y_splitter(at=1, shape="split", make_room=True)           # the waveguides below shift down to make room

Run this file to self-test it against the web app.
"""
from __future__ import annotations

import warnings

import numpy as np

S2 = np.sqrt(0.5)


# =====================================================================================================
# Components. Each returns its LOCAL bidirectional matrix: 2x2 for one waveguide, 4x4 for two, and so on.
# Use forward_only(M) for the forward-only model (Circuit does this for you).
# =====================================================================================================
def beta(wavelength_nm, n_eff=2.4, n_g=None, lam0_nm=1550.0):
    """Propagation constant in rad/m, used by every component that has a length or a phase that follows the wavelength.

    n_g None (the default): constant index, beta = 2*pi*n_eff/lambda, so the group index equals n_eff. This is the
        original model, unchanged.
    n_g given: first-order dispersion about the reference wavelength lam0,
        beta = 2*pi*n_eff/lam0 + (2*pi*n_g/lam0**2) * (lam0 - lambda),
        so n_eff sets the phase at lam0 and n_g sets how fast it changes (resonance spacing lambda^2/(n_g L)).
        With n_g = n_eff this is the straight-line approximation of the constant-index model about lam0: equal
        at lam0, and off by about 2*pi*n_eff*(lambda - lam0)^2/lam0^3 elsewhere."""
    if n_g is None:
        return 2 * np.pi * n_eff / (wavelength_nm * 1e-9)
    if not lam0_nm > 0:
        raise ValueError("lam0_nm must be above 0")
    l0 = lam0_nm * 1e-9
    return 2 * np.pi * n_eff / l0 + 2 * np.pi * n_g / l0 ** 2 * (l0 - wavelength_nm * 1e-9)


def _alpha(loss_db_per_cm):
    """Power loss coefficient in 1/m from dB/cm."""
    return loss_db_per_cm * 100 * np.log(10) / 10


def _gain(length_um, wavelength_nm, n_eff, loss_db_per_cm, n_g=None, lam0_nm=1550.0):
    """Field after one pass of a waveguide section: exp(-(j*beta + alpha/2) * L)."""
    return np.exp(-(1j * beta(wavelength_nm, n_eff, n_g, lam0_nm) + _alpha(loss_db_per_cm) / 2) * length_um * 1e-6)


def waveguide(length_um, wavelength_nm, n_eff=2.4, loss_db_per_cm=0.0, n_g=None, lam0_nm=1550.0):
    """Waveguide section on one waveguide (2x2)."""
    g = _gain(length_um, wavelength_nm, n_eff, loss_db_per_cm, n_g, lam0_nm)
    return np.array([[g, 0], [0, 1 / g]], dtype=complex)


wg = waveguide


def lanes(*ms):
    """Block-diagonal: one 2x2 block per waveguide, e.g. lanes(wg(100, w), wg(120, w))."""
    T = np.zeros((2 * len(ms), 2 * len(ms)), complex)
    for i, m in enumerate(ms):
        T[2 * i:2 * i + 2, 2 * i:2 * i + 2] = m
    return T


pair = lanes                                    # name kept from the original two-waveguide version


def coupler_cs(tau, kappa):
    return np.array([[tau, 0, -1j * kappa, 0], [0, tau, 0, 1j * kappa],
                     [-1j * kappa, 0, tau, 0], [0, 1j * kappa, 0, tau]], dtype=complex)


def coupler(kappa):
    """Directional coupler on two waveguides; kappa = field coupling, kappa^2 of the power crosses."""
    return coupler_cs(np.sqrt(1 - kappa ** 2), kappa)


def coupler_wl(kappa0, lam0_nm, s_pct_per_nm, wavelength_nm):
    """Coupler whose coupling angle grows by s % per nm above the reference wavelength lam0."""
    p = np.arcsin(kappa0) * (1 + s_pct_per_nm / 100 * (wavelength_nm - lam0_nm))
    return coupler_cs(np.cos(p), np.sin(p))


def mmi(loss_db=0.0):
    """Fixed 50/50 multimode-interference splitter, with optional excess loss."""
    a = 10 ** (-loss_db / 20)
    return coupler(S2) * np.array([a, 1 / a, a, 1 / a])[:, None]


def crossing():
    """Swaps two neighbouring waveguides (ideal)."""
    M = np.zeros((4, 4), complex)
    M[0, 2] = M[1, 3] = M[2, 0] = M[3, 1] = 1
    return M


def phase(theta):
    """Extra phase theta (rad) on one waveguide, e.g. a heater."""
    return np.diag([np.exp(-1j * theta), np.exp(1j * theta)])


def atten(loss_db):
    """Loss element on one waveguide."""
    a = 10 ** (-loss_db / 20)
    return np.diag([a, 1 / a]).astype(complex)


def mirror(R=None, r=None):
    """Lossless partial mirror on one waveguide. Give one of:

    R: power reflectance, 0 <= R < 1 (the original form). Symmetric convention: the field reflection is
       j*sqrt(R) from either side, and the field transmission sqrt(1 - R).
    r: real field reflection coefficient, -1 < r < 1 (Fresnel / Stokes convention): it reflects r from the left
       and -r from the right, and transmits sqrt(1 - r^2) both ways. The power reflectance is R = r^2.
       Matrix: [[1, -r], [-r, 1]] / sqrt(1 - r^2)."""
    if (R is None) == (r is None):
        raise ValueError("mirror: give either R (power reflectance) or r (field reflection), not both")
    if r is not None:
        if not abs(r) < 1:
            raise ValueError("mirror: the field reflection r must be between -1 and 1 (|r| = 1 lets nothing through)")
        return np.array([[1, -r], [-r, 1]], dtype=complex) / np.sqrt(1 - r ** 2)
    if not 0 <= R < 1:
        raise ValueError("mirror: the power reflectance R must be at least 0 and below 1 (R = 1 lets nothing through)")
    t, rr = np.sqrt(1 - R), np.sqrt(R)
    return np.array([[1, 1j * rr], [-1j * rr, 1]]) / t


def edge(loss_db, R=0.0):
    """Chip edge / grating coupler on one waveguide: loss, then a mirror (reflection seen from the input side)."""
    a = 10 ** (-loss_db / 20)
    return np.diag([a, 1 / a]) @ mirror(R)


def slm(kappa, phi):
    """Sagnac loop mirror on one waveguide; undefined at kappa^2 = 0.5 (a perfect mirror).
    phi is the phase of one trip round the loop. It may be complex: loop_phase(..., loss_db_per_cm=...) gives
    (beta - j*alpha/2) * L, so a loop set by its length attenuates the light going round it."""
    t = np.sqrt(1 - kappa ** 2)
    m = 2j * kappa * t
    return np.array([[np.exp(-1j * phi), -m], [m, np.exp(1j * phi)]], dtype=complex) / (1 - 2 * kappa ** 2)


def slm_pair(kappa1, phi1, kappa2, phi2):
    """An SLM on each of two waveguides (name kept from the original version)."""
    return lanes(slm(kappa1, phi1), slm(kappa2, phi2))


def loop_phase(loop_um, wavelength_nm, n_eff=2.4, n_g=None, lam0_nm=1550.0, loss_db_per_cm=0.0):
    """Phase of an SLM loop set by its length: beta(lambda) * L (2*pi*n_eff*L/lambda without dispersion).
    With a waveguide loss it is complex, (beta - j*alpha/2) * L, so slm(kappa, loop_phase(...)) includes the
    loss of the loop, as the web app does."""
    ph = beta(wavelength_nm, n_eff, n_g, lam0_nm) * loop_um * 1e-6
    return ph - 0.5j * _alpha(loss_db_per_cm) * loop_um * 1e-6 if loss_db_per_cm else ph


def ring_ports(kappa, length_um, wavelength_nm, n_eff=2.4, loss_db_per_cm=0.0, backscatter=0.0, n_g=None, lam0_nm=1550.0):
    """All-pass ring: (t, rho) = (forward transmission, reflection back into the bus). The same from either side.

    Without backscatter: t = (tau - e) / (1 - tau e), rho = 0, with e = a e^(-j theta) one trip round the ring.
    backscatter = field reflection r_s of a lossless point scatterer half-way round the ring, coupling the
    clockwise and counter-clockwise waves (transmission t_s = sqrt(1 - r_s^2), reflection j r_s). It splits each
    resonance into two, at round-trip phases 2*pi*m +- asin(r_s). With
        q = 1 - tau t_s e,   G = t_s e - r_s^2 tau e^2 / q   (one trip, including the detour through the other direction)
        t = (tau - G) / (1 - tau G),   rho = -j kappa^2 r_s e / ((1 - tau G) q)."""
    if kappa == 0:                                   # the ring is not coupled to the waveguide
        return 1 + 0j, 0j
    tau = np.sqrt(1 - kappa ** 2)
    e = _gain(length_um, wavelength_nm, n_eff, loss_db_per_cm, n_g, lam0_nm)
    if not backscatter:
        return (tau - e) / (1 - tau * e), 0j
    rs = backscatter
    ts = np.sqrt(1 - rs ** 2)
    q = 1 - tau * ts * e
    G = ts * e - rs ** 2 * tau * e ** 2 / q
    den = 1 - tau * G
    return (tau - G) / den, -1j * kappa ** 2 * rs * e / (den * q)


def ring(kappa, length_um, wavelength_nm, n_eff=2.4, loss_db_per_cm=0.0, backscatter=0.0, n_g=None, lam0_nm=1550.0):
    """All-pass ring beside one waveguide (2x2): t = (tau - a e^-j theta) / (1 - tau a e^-j theta).
    With backscatter (see ring_ports) it also reflects rho, from either side: [[t^2 - rho^2, rho], [-rho, 1]] / t."""
    if not 0 <= backscatter < 1:
        raise ValueError("ring: backscatter (field reflection of the scatterer) must be at least 0 and below 1")
    t, rho = ring_ports(kappa, length_um, wavelength_nm, n_eff, loss_db_per_cm, backscatter, n_g, lam0_nm)
    if rho == 0:
        return np.diag([t, 1 / t])
    return np.array([[t * t - rho * rho, rho], [-rho, 1]]) / t


def add_drop(kappa1, kappa2, length_um, wavelength_nm, n_eff=2.4, loss_db_per_cm=0.0, n_g=None, lam0_nm=1550.0):
    """Ring between two waveguides. Dropped light leaves the other waveguide travelling BACKWARDS."""
    if kappa1 == 0 and kappa2 == 0:                  # the ring touches neither waveguide
        return np.eye(4, dtype=complex)
    t1, t2 = np.sqrt(1 - kappa1 ** 2), np.sqrt(1 - kappa2 ** 2)
    eh = _gain(length_um / 2, wavelength_nm, n_eff, loss_db_per_cm, n_g, lam0_nm)
    e = eh ** 2
    den = 1 - t1 * t2 * e
    t, tp, d = (t1 - t2 * e) / den, (t2 - t1 * e) / den, -kappa1 * kappa2 * eh / den
    return np.array([[t - d ** 2 / tp, 0, 0, d / tp], [0, 1 / t, -d / t, 0],
                     [0, d / t, tp - d ** 2 / t, 0], [-d / tp, 0, 0, 1 / tp]])


def crow_ports(n_rings, kappa1, kappa_rings, kappa2, length_um, wavelength_nm, n_eff=2.4, loss_db_per_cm=0.0, n_g=None, lam0_nm=1550.0):
    """CROW (n coupled rings in a chain between two waveguides): (t, t', d) = through on waveguide 1, through on
    waveguide 2, drop (the same both ways). Couplers kappa_0 = kappa1 (waveguide 1 - ring 1), kappa_rings between
    neighbouring rings, kappa_n = kappa2 (ring n - waveguide 2); h = half a trip round a ring. Looking up the chain
    from coupler i, G_n = 0, G_(i-1) = h^2 (tau_i - G_i) / (1 - tau_i G_i); t = (tau_0 - G_0) / (1 - tau_0 G_0);
    d = x_n with x_0 = -j kappa_0 / (1 - tau_0 G_0), x_i = -j kappa_i h x_(i-1) / (1 - tau_i G_i)."""
    n = int(n_rings)
    if n < 1:
        raise ValueError("crow: n_rings must be 1 or more")
    ks = [kappa1] + [kappa_rings] * (n - 1) + [kappa2]
    if not any(ks):
        return 1 + 0j, 1 + 0j, 0j
    h = _gain(length_um / 2, wavelength_nm, n_eff, loss_db_per_cm, n_g, lam0_nm)

    def chain_of(ks):
        tau = [np.sqrt(1 - k ** 2) for k in ks]
        G = [0j] * (n + 1)
        for i in range(n, 0, -1):
            G[i - 1] = h ** 2 * (tau[i] - G[i]) / (1 - tau[i] * G[i])
        x = -1j * ks[0] / (1 - tau[0] * G[0])
        for i in range(1, n + 1):
            x = -1j * ks[i] * h * x / (1 - tau[i] * G[i])
        return (tau[0] - G[0]) / (1 - tau[0] * G[0]), x

    t, d = chain_of(ks)
    return t, chain_of(ks[::-1])[0], d


def crow(n_rings, kappa1, kappa_rings, kappa2, length_um, wavelength_nm, n_eff=2.4, loss_db_per_cm=0.0, n_g=None, lam0_nm=1550.0):
    """Coupled-resonator optical waveguide: n identical rings in a chain between two waveguides (4x4). With an odd
    number of rings the dropped light leaves the other waveguide travelling BACKWARDS (as for add_drop, which is the
    one-ring case); with an even number it goes forwards."""
    t, tp, d = crow_ports(n_rings, kappa1, kappa_rings, kappa2, length_um, wavelength_nm, n_eff, loss_db_per_cm, n_g, lam0_nm)
    if int(n_rings) % 2:
        return np.array([[t - d ** 2 / tp, 0, 0, d / tp], [0, 1 / t, -d / t, 0],
                         [0, d / t, tp - d ** 2 / t, 0], [-d / tp, 0, 0, 1 / tp]])
    D = t * tp - d ** 2                             # forward [[t, d], [d, t']]; backward its inverse
    return np.array([[t, 0, d, 0], [0, tp / D, 0, -d / D], [d, 0, tp, 0], [0, -d / D, 0, t / D]])


def mzi(kappa1, kappa2, arm_a_um, arm_b_um, theta, wavelength_nm, n_eff=2.4, loss_db_per_cm=0.0, n_g=None, lam0_nm=1550.0):
    """Mach-Zehnder interferometer (4x4): coupler kappa1, arm a (waveguide 1, with a heater theta) and arm b, coupler kappa2."""
    arm = lambda L: waveguide(L, wavelength_nm, n_eff, loss_db_per_cm, n_g, lam0_nm)
    return coupler(kappa2) @ lanes(phase(theta) @ arm(arm_a_um), arm(arm_b_um)) @ coupler(kappa1)


def ossb(kappa, kappa_ring, ring_um, arm_a_um, arm_b_um, theta, wavelength_nm, n_eff=2.4, loss_db_per_cm=0.0, n_g=None, lam0_nm=1550.0):
    """OSSB filter: a ring-assisted Mach-Zehnder (4x4). Two couplers kappa; arm a has an all-pass ring (kappa_ring,
    ring_um) and a heater theta. The ring's steep phase gives one sharp edge, used to keep one sideband of a
    modulated carrier and block the other."""
    arm = lambda L: waveguide(L, wavelength_nm, n_eff, loss_db_per_cm, n_g, lam0_nm)
    rg = ring(kappa_ring, ring_um, wavelength_nm, n_eff, loss_db_per_cm, 0.0, n_g, lam0_nm)
    return coupler(kappa) @ lanes(rg @ phase(theta) @ arm(arm_a_um), arm(arm_b_um)) @ coupler(kappa)


# Chip materials: typical single-mode strip/ridge waveguides at 1550 nm, TE (the same values as the web app's
# "chip material" menu): name -> (n_eff, n_g, loss in dB/cm). For example:
#     n, ng, loss = PLATFORMS["sin"]; add_drop(0.2, 0.2, 200, w, n, loss, n_g=ng)
PLATFORMS = {"si": (2.44, 4.2, 2.0), "sin": (1.8, 2.05, 0.1), "inp": (3.2, 3.65, 2.0), "ln": (1.85, 2.3, 0.3), "sio2": (1.46, 1.48, 0.02)}


def bragg(lam_bragg_nm, length_um, kappa_per_mm, wavelength_nm, n_eff=2.4, loss_db_per_cm=0.0, n_g=None, lam0_nm=1550.0):
    """Uniform Bragg grating on one waveguide (coupled-mode theory). Lossless peak reflection tanh^2(kappa*L).

    The grating period is set so the reflection peaks at lam_bragg: beta_B = beta(lam_bragg) (= pi / period), and the
    detuning is delta = beta(lambda) - beta_B. Waveguide loss enters as beta -> beta - j*alpha/2, so with
    kappa = 0 the grating is exactly a lossy waveguide section of the same length."""
    bB = beta(lam_bragg_nm, n_eff, n_g, lam0_nm)
    d = beta(wavelength_nm, n_eff, n_g, lam0_nm) - bB - 0.5j * _alpha(loss_db_per_cm)
    k, L = kappa_per_mm * 1e3, length_um * 1e-6
    g = np.sqrt(complex(k ** 2 - d ** 2))
    C, S = np.cosh(g * L), (np.sinh(g * L) / g if abs(g) > 1e-9 else L)
    p, q = np.exp(-1j * bB * L), np.exp(1j * bB * L)
    return np.array([[p * (C - 1j * d * S), -1j * k * S * p], [1j * k * S * q, q * (C + 1j * d * S)]])


def y_matrix(m=3, stem=1, shape="split"):
    """Y-junction on (stem, branch, the waveguides below that shift..., radiation channel): m entries.

    m = 3 is the plain junction. With more entries the waveguides below the branch shift down one place
    when splitting (or up when combining). The matrix is real and orthogonal, so it is lossless."""
    if stem == 2:                                # stem on the lower waveguide; plain junction only
        U = np.array([[0, S2, -S2], [0, S2, S2], [1, 0, 0]]) if shape == "split" else np.array([[0, 0, 1], [S2, S2, 0], [-S2, S2, 0]])
    else:
        U, last = np.zeros((m, m)), m - 2
        if shape == "split":
            U[0, 0] = U[0, m - 1] = U[1, 0] = S2
            U[1, m - 1] = -S2
            for j in range(1, last):
                U[j + 1, j] = 1
            U[m - 1, last] = 1
        else:
            U[0, 0] = U[0, 1] = U[m - 1, 0] = S2
            U[m - 1, 1] = -S2
            for j in range(1, last):
                U[j, j + 1] = 1
            U[last, m - 1] = 1
    M = np.zeros((2 * len(U), 2 * len(U)))
    M[0::2, 0::2] = U
    M[1::2, 1::2] = U
    return M


def forward_only(M):
    """Forward-only version of a local bidirectional matrix: the forward transmission with nothing entering
    from the right and reflections discarded (the Schur complement M_ff - M_fb M_bb^-1 M_bf; 1/M22 for one
    waveguide)."""
    M = np.atleast_2d(M)
    f, b = slice(0, None, 2), slice(1, None, 2)
    Mff, Mfb, Mbf, Mbb = M[f, f], M[f, b], M[b, f], M[b, b]
    if not np.any(Mfb) and not np.any(Mbf):
        return Mff
    return Mff - Mfb @ np.linalg.solve(Mbb, Mbf)


# =====================================================================================================
# Circuits with any number of waveguides
# =====================================================================================================
class Circuit:
    """A circuit on `waveguides` waveguides. model = "bidirectional" (2N x 2N) or "forward" (N x N)."""

    def __init__(self, waveguides, model="bidirectional"):
        if not (isinstance(waveguides, (int, np.integer)) and waveguides >= 1):
            raise ValueError("Circuit: waveguides must be a whole number, 1 or more")
        if model not in ("bidirectional", "forward"):
            raise ValueError('Circuit: model must be "bidirectional" or "forward"')
        self.W, self.model, self.items, self.n_rad, self.rad_info = waveguides, model, [], 0, []

    def _lanes(self, M, at, label):
        """The waveguides a local matrix M sits on, after checking that M fits them and the circuit."""
        name = label or "component"
        if M.ndim != 2 or M.shape[0] != M.shape[1] or M.shape[0] % 2:
            raise ValueError(f"{name}: needs a square local bidirectional matrix, 2 rows per waveguide; got shape {M.shape}")
        n = M.shape[0] // 2
        if np.isscalar(at):
            ls = list(range(at, at + n))
        else:
            ls = [self.W + L[1] if isinstance(L, tuple) else L for L in at]
            if len(ls) != n:
                raise ValueError(f"{name}: its matrix covers {n} waveguides but {len(ls)} were given")
        if len(set(ls)) != len(ls) or any(not 1 <= L <= self.W + self.n_rad for L in ls) or \
                (np.isscalar(at) and ls[-1] > self.W):
            raise ValueError(f"{name}: needs waveguides {ls}, but the circuit has waveguides 1 to {self.W}")
        return ls

    def add(self, component, at, label=""):
        """component: a local bidirectional matrix, or a function of wavelength (nm) returning one.
        at: the first waveguide it sits on (int), or an explicit list of waveguides."""
        if not callable(component):
            self._lanes(np.atleast_2d(component), at, label)        # fail now, not at the first wavelength
        self.items.append((component, at, label))
        return self

    def y_splitter(self, at, stem=1, shape="split", make_room=True, branch_um=0.0, shift_um=0.0,
                   n_eff=2.4, loss_db_per_cm=0.0, n_g=None, lam0_nm=1550.0):
        """A Y-splitter on waveguides at and at+1, with its own radiation channel (reported as lost light).
        make_room: the waveguides below shift down (split) or up (combine). branch_um adds the path length of
        both branches; shift_um the extra length of every other waveguide that steps aside at the junction
        (half a spacing, above and below). With shift_um == branch_um all paths gain the same phase.
        A make-room split has nowhere to put light already on the bottom waveguide: it goes into the radiation
        channel, and outputs() warns when that happens."""
        if not (1 <= at and at + 1 <= self.W):
            raise ValueError(f"y_splitter: needs waveguides {at} and {at + 1}, but the circuit has waveguides 1 to {self.W}")
        if stem not in (1, 2) or shape not in ("split", "combine"):
            raise ValueError('y_splitter: stem must be 1 or 2, and shape "split" or "combine"')
        self.n_rad += 1
        rad = ("rad", self.n_rad)
        room = make_room and stem == 1
        self.rad_info.append({"at": at, "shape": shape, "make_room": room})
        ls = list(range(at, self.W + 1)) if room else [at, at + 1]
        branches = lambda w: lanes(wg(branch_um, w, n_eff, loss_db_per_cm, n_g, lam0_nm), wg(branch_um, w, n_eff, loss_db_per_cm, n_g, lam0_nm))
        if shape == "combine" and branch_um:
            self.add(branches, at, "Y branches")
        self.add(y_matrix(len(ls) + 1, stem, shape), ls + [rad], "Y-splitter")
        if shape == "split" and branch_um:
            self.add(branches, at, "Y branches")
        if shift_um:      # every waveguide that steps aside at the junction (as the app draws it): above, and below
            runs = [(1, at - 1)]
            if room:
                runs.append((at + 2, self.W) if shape == "split" else (at + 1, self.W - 1))
            else:
                runs.append((at + 2, self.W))
            for first, last in runs:
                if last >= first:
                    self.add(lambda w, k=last - first + 1: lanes(*[wg(shift_um, w, n_eff, loss_db_per_cm, n_g, lam0_nm)] * k), first, "shifted waveguides")
        return self

    def total(self, wavelength_nm):
        """The circuit matrix: 2N x 2N (bidirectional) or N x N (forward), N = waveguides + radiation channels."""
        per = 1 if self.model == "forward" else 2
        N = self.W + self.n_rad
        T = np.eye(per * N, dtype=complex)
        for comp, at, label in self.items:
            M = np.atleast_2d(comp(wavelength_nm) if callable(comp) else comp)
            ls = self._lanes(M, at, label)
            if per == 1:
                M = forward_only(M)
            ix = [per * (L - 1) + d for L in ls for d in range(per)]
            E = np.eye(per * N, dtype=complex)
            E[np.ix_(ix, ix)] = M
            T = E @ T
        return T

    def outputs(self, wavelength_nm, inputs):
        """inputs: field amplitude into each waveguide on the left. Nothing enters from the right.
        Returns complex 'out' (right end) and 'back' (reflected, left end) per waveguide, and the powers as
        fractions of the input: out_power, back_power, lost_power (one per Y-splitter)."""
        N, W = self.W + self.n_rad, self.W
        inputs = np.asarray(inputs, dtype=complex).ravel()
        if inputs.size != W:
            raise ValueError(f"outputs: give one input amplitude per waveguide ({W}), got {inputs.size}")
        if not np.any(inputs):
            raise ValueError("outputs: every input amplitude is 0, so there is no light to follow; put light into at least one waveguide")
        A = np.zeros(N, complex)
        A[:W] = inputs
        T = self.total(wavelength_nm)
        if self.model == "forward":
            out, back = T @ A, np.zeros(N, complex)
        else:
            f, b = list(range(0, 2 * N, 2)), list(range(1, 2 * N, 2))
            back = np.linalg.solve(T[np.ix_(b, b)], -(T[np.ix_(b, f)] @ A))
            left = np.zeros(2 * N, complex)
            left[f], left[b] = A, back
            out = (T @ left)[f]
        Pin = np.sum(np.abs(A) ** 2)
        lost = (np.abs(out[W:]) ** 2 + np.abs(back[W:]) ** 2) / Pin
        for k, info in enumerate(self.rad_info):         # forward light in the radiation channel of a make-room split
            if info["make_room"] and info["shape"] == "split" and np.abs(out[W + k]) ** 2 / Pin > 1e-9:   # = the bottom waveguide's
                warnings.warn(f"Y-splitter {k + 1} (make_room, at waveguide {info['at']}) is losing light: light on the bottom "
                              f"waveguide {W} has no waveguide below to move to, so it goes into the radiation channel "
                              f"(lost_power). Add a waveguide if it should keep going.", RuntimeWarning, stacklevel=2)
        return {"out": out[:W], "back": back[:W],
                "out_power": np.abs(out[:W]) ** 2 / Pin, "back_power": np.abs(back[:W]) ** 2 / Pin,
                "lost_power": lost}

    def sweep(self, wavelengths_nm, inputs):
        rows = [self.outputs(w, inputs) for w in wavelengths_nm]
        return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def db(x):
    """Power fraction (or complex amplitude) in dB."""
    x = np.asarray(x)
    p = np.abs(x) ** 2 if np.iscomplexobj(x) else x
    return 10 * np.log10(np.maximum(p, 1e-30))


# =====================================================================================================
# The original two-waveguide interface, kept so older scripts still run
# =====================================================================================================
def waveguide_pair(l1_um, l2_um, wavelength_nm, n_eff=2.4, loss_db_per_cm=0.0, n_g=None, lam0_nm=1550.0):
    return lanes(waveguide(l1_um, wavelength_nm, n_eff, loss_db_per_cm, n_g, lam0_nm),
                 waveguide(l2_um, wavelength_nm, n_eff, loss_db_per_cm, n_g, lam0_nm))


def chain(blocks):
    """Total 4x4 matrix of two-waveguide blocks, in the order light meets them."""
    T = np.eye(4, dtype=complex)
    for B in blocks:
        T = B @ T
    return T


def ports(T, into=1):
    """Light (amplitude 1) into waveguide `into` from the left of a two-waveguide circuit T."""
    a = np.zeros(4, complex)
    a[0 if into == 1 else 2] = 1
    A = np.array([[T[1, 1], T[1, 3]], [T[3, 1], T[3, 3]]])
    b1, b2 = np.linalg.solve(A, -np.array([T[1] @ a, T[3] @ a]))
    right = T @ (a + np.array([0, b1, 0, b2]))
    o1, o2 = right[0], right[2]
    same, other = (o1, o2) if into == 1 else (o2, o1)
    rs, ro = (b1, b2) if into == 1 else (b2, b1)
    return {"through": same, "cross": other, "reflect": rs, "reflect_cross": ro,
            "energy": abs(o1) ** 2 + abs(o2) ** 2 + abs(b1) ** 2 + abs(b2) ** 2}


def sweep(build, wavelengths_nm, into=1):
    """build(wavelength_nm) -> list of two-waveguide blocks."""
    rows = [ports(chain(build(w)), into) for w in wavelengths_nm]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# =====================================================================================================
# Self-test: physics checks, and agreement with numbers computed by the OpticsBox web app
# =====================================================================================================
if __name__ == "__main__":
    results = []

    def ok(cond, msg):
        results.append(bool(cond))
        print(("ok    " if cond else "FAIL  ") + msg)

    # 1. the original two-waveguide example (same numbers as the app's "Notes" preset)
    p = ports(chain([coupler(0.5), slm_pair(0.3, 0.7, 0.2, 1.9), coupler(0.4)]))
    ok(np.allclose([abs(p[k]) ** 2 for k in ("through", "cross", "reflect", "reflect_cross")],
                   [0.37067804, 0.34522196, 0.1633935, 0.1207065], atol=1e-8), "two-waveguide example matches the app")

    # 2. every lossless component conserves energy inside an interferometer, with light into either waveguide
    for name, part, at in [("ring", lambda w: ring(0.2, 200, w), 1), ("mirror", mirror(0.3), 2), ("Bragg grating", lambda w: bragg(1550, 500, 2, w), 1),
                           ("MMI", mmi(), 1), ("crossing", crossing(), 1), ("add-drop ring", lambda w: add_drop(0.2, 0.3, 200, w), 1),
                           ("SLM", slm(0.3, 0.7), 2), ("phase shifter", phase(1.2), 1), ("coupler (lambda)", lambda w: coupler_wl(0.7, 1550, 0.3, w), 1),
                           ("chip edge (0 dB)", lanes(edge(0, 0.3), edge(0, 0.3)), 1), ("MZI", lambda w: mzi(0.6, 0.7071, 300, 350, 0.4, w), 1),
                           ("OSSB filter", lambda w: ossb(0.7071, 0.8, 200, 0, 100, 1.5708, w), 1), ("CROW, 3 rings", lambda w: crow(3, 0.6, 0.2, 0.6, 200, w), 1),
                           ("CROW, 4 rings", lambda w: crow(4, 0.5, 0.25, 0.4, 170, w), 1)]:
        c = Circuit(2).add(coupler(0.5), 1).add(part, at).add(lambda w: lanes(wg(100, w), wg(137, w)), 1).add(part, at).add(coupler(0.4), 1)
        E = [np.sum(r["out_power"]) + np.sum(r["back_power"]) for w in np.linspace(1546, 1554, 9) for r in (c.outputs(w, [1, 0]), c.outputs(w, [0, 1]))]
        ok(np.allclose(E, 1, atol=1e-12), f"energy conserved with {name}")

    # 3. standard results
    r = Circuit(1).add(lambda w: bragg(1550, 500, 2, w), 1).outputs(1550, [1])
    ok(np.isclose(r["back_power"][0], np.tanh(1) ** 2), "Bragg grating peak reflection = tanh^2(kappa L)")
    ad = add_drop(0.2, 0.3, 200, 1550.3)
    ok(np.allclose(forward_only(ad), np.diag(1 / np.diag(ad[1::2, 1::2]))), "forward-only add-drop ring = its through transmissions")

    # 4. Y-splitter that makes room: the waveguide below keeps going, nothing lost
    r = Circuit(3).add(coupler(0.5), 1).y_splitter(at=1, make_room=True).outputs(1550, [1, 0, 0])
    ok(np.allclose(r["out_power"], [0.375, 0.375, 0.25]) and np.allclose(r["lost_power"], 0), "Y-splitter makes room: 37.5 / 37.5 / 25 %, nothing lost")

    # 5. 4-waveguide circuits with two Y-splitters, checked against numbers from the web app
    A = [1, 0.7 * np.exp(0.4j), 0.5 * np.exp(1.1j), 0]
    for model, lengths, ref in [("bidirectional", (0, 0, 0, 0), [0.18089258, 0.00349098, 0.04152742, 0, 0, 0.63948762]),
                                ("forward", (0, 0, 0, 0), [0.18089258, 0.00349098, 0.04152742, 0, 0, 0.63948762]),
                                ("bidirectional", (25, 9, 15, 4), [0.18118855, 0.26525338, 0.04159153, 0, 0, 0.22584446])]:
        c = Circuit(4, model)
        c.add(coupler(0.5), 2)
        if lengths == (0, 0, 0, 0):          # the app's version without Y path lengths has a waveguide section here
            c.y_splitter(at=1, shape="split", make_room=True)
            c.add(lambda w: lanes(wg(100, w, 2.4, 1), wg(130, w, 2.4, 1), wg(100, w, 2.4, 1), wg(80, w, 2.4, 1)), 1)
        else:
            c.y_splitter(at=1, shape="split", make_room=True, branch_um=lengths[0], shift_um=lengths[1], loss_db_per_cm=1)
        c.add(coupler(0.6), 2)
        c.y_splitter(at=2, shape="combine", make_room=True, branch_um=lengths[2], shift_um=lengths[3], loss_db_per_cm=1)
        c.add(lanes(*[edge(2, 0)] * 4), 1)
        r = c.outputs(1550, A)
        ok(np.allclose(np.concatenate([r["out_power"], r["lost_power"]]), ref, atol=1e-8),
           f"4-waveguide circuit matches the app ({model}{', with Y path lengths' if any(lengths) else ''})")

    # 6. dispersion: beta(lambda) = 2 pi n_eff/lam0 + (2 pi n_g/lam0^2)(lam0 - lambda)
    ne, ng, l0, L = 2.4, 4.2, 1550.0, 200.0
    ok(beta(1551.3, ne) == 2 * np.pi * ne / 1551.3e-9 and np.allclose(ring(0.2, 200, 1551.3), ring(0.2, 200, 1551.3, n_g=None)),
       "no group index: the constant-index model, unchanged")
    b0, b1 = beta(l0, ne, ng, l0), beta(l0 + 1, ne, ng, l0)
    ok(np.isclose(b0, 2 * np.pi * ne / (l0 * 1e-9), rtol=1e-14) and np.isclose(b0 - b1, 2 * np.pi * ng / (l0 * 1e-9) ** 2 * 1e-9, rtol=1e-9),
       "beta(lam0) = 2 pi n_eff/lam0, slope -2 pi n_g/lam0^2")
    h = 1e-3                              # n_g = n_eff: same value and slope as the constant-index model at lam0
    ok(np.isclose(beta(l0, ne, ne, l0), beta(l0, ne)) and
       np.isclose((beta(l0 + h, ne, ne, l0) - beta(l0 - h, ne, ne, l0)), (beta(l0 + h, ne) - beta(l0 - h, ne)), rtol=1e-6),
       "n_g = n_eff: agrees with the constant-index model to first order about lam0")

    def minima(f, lo, hi, n=40001):       # wavelengths of the dips of f, refined by a parabola through 3 points
        x = np.linspace(lo, hi, n); y = np.array([f(w) for w in x]); dx = x[1] - x[0]
        i = [k for k in range(1, n - 1) if y[k] < y[k - 1] and y[k] <= y[k + 1]]
        return np.array([x[k] + dx * (y[k - 1] - y[k + 1]) / (2 * (y[k - 1] - 2 * y[k] + y[k + 1])) for k in i])

    fsr = l0 ** 2 / (ng * L * 1e3)        # nm
    dips = minima(lambda w: abs(ring(0.2, L, w, ne, 3.0, n_g=ng, lam0_nm=l0)[0, 0]) ** 2, l0 - 1.5 * fsr, l0 + 1.5 * fsr)
    ok(len(dips) >= 2 and np.allclose(np.diff(dips), fsr, rtol=1e-6), f"ring FSR = lam0^2/(n_g L) = {fsr:.5f} nm (found {np.diff(dips).round(5)})")
    dips_nd = minima(lambda w: abs(ring(0.2, L, w, ne, 3.0)[0, 0]) ** 2, l0 - 1.5 * fsr, l0 + 1.5 * fsr)
    ok(np.allclose(np.diff(dips_nd), l0 ** 2 / (ne * L * 1e3), rtol=2e-3), "no group index: ring FSR = lam^2/(n_eff L)")
    ad = Circuit(2).add(lambda w: add_drop(0.2, 0.2, L, w, ne, 3.0, n_g=ng, lam0_nm=l0), 1)
    dips = minima(lambda w: ad.outputs(w, [1, 0])["out_power"][0], l0 - 1.5 * fsr, l0 + 1.5 * fsr)
    ok(np.allclose(np.diff(dips), fsr, rtol=1e-6), "add-drop ring FSR = lam0^2/(n_g L)")

    # 7. Fabry-Perot from two mirrors: the textbook Airy transmission and FSR = lam0^2/(2 n_g L)
    Lc = 500.0
    for conv, m1, m2, R1, R2, inside in [("field r", mirror(r=0.8), mirror(r=0.6), 0.64, 0.36, -0.8 * 0.6),
                                         ("field r, second mirror reversed", mirror(r=0.8), mirror(r=-0.6), 0.64, 0.36, 0.8 * 0.6),
                                         ("power R", mirror(0.64), mirror(0.36), 0.64, 0.36, (0.8j) * (0.6j))]:
        for loss in (0.0, 5.0):
            fp = Circuit(1).add(m1, 1).add(lambda w: wg(Lc, w, ne, loss, n_g=ng, lam0_nm=l0), 1).add(m2, 1)
            ws = np.linspace(l0 - 1, l0 + 1, 801)
            a2 = np.exp(-_alpha(loss) * Lc * 1e-6)                      # power left after one pass
            dlt = 2 * beta(ws, ne, ng, l0) * Lc * 1e-6 - np.angle(inside)  # round-trip phase, including both mirrors
            rr = np.sqrt(R1 * R2) * a2
            airy = (1 - R1) * (1 - R2) * a2 / ((1 - rr) ** 2 + 4 * rr * np.sin(dlt / 2) ** 2)
            T = np.array([fp.outputs(w, [1])["out_power"][0] for w in ws])
            ok(np.allclose(T, airy, rtol=1e-10, atol=1e-13), f"Fabry-Perot = Airy formula ({conv}, loss {loss} dB/cm)")
    fp = Circuit(1).add(mirror(r=0.9), 1).add(lambda w: wg(Lc, w, ne, 0, n_g=ng, lam0_nm=l0), 1).add(mirror(r=-0.9), 1)
    pk = minima(lambda w: -fp.outputs(w, [1])["out_power"][0], l0 - 0.5, l0 + 0.5)
    ok(np.allclose(np.diff(pk), l0 ** 2 / (2 * ng * Lc * 1e3), rtol=1e-6), f"Fabry-Perot FSR = lam0^2/(2 n_g L) = {l0 ** 2 / (2 * ng * Lc * 1e3):.5f} nm")
    ok(np.allclose(mirror(r=0.3) @ mirror(r=-0.3), np.eye(2)) and np.isclose(Circuit(1).add(mirror(r=-0.7), 1).outputs(1550, [1])["back"][0], -0.7)
       and np.isclose(Circuit(1).add(mirror(0.49), 1).outputs(1550, [1])["back"][0], 0.7j), "mirror reflects r (field r) or j sqrt(R) (power R)")

    # 8. ring backscattering: 0 changes nothing; otherwise lossless, reciprocal, and each resonance splits by asin(r_s)/pi FSR
    ok(np.allclose(ring(0.2, L, 1550.2, backscatter=0), ring(0.2, L, 1550.2)), "ring with no backscatter = the plain ring")
    E = []
    for w in np.linspace(1549, 1551, 41):
        c = Circuit(2).add(coupler(0.5), 1).add(lambda w: ring(0.3, L, w, backscatter=0.05), 1).add(coupler(0.4), 1)
        E += [np.sum(r["out_power"]) + np.sum(r["back_power"]) for r in (c.outputs(w, [1, 0]), c.outputs(w, [0, 1]))]
    ok(np.allclose(E, 1, atol=1e-12), "energy conserved with a backscattering ring")
    rs = 0.05
    c = Circuit(1).add(lambda w: ring(0.02, L, w, ne, 0, backscatter=rs, n_g=ng, lam0_nm=l0), 1)   # weak coupling: the bus
    m0 = np.round(beta(l0, ne, ng, l0) * L * 1e-6 / (2 * np.pi))                                  # barely pulls the peaks
    lam_m = l0 + (beta(l0, ne, ng, l0) * L * 1e-6 - 2 * np.pi * m0) / (2 * np.pi) * fsr          # unsplit resonance
    pk = minima(lambda w: -c.outputs(w, [1])["back_power"][0], lam_m - 0.02 * fsr, lam_m + 0.02 * fsr)
    ok(len(pk) == 2 and np.isclose(pk[1] - pk[0], np.arcsin(rs) / np.pi * fsr, rtol=1e-4),
       f"backscatter splits the resonance by asin(r_s)/pi * FSR = {np.arcsin(rs) / np.pi * fsr * 1e3:.3f} pm")

    # 9. Bragg grating: loss, dispersion and edge cases
    ok(np.allclose(bragg(1550, 500, 0, 1550.7, ne, 4.0), wg(500, 1550.7, ne, 4.0)), "Bragg grating with kappa = 0 = a lossy waveguide")
    r = Circuit(1).add(lambda w: bragg(1550, 500, 2, w, ne, 4.0), 1).outputs(1550, [1])
    ok(r["back_power"][0] < np.tanh(1) ** 2 and r["out_power"][0] + r["back_power"][0] < 1, "Bragg grating uses the waveguide loss")
    kL, Lg = 2.0, 1000.0
    null = minima(lambda w: Circuit(1).add(bragg(1550, Lg, kL / Lg * 1e3, w, ne, 0, n_g=ng, lam0_nm=l0), 1).outputs(w, [1])["back_power"][0], 1550.01, 1551.5)[0]
    ok(np.isclose(null - 1550, l0 ** 2 / (2 * np.pi * ng * Lg * 1e3) * np.sqrt(kL ** 2 + np.pi ** 2), rtol=1e-5),
       "Bragg first null at lam0^2/(2 pi n_g L) sqrt((kappa L)^2 + pi^2) from the peak")
    ok(np.isclose(Circuit(1).add(lambda w: bragg(1550, 500, 2, w, ne, 0, n_g=ng, lam0_nm=l0), 1).outputs(1550, [1])["back_power"][0], np.tanh(1) ** 2),
       "Bragg peak still tanh^2(kappa L) at the Bragg wavelength with dispersion")

    # 10. edge cases and validation
    lam_res = 2.4 * 200e3 / 310           # nm: exactly 310 wavelengths round a 200 um ring
    ok(np.allclose(ring(0, 200, lam_res), np.eye(2)) and np.allclose(ring(0, 200, 1550), np.eye(2)), "ring with kappa = 0 passes light (also on an exact resonance)")
    ok(np.allclose(add_drop(0, 0, 200, lam_res), np.eye(4)), "add-drop ring with both kappa = 0 passes light (also on resonance)")
    ok(np.allclose(mirror(0), np.eye(2)) and np.allclose(mirror(r=0), np.eye(2)), "mirror with no reflection = nothing")

    def raises(f):
        try:
            f()
        except ValueError:
            return True
        return False
    ok(raises(lambda: Circuit(2).outputs(1550, [0, 0])), "all-zero input is refused before solving")
    ok(raises(lambda: Circuit(2).add(coupler(0.5), 2)) and raises(lambda: Circuit(2).add(lambda w: coupler(0.5), 2).outputs(1550, [1, 0]))
       and raises(lambda: Circuit(1).y_splitter(1)) and raises(lambda: Circuit(2).add(np.eye(3), 1))
       and raises(lambda: Circuit(2).add(coupler(0.5), [1])), "components that don't fit the circuit are refused")
    ok(raises(lambda: mirror(1.0)) and raises(lambda: mirror(r=1.0)) and raises(lambda: mirror(0.3, r=0.3))
       and raises(lambda: beta(1550, 2.4, 4.2, 0)) and raises(lambda: Circuit(2).outputs(1550, [1])), "invalid mirror, lam0 and inputs are refused")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Circuit(3).y_splitter(at=1, make_room=True).outputs(1550, [1, 0, 0])
        quiet = not caught
        r = Circuit(3).y_splitter(at=1, make_room=True).outputs(1550, [1, 0, 1])
    ok(quiet and len(caught) == 1 and np.isclose(r["lost_power"][0], 0.5), "make-room Y-splitter warns when the bottom waveguide's light is lost")

    # 11. independent derivations. Nothing below reuses the closed formulas above: every resonant component is rebuilt
    #     from couplers, lengths of waveguide and point reflectors, and the equations for its internal waves are solved
    #     numerically. S is a scattering matrix: out = S @ in, with in = [E+ entering on the left, E- entering on the
    #     right] and out = [E- leaving on the left, E+ leaving on the right].
    def to_S(M):
        M = np.asarray(M, complex); f, b = slice(0, None, 2), slice(1, None, 2); inv = np.linalg.inv(M[b, b])
        return np.block([[-inv @ M[b, f], inv], [M[f, f] - M[f, b] @ inv @ M[b, f], M[f, b] @ inv]])

    def from_S(S):
        n = len(S) // 2; inv = np.linalg.inv(S[:n, n:]); M = np.zeros((2 * n, 2 * n), complex)
        M[1::2, 0::2], M[1::2, 1::2] = -inv @ S[:n, :n], inv
        M[0::2, 0::2], M[0::2, 1::2] = S[n:, :n] - S[n:, n:] @ inv @ S[:n, :n], S[n:, n:] @ inv
        return M

    def solve_waves(eqs, n, n_in):     # eqs(x, a) = 0, linear in the unknown waves x and the inputs a; returns X with x = X @ a
        P = np.array([eqs(e, np.zeros(n_in)) for e in np.eye(n, dtype=complex)]).T
        Q = np.array([eqs(np.zeros(n), e) for e in np.eye(n_in, dtype=complex)]).T
        return -np.linalg.solve(P, Q)

    dc = lambda k: np.array([[np.sqrt(1 - k * k), -1j * k], [-1j * k, np.sqrt(1 - k * k)]])    # a coupler, forward fields

    def slm_direct(k, g):              # a coupler whose two far ports are joined by a loop with one-trip factor g
        C = dc(k)
        def eqs(x, a):
            u1, u2, o1, o2 = x         # waves sent into the two ends of the loop; waves leaving on the left and on the right
            return np.array([u1 - C[0] @ a, u2 - C[1] @ a, o1 - g * (C[0, 0] * u2 + C[0, 1] * u1), o2 - g * (C[1, 0] * u2 + C[1, 1] * u1)])
        return from_S(solve_waves(eqs, 4, 2)[2:])

    ok(all(np.allclose(slm_direct(k, np.exp(-1j * ph)), slm(k, ph)) for k, ph in [(0.3, 0.7), (0.2, 1.9), (0.6, 0.0), (0.9, -2.2)]),
       "Sagnac loop mirror = a coupler closed by a loop, solved for its internal waves")
    ph = loop_phase(80, 1550.3, 2.4, None, 1550.0, 30.0)
    ok(np.allclose(slm_direct(0.3, _gain(80, 1550.3, 2.4, 30.0)), slm(0.3, ph)) and np.isclose(abs(forward_only(slm(0.3, ph))[0, 0]), (1 - 2 * 0.09) * abs(_gain(80, 1550.3, 2.4, 30.0)))
       and loop_phase(80, 1550.3) == beta(1550.3) * 80 * 1e-6, "an SLM loop set by its length includes the waveguide loss (no loss: a plain real phase)")

    def ring_direct(k, L_um, w, loss, rs):          # coupler + two half rings + a point reflector (j rs, sqrt(1 - rs^2))
        C = dc(k); h = _gain(L_um / 2, w, 2.4, loss); ts = np.sqrt(1 - rs * rs)
        def eqs(x, a):                 # a = [forward in on the left, backward in on the right]; cw goes with forward, ccw with backward
            cw_o, ccw_o, cw_i, ccw_i, o_right, o_left = x
            return np.array([o_right - (C[0, 0] * a[0] + C[0, 1] * cw_i), cw_o - (C[1, 0] * a[0] + C[1, 1] * cw_i),
                             o_left - (C[0, 0] * a[1] + C[0, 1] * ccw_i), ccw_o - (C[1, 0] * a[1] + C[1, 1] * ccw_i),
                             cw_i - h * h * (ts * cw_o + 1j * rs * ccw_o), ccw_i - h * h * (ts * ccw_o + 1j * rs * cw_o)])
        X = solve_waves(eqs, 6, 2)
        return from_S(np.array([X[5], X[4]]))

    ok(all(np.allclose(ring_direct(k, L_, w, lo, rs), ring(k, L_, w, 2.4, lo, backscatter=rs), rtol=1e-9, atol=1e-9)
           for k, L_, w, lo, rs in [(0.2, 200, 1550.3, 0, 0), (0.2, 200, 1550.3, 3, 0), (0.3, 200, 1549.1, 0, 0.05), (0.5, 137, 1551.7, 4, 0.2), (0.05, 400, 1550.01, 1, 0.01)]),
       "ring (with loss and backscatter) = a coupler, two half rings and a point reflector, solved for its internal waves")

    def add_drop_direct(k1, k2, L_um, w, loss):
        C1, C2 = dc(k1), dc(k2); h = _gain(L_um / 2, w, 2.4, loss)
        def eqs(x, a):                 # a = [E1+ left, E1- right, E2+ left, E2- right]; c, d = the two ways round the ring
            o1L, o1R, o2L, o2R, c1, c2, d1, d2 = x
            return np.array([o1R - (C1[0, 0] * a[0] + C1[0, 1] * h * c2), c1 - (C1[1, 0] * a[0] + C1[1, 1] * h * c2),
                             o2L - (C2[0, 0] * a[3] + C2[0, 1] * h * c1), c2 - (C2[1, 0] * a[3] + C2[1, 1] * h * c1),
                             o1L - (C1[0, 0] * a[1] + C1[0, 1] * h * d2), d1 - (C1[1, 0] * a[1] + C1[1, 1] * h * d2),
                             o2R - (C2[0, 0] * a[2] + C2[0, 1] * h * d1), d2 - (C2[1, 0] * a[2] + C2[1, 1] * h * d1)])
        X = solve_waves(eqs, 8, 4)[:4]
        return from_S(X[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])])

    ok(all(np.allclose(add_drop_direct(k1, k2, L_, w, lo), add_drop(k1, k2, L_, w, 2.4, lo), rtol=1e-9, atol=1e-9)
           for k1, k2, L_, w, lo in [(0.2, 0.2, 200, 1550.3, 0), (0.2, 0.3, 200, 1549.2, 2), (0.6, 0.1, 317, 1551.1, 5), (0.2, 0.0, 200, 1550.4, 1), (0.0, 0.3, 200, 1550.4, 1)]),
       "add-drop ring = two couplers and two half rings, solved for its internal waves (all four inputs)")

    def crow_direct(n, k1, kr, k2, L_um, w, loss):  # couplers and half rings only; both ways round, all four inputs
        ks = [k1] + [kr] * (n - 1) + [k2]; h = _gain(L_um / 2, w, 2.4, loss)
        def eqs(x, a):                 # one family of waves: B_i leaves coupler i-1 into ring i, T_i leaves coupler i into ring i
            B, T, o1, o2 = x[:n], x[n:2 * n], x[2 * n], x[2 * n + 1]; e = []
            for i in range(n + 1):
                C = dc(ks[i])
                lo_in = a[0] if i == 0 else h * B[i - 1]
                up_in = a[1] if i == n else h * T[i]
                lo_out, up_out = C @ np.array([lo_in, up_in])
                e.append((o1 if i == 0 else T[i - 1]) - lo_out)
                e.append((o2 if i == n else B[i]) - up_out)
            return np.array(e)
        X = solve_waves(eqs, 2 * n + 2, 2)[2 * n:]          # [bus 1 out, bus 2 out] from [bus 1 in, bus 2 in]
        S = np.zeros((4, 4), complex)       # in: E1+ left, E2+ left, E1- right, E2- right; out: E1- left, E2- left, E1+ right, E2+ right
        fams = [((0, 3), (2, 1)), ((2, 1), (0, 3))] if n % 2 else [((0, 1), (2, 3)), ((2, 3), (0, 1))]
        for ins, outs in fams:
            for p in range(2):
                for q in range(2):
                    S[outs[p], ins[q]] = X[p, q]
        return from_S(S)

    ok(all(np.allclose(crow_direct(n, k1, kr, k2, L_, w, lo), crow(n, k1, kr, k2, L_, w, 2.4, lo), rtol=1e-9, atol=1e-9)
           for n, k1, kr, k2, L_, w, lo in [(1, 0.2, 0.5, 0.3, 200, 1550.3, 0), (2, 0.6, 0.2, 0.5, 200, 1550.1, 0), (3, 0.6, 0.2, 0.6, 180, 1549.7, 3),
                                             (4, 0.5, 0.3, 0.4, 150, 1551.2, 1), (5, 0.7, 0.15, 0.7, 210, 1550.02, 0.5)])
       and np.allclose(crow(1, 0.2, 0.9, 0.3, 200, 1550.3, 2.4, 2.0), add_drop(0.2, 0.3, 200, 1550.3, 2.4, 2.0)),
       "CROW (1 to 5 rings) = couplers and half rings, solved for its internal waves; one ring = the add-drop ring")
    ww = 1550.2; arm = lambda L: wg(L, ww, 2.4, 2.0)
    ok(np.allclose(mzi(0.6, 0.7, 300, 350, 0.4, ww, 2.4, 2.0), Circuit(2).add(coupler(0.6), 1).add(phase(0.4), 1).add(lanes(arm(300), arm(350)), 1).add(coupler(0.7), 1).total(ww))
       and np.allclose(ossb(0.7, 0.8, 200, 10, 100, 1.5, ww, 2.4, 2.0), Circuit(2).add(coupler(0.7), 1).add(ring(0.8, 200, ww, 2.4, 2.0), 1).add(phase(1.5), 1)
                       .add(lanes(arm(10), arm(100)), 1).add(coupler(0.7), 1).total(ww)), "MZI and OSSB filter = their couplers, ring, heater and arms placed one by one")
    for mdl in ("bidirectional", "forward"):
        c = Circuit(2, mdl)
        for part in (lambda w: crow(3, 0.6, 0.2, 0.6, 200, w, 2.4, 1.0, 4.2), lambda w: mzi(0.6, 0.7071, 300, 350, 0.4, w, 2.4, 1.0, 4.2),
                     lambda w: ossb(0.7071, 0.8, 200, 0, 100, 1.5708, w, 2.4, 1.0, 4.2), lambda w: crow(2, 0.5, 0.3, 0.4, 150, w, 2.4, 1.0, 4.2)):
            c.add(part, 1)
        r = c.outputs(1550.13, [1, 0])
        ok(np.allclose(list(r["out_power"]) + list(r["back_power"]), [0.68829962977, 0.301418618502] + ([0, 8.76877902797e-06] if mdl == "bidirectional" else [0, 0]), rtol=1e-9, atol=1e-15),
           f"CROW, MZI and OSSB filter: same powers as the app ({mdl})")

    def stack_reflection(n0, dn, periods, lam_b, w):      # quarter-wave layers n0 + dn, n0 - dn between n0: a real thin-film Bragg mirror
        step = lambda n1, n2: 0.5 * np.array([[1 + n1 / n2, 1 - n1 / n2], [1 - n1 / n2, 1 + n1 / n2]])     # E and H continuous
        M, n_prev = np.eye(2, dtype=complex), n0
        for _ in range(periods):
            for n in (n0 + dn, n0 - dn):
                phi = 2 * np.pi * n * (lam_b / (4 * n)) / w
                M = np.diag([np.exp(-1j * phi), np.exp(1j * phi)]) @ step(n_prev, n) @ M; n_prev = n
        M = step(n_prev, n0) @ M
        return abs(M[1, 0] / M[1, 1]) ** 2

    n0, dn, periods = 2.4, 0.004, 600
    kap = 4 * dn / 1550e-9                           # coupling of a square-wave index profile: 4 dn / lambda
    L_st = periods * (1550 / (4 * (n0 + dn)) + 1550 / (4 * (n0 - dn))) * 1e-3
    ws = np.linspace(1546, 1554, 81)
    R_stack = np.array([stack_reflection(n0, dn, periods, 1550.0, w) for w in ws])
    R_cmt = np.array([Circuit(1).add(bragg(1550, L_st, kap * 1e-3, w, n0), 1).outputs(w, [1])["back_power"][0] for w in ws])
    ok(np.max(np.abs(R_stack - R_cmt)) < 2e-3, f"Bragg grating = a thin-film stack of {periods} periods (reflection spectra agree to {np.max(np.abs(R_stack - R_cmt)):.0e})")

    w = 1550.37
    parts = {"waveguide": (wg(123.4, w), True), "lossy waveguide": (wg(123.4, w, 2.4, 5.0), False), "phase shifter": (phase(1.234), True),
             "loss element": (atten(2.5), False), "mirror R": (mirror(0.37), True), "mirror r": (mirror(r=-0.61), True),
             "chip edge": (edge(1.5, 0.2), False), "SLM": (slm(0.31, 0.77), True), "lossy SLM loop": (slm(0.31, loop_phase(90, w, loss_db_per_cm=20.0)), False),
             "coupler": (coupler(0.42), True), "coupler (lambda)": (coupler_wl(0.7, 1550, 0.3, w), True), "MMI": (mmi(), True), "lossy MMI": (mmi(0.8), False),
             "crossing": (crossing(), True), "ring": (ring(0.2, 200, w), True), "lossy ring": (ring(0.2, 200, w, 2.4, 4.0), False),
             "ring with backscatter": (ring(0.25, 200, w, backscatter=0.07), True), "lossy ring with backscatter": (ring(0.25, 200, w, 2.4, 3.0, backscatter=0.07), False),
             "add-drop ring": (add_drop(0.2, 0.3, 200, w), True), "lossy add-drop ring": (add_drop(0.2, 0.3, 200, w, 2.4, 6.0), False),
             "MZI": (mzi(0.6, 0.7071, 300, 350, 0.4, w), True), "lossy MZI": (mzi(0.6, 0.7071, 300, 350, 0.4, w, 2.4, 4.0), False),
             "OSSB filter": (ossb(0.7071, 0.8, 200, 0, 100, 1.5708, w), True), "lossy OSSB filter": (ossb(0.7071, 0.8, 200, 0, 100, 1.5708, w, 2.4, 3.0), False),
             **{f"CROW, {n} rings": (crow(n, 0.6, 0.2, 0.5, 190, w), True) for n in (1, 2, 3, 6)}, "lossy CROW": (crow(3, 0.6, 0.2, 0.5, 190, w, 2.4, 5.0), False),
             "Bragg grating": (bragg(1550, 500, 2, w), True), "lossy Bragg grating": (bragg(1550, 500, 2, w, 2.4, 5.0), False),
             **{f"Y {shape}, stem {stem}, {m} lanes": (y_matrix(m, stem, shape), True) for shape in ("split", "combine") for stem, m in ((1, 3), (2, 3), (1, 5))}}
    S_of = {name: to_S(M) for name, (M, _) in parts.items()}
    ok(all(np.allclose(S, S.T, atol=1e-10) for S in S_of.values()), f"all {len(parts)} component matrices are reciprocal (symmetric scattering matrix)")
    ok(all(np.allclose(S_of[n].conj().T @ S_of[n], np.eye(len(S_of[n])), atol=1e-10) and np.isclose(abs(np.linalg.det(M)), 1, atol=1e-9) for n, (M, ll) in parts.items() if ll)
       and all(np.linalg.svd(S_of[n], compute_uv=False)[0] < 1 + 1e-10 for n, (M, ll) in parts.items() if not ll),
       "lossless components: unitary scattering matrix and |det M| = 1; lossy components never amplify")
    ok(all(np.allclose(forward_only(M), S_of[n][len(M) // 2:, :len(M) // 2], atol=1e-10) for n, (M, _) in parts.items()),
       "forward_only(M) = the forward transmission block of the scattering matrix, for every component")

    def solve_by_scattering(comps, N, A):           # a different solver: every wave in every gap is an unknown, joined by the S matrices
        n = len(comps); F = lambda g, L: 2 * (g * N + L)          # forward wave in gap g on waveguide L; the backward one is F + 1
        P, rhs, row = np.zeros((2 * N * (n + 1),) * 2, complex), np.zeros(2 * N * (n + 1), complex), 0
        for L in range(N):
            P[row, F(0, L)] = 1; rhs[row] = A[L]; P[row + 1, F(n, L) + 1] = 1; row += 2
        for g, (S, ls) in enumerate(comps):
            ins, outs = [F(g, L) for L in ls] + [F(g + 1, L) + 1 for L in ls], [F(g, L) + 1 for L in ls] + [F(g + 1, L) for L in ls]
            for r in range(len(ins)):
                P[row, outs[r]] = 1; P[row, ins] -= S[r]; row += 1
            for L in set(range(N)) - set(ls):       # waveguides the component does not touch
                P[row, [F(g + 1, L), F(g, L)]] = 1, -1; P[row + 1, [F(g, L) + 1, F(g + 1, L) + 1]] = 1, -1; row += 2
        x = np.linalg.solve(P, rhs)
        return np.array([x[F(n, L)] for L in range(N)]), np.array([x[F(0, L) + 1] for L in range(N)])

    rng = np.random.default_rng(1); U = rng.uniform; worst = 0.0
    makers = [lambda w: (coupler(U(0.1, 0.9)), 2), lambda w: (slm(U(0.1, 0.6), U(0, 6)), 1), lambda w: (ring(U(0.1, 0.9), U(50, 300), w, 2.4, U(0, 5), backscatter=rng.choice([0, 0.05, 0.2])), 1),
              lambda w: (add_drop(U(0.1, 0.9), U(0.1, 0.9), U(50, 300), w, 2.4, U(0, 5)), 2), lambda w: (bragg(1550, U(100, 600), U(0.5, 4), w, 2.4, U(0, 5)), 1),
              lambda w: (mirror(U(0.05, 0.9)), 1), lambda w: (mirror(r=U(-0.9, 0.9)), 1), lambda w: (mmi(U(0, 1)), 2), lambda w: (crossing(), 2),
              lambda w: (wg(U(10, 500), w, 2.4, U(0, 5)), 1), lambda w: (edge(U(0, 3), U(0, 0.4)), 1), lambda w: (phase(U(0, 6)), 1),
              lambda w: (mzi(U(0.1, 0.9), U(0.1, 0.9), U(0, 500), U(0, 500), U(0, 6), w, 2.4, U(0, 5)), 2),
              lambda w: (ossb(U(0.1, 0.9), U(0.1, 0.95), U(50, 300), U(0, 300), U(0, 300), U(0, 6), w, 2.4, U(0, 5)), 2),
              lambda w: (crow(int(rng.integers(1, 7)), U(0.1, 0.9), U(0.05, 0.9), U(0.1, 0.9), U(50, 300), w, 2.4, U(0, 5)), 2)]
    for _ in range(200):
        N = int(rng.integers(1, 6)); w = U(1545, 1555); c = Circuit(N); comps = []
        for _ in range(int(rng.integers(1, 9))):
            M, span = makers[rng.integers(len(makers))](w)
            if span <= N:
                at = int(rng.integers(1, N - span + 2)); c.add(M, at); comps.append((to_S(M), list(range(at - 1, at - 1 + span))))
        A = rng.normal(size=N) + 1j * rng.normal(size=N)
        r = c.outputs(w, A); o2, b2 = solve_by_scattering(comps, N, A)
        worst = max(worst, np.max(np.abs(o2 - r["out"])), np.max(np.abs(b2 - r["back"])))
    ok(worst < 1e-8, f"200 random circuits: the transfer-matrix result = a separate scattering-network solve (largest difference {worst:.0e})")

    yc = lambda A: Circuit(2).y_splitter(at=1, shape="combine", make_room=False).outputs(1550, A)
    ok(np.isclose(yc([1, 1])["out_power"][0], 1) and np.isclose(yc([1, -1])["lost_power"][0], 1) and np.isclose(yc([1, 0])["out_power"][0], 0.5),
       "Y-combiner: light in step all joins the stem, light out of step is all radiated, one arm alone loses half")
    c0 = 299792458.0
    tau = lambda f, w, h=1e-4: (w * 1e-9) ** 2 / (2 * np.pi * c0) * np.angle(f(w + h) / f(w - h)) / (2 * h * 1e-9)     # -d(phase)/d(omega)
    ok(np.isclose(tau(lambda x: wg(1000, x, 2.4, 0, n_g=4.2, lam0_nm=1550)[0, 0], 1550.0), 4.2 * 1e-3 / c0, rtol=1e-6)
       and np.isclose(tau(lambda x: wg(1000, x, 2.4)[0, 0], 1550.0), 2.4 * 1e-3 / c0, rtol=1e-6), "group delay of a waveguide = n_g L / c (n_eff L / c with no group index)")

    print(f"\n{sum(results)} of {len(results)} checks passed")
