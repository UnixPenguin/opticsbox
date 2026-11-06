"""OpticsBox - transfer-matrix building blocks for multi-waveguide photonic circuits.

The same physics as the OpticsBox web app (https://unixpenguin.github.io/opticsbox/), as a Python library.

Conventions
    Fields at any cross-section: [E1+, E1-, E2+, E2-, ...]; + travels forward (left to right), - backward,
    and the number is the waveguide. Every component maps the fields on its left to those on its right:
    right = M @ left. A circuit is T = Bn @ ... @ B2 @ B1, in the order light meets the components.
    Forward propagation picks up exp(-j*beta*L); the directional coupler is [[tau, -j*kappa], [-j*kappa, tau]].
    Components have no length of their own unless a length is given.
    Phase follows beta(lambda): 2*pi*n_eff/lambda by default (constant index). Pass n_g (group index) and lam0_nm to
    any component with a length (waveguide, ring, add_drop, bragg, loop_phase, Circuit.y_splitter) for first-order
    dispersion, beta = 2*pi*n_eff/lam0 + (2*pi*n_g/lam0^2)*(lam0 - lambda); use the same values everywhere.
    Mirrors: mirror(R) takes the power reflectance (reflects j*sqrt(R)); mirror(r=...) a real field reflection r.

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
    """Sagnac loop mirror on one waveguide; undefined at kappa^2 = 0.5 (a perfect mirror)."""
    t = np.sqrt(1 - kappa ** 2)
    m = 2j * kappa * t
    return np.array([[np.exp(-1j * phi), -m], [m, np.exp(1j * phi)]], dtype=complex) / (1 - 2 * kappa ** 2)


def slm_pair(kappa1, phi1, kappa2, phi2):
    """An SLM on each of two waveguides (name kept from the original version)."""
    return lanes(slm(kappa1, phi1), slm(kappa2, phi2))


def loop_phase(loop_um, wavelength_nm, n_eff=2.4, n_g=None, lam0_nm=1550.0):
    """Phase of an SLM loop set by its length: beta(lambda) * L (2*pi*n_eff*L/lambda without dispersion)."""
    return beta(wavelength_nm, n_eff, n_g, lam0_nm) * loop_um * 1e-6


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
                           ("chip edge (0 dB)", lanes(edge(0, 0.3), edge(0, 0.3)), 1)]:
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

    print(f"\n{sum(results)} of {len(results)} checks passed")
