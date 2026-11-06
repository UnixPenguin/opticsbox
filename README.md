# OpticsBox

**A free, browser-based transfer-matrix simulator for integrated photonic circuits.**

Build a circuit by dragging components onto waveguides, and see the output spectrum, the transfer matrix of every component and of the whole circuit, and ready-to-run Python, instantly. There is nothing to install, it needs no account, and it works offline.

**▶ Open OpticsBox:** https://unixpenguin.github.io/opticsbox/

---

## Features

- **Drag-and-drop circuit editor** with up to 6 waveguides. Drag components sideways to reorder them, or up and down to move them between waveguides. Drag the ⟷ handles to change waveguide lengths.
- **16 components**, from simplest to most complex: waveguide sections, phase shifter, loss element, chip edge / grating coupler, partial mirror, waveguide crossing, Y-splitter, Y-combiner, MMI 50/50, directional coupler, wavelength-dependent coupler, Sagnac loop mirror (single and pair), ring resonator, add-drop ring and Bragg grating. You can also save a whole circuit as a reusable **custom component**.
- **Dispersion:** enter a group index to get realistic resonance spacings (see below).
- **Bidirectional or forward-only models.** Bidirectional uses 2N×2N matrices with forward and backward fields, including multiple reflections. Forward only uses N×N matrices and discards reflected light.
- **Results:** the power and phase at every output and reflected port, light lost at Y-junctions, the spectrum over a wavelength sweep (in dB or %), and an energy-conservation check. **Download CSV** exports the spectrum.
- **Probe:** shows how much light travels forward and backward in every waveguide at any point along the circuit.
- **Matrices tab:** the formula and numerical value of every component's matrix and of the whole circuit. Hover over any cell to see what it means.
- **Reference tab:** the general matrix of every component, in both models, with a search box.
- **Parameters:** type a name instead of a number to link fields together and sweep them with a slider.
- **Python export:** self-contained NumPy (numerical) or SymPy (symbolic, fully multiplied-out formulas) code.
- **Inputs:** light into one waveguide, all waveguides, or a custom amplitude and phase for each.
- **Sharing:** the Share button creates a link that opens your exact circuit.
- **Offline use:** download `index.html` and open it in any browser.

## Using it offline

Download `index.html` (click it above, then **Download raw file**) and double-click it. It runs entirely in your browser; no data leaves your computer.

## How it works

The fields at any cross-section are ordered `[E1+, E1-, E2+, E2-, …]`, where `+` means forward-travelling and `-` backward-travelling, and the number is the waveguide. Each component has a transfer matrix `M` with `right = M · left`, acting only on the waveguides it occupies; all other waveguides pass unchanged. The circuit matrix is `T = Bₙ ··· B₂ · B₁`, in the order light meets the components. The outputs are found by solving for the reflected fields under the condition that no light enters from the output end.

### Effective index and group index

The phase light picks up along a waveguide is set by the propagation constant β. Every component with a length uses it: waveguide sections, rings, SLM loops, Y-splitter branches and Bragg gratings.

- **Effective index only** (group index left empty): `β = 2π·n_eff/λ`. Simple, but for real waveguides it puts resonances too far apart, because it ignores dispersion.
- **With a group index n_g:** `β = 2π·n_eff/λ₀ + (2π·n_g/λ₀²)·(λ₀ − λ)`. The effective index sets the phase at the reference wavelength λ₀, and the group index sets the spacing between resonances:
  - ring of length L: `λ²/(n_g·L)`
  - Fabry–Pérot of length L: `λ²/(2·n_g·L)`

  For silicon wire waveguides n_g is about 4.2, and n_eff about 2.4. Use a λ₀ near the middle of your sweep; the formula is most accurate within a few tens of nm of it.

Propagation loss (the loss setting, in dB/cm) applies to the same components.

### Python library

`opticsbox.py` is the same physics as a Python library: every component, a `Circuit` class for any number of waveguides (including Y-splitters with their radiation channels), both models, and wavelength sweeps. Pass `n_g=` and `lam0_nm=` to a component to use a group index. Run the file directly to self-test it:

```python
from opticsbox import *
c = Circuit(waveguides=2)
c.add(coupler(0.7071), at=1)
c.add(lambda w: lanes(wg(1000, w), wg(1100, w)), at=1)
c.add(coupler(0.7071), at=1)
print(c.outputs(1550, inputs=[1, 0])["out_power"])
```

## Status

OpticsBox was built with AI assistance and is under active review. Please treat its results as a helpful first estimate, and verify them before relying on them in published work or lab decisions.

**What has been checked**
- Lossless components conserve energy to machine precision, and every lossless component matrix has |det M| = 1.
- The web app, `opticsbox.py`, and the NumPy and SymPy code the app exports give the same results, in both models and for 1–6 waveguides.
- Textbook results are reproduced:
  - Bragg grating: peak reflection tanh²(κL), and the right bandwidth.
  - All-pass and add-drop rings: the standard formulas, and resonance spacing λ²/(n_g·L).
  - Fabry–Pérot from two mirrors: the Airy transmission formula, and resonance spacing λ²/(2·n_g·L).
  - Ring backscattering: each resonance splits by the expected amount.
- Moving a circuit to different waveguides leaves its results unchanged.

These checks compare the maths with analytic results and between independent implementations. They are not comparisons with measured devices.

**Conventions and modelling choices.** These are self-consistent, but may differ from the conventions in a particular textbook or lab:
- Field basis `[E1+, E1-, E2+, E2-, …]`, with `right = M · left`; the phase convention is e^(−jβL) for forward propagation.
- Directional coupler: `[[τ, −jκ], [−jκ, τ]]` for forward fields, with τ = √(1 − κ²).
- Sagnac loop mirror: transfer matrix `(1/(1 − 2κ²)) · [[e^(−jφ), −2jκτ], [2jκτ, e^(jφ)]]`, which is singular at κ² = ½ (a perfect mirror).
- Forward only: each single-waveguide component is reduced to its transmission 1/M₂₂, and reflected light is discarded.
- Y-splitter: a 3-port junction with its own radiation channel. Light on a waveguide that ends at the junction, and out-of-phase light when combining, leaves through that channel and is reported as "Lost at B…". By default the waveguides below shift down one place to make room for the new branch (or up to close the gap when combining). Light already on the bottom waveguide has nowhere to move to and is lost; the app tells you when this happens.
- Partial mirror: lossless and the same at every wavelength. Set it in one of two ways; both reflect the same power, but the reflected light has a different phase, which matters when reflections interfere:
  - **Power reflectance R:** the share of power reflected. The reflected field is j√R from either side.
  - **Field reflection r:** a real amplitude, as at a refractive-index step: r from the left, −r from the right. The power reflected is r².
- Chip edge: loss followed by a lossless mirror, so the reflection is the one seen from the input side.
- Ring backscattering (off by default): sidewall roughness is modelled as one small reflector inside the ring. It couples light travelling clockwise and counter-clockwise, so each resonance splits in two and some light is reflected back along the waveguide.
- Bragg grating: uniform, from coupled-mode theory, with its peak at the Bragg wavelength you set.
- Components have no length of their own: couplers, MMIs, crossings, mirrors, Y-junctions, and the sideways shift of waveguides at a "make room" Y-splitter add no path length or bend loss unless set. The Y-splitter has optional branch and shift lengths; elsewhere, add a Waveguides section to model real path lengths. This matters only where paths of different length meet again.
- Loss: waveguide sections, rings, add-drop rings, Y-splitter path lengths and Bragg gratings use the loss setting (dB/cm). Other components are lossless unless you give them a loss.
- SymPy export: β stays a symbol, and ring, add-drop and grating formulas are lossless. Simplifying a large bidirectional circuit (for example one with a ring and a grating) can take several minutes.
- The wavelength-dependent coupler uses a simple linear model for how its coupling changes with wavelength.

**Found a mistake or a convention that should change?** Please [open an issue](https://github.com/UnixPenguin/opticsbox/issues) with the circuit (use the Share button to get a link) and the result you expected.

## Citing

If OpticsBox helps your work, a mention or a link back is appreciated.

## License

MIT. Free to use, modify and share. See [LICENSE](LICENSE).
