# OpticsBox

**A free, browser-based transfer-matrix simulator for integrated photonic circuits.**

Build a circuit by dragging components onto waveguides, and see the output spectrum, the light inside the circuit, the transfer matrix of every component and of the whole circuit, and ready-to-run Python, instantly. There is nothing to install, it needs no account, and it works offline.

**▶ Open OpticsBox:** https://unixpenguin.github.io/opticsbox/

---

## Features

- **Drag-and-drop circuit editor** with up to 6 waveguides. Drag components sideways to reorder them, or up and down to move them between waveguides. Drag the ⟷ handles to change waveguide lengths. Click a component to edit it; ← → step through the components, Ctrl+D duplicates, Delete removes, Ctrl+Z undoes. Scroll over the circuit to zoom, and drag its background to move around.
- **19 components:** waveguide sections, phase shifter, loss element, chip edge / grating coupler, partial mirror, waveguide crossing, Y-splitter, Y-combiner, MMI 50/50, directional coupler, wavelength-dependent coupler, Mach–Zehnder interferometer (MZI), Sagnac loop mirror (single and pair), microring resonator (MRR), add-drop MRR, coupled-resonator waveguide (CROW, 1–8 rings), OSSB filter (ring-assisted MZI) and Bragg grating. You can also save a whole circuit as a reusable **custom component**.
- **Light view:** the circuit drawn as a glowing chip. Brightness shows how much light is in each waveguide: blue travels forward, pink travels back, amber is lost at a Y-junction. Rings and cavities light up with the power built up inside them. **Sweep** runs the wavelength back and forth across the spectrum, and moving over the spectrum shows the light at that wavelength. It is drawn from the same matrices as the results, and the circuit stays fully editable.
- **3D view:** the same glowing chip in perspective, with the waveguides as ridges on the surface and the light leaving as beams at the chip's edges. Drag to turn it, right-drag to move it, scroll to zoom, and click a component to edit it.
- **Bidirectional or forward-only models.** Bidirectional uses 2N×2N matrices with forward and backward fields, including multiple reflections. Forward only uses N×N matrices and discards reflected light.
- **Results:** the power and phase at every output and reflected port, light lost at Y-junctions, and an energy-conservation check. The spectrum over a wavelength sweep can be shown as power (dB or %), **phase**, or **group delay**. **Download CSV** exports it.
- **Rings tuned to a wavelength:** any ring (MRR, add-drop MRR, CROW, OSSB filter) can be set by its length, or by the wavelength one of its resonances should sit at. It then takes the length nearest the one you give that fits a whole number of wavelengths, ℓ = 2πm/β(λ), so the resonance lands exactly there, even with dispersion or a changed index.
- **Dispersion:** enter a group index to get realistic resonance spacings (see below).
- **Chip materials:** silicon (SOI), silicon nitride, indium phosphide, thin-film lithium niobate and silica fill in a typical effective index, group index and loss (1550 nm, TE). `opticsbox.PLATFORMS` has the same values.
- **Probe:** shows how much light travels forward and backward in every waveguide at any point along the circuit.
- **Matrices tab:** the formula and numerical value of every component's matrix and of the whole circuit. Hover over any cell to see what it means.
- **Reference tab:** the general matrix of every component, in both models, with a search box.
- **Parameters:** type a name instead of a number to link fields together and sweep them with a slider.
- **Python export:** self-contained NumPy (numerical) or SymPy (symbolic, fully multiplied-out formulas) code.
- **Inputs:** light into one waveguide, all waveguides, or a custom amplitude and phase for each.
- **Examples:** built-in presets, including a Mach–Zehnder, ring filters, a Fabry–Pérot, a Michelson interferometer, a phase-shifted Bragg grating and a 1×4 splitter tree.
- **Sharing:** the Share button creates a link that opens your exact circuit.
- **Light and dark themes,** and it works on a phone.
- **Offline use:** download `index.html` and open it in any browser.

## Using it offline

Download `index.html` (click it above, then **Download raw file**) and double-click it. It runs entirely in your browser; no data leaves your computer.

## How it works

**Every equation and matrix the program uses is collected in [docs/OpticsBox-equations.pdf](docs/OpticsBox-equations.pdf)** (LaTeX source alongside). The short version:

The fields at any cross-section are ordered `[E1+, E1-, E2+, E2-, …]`, where `+` means forward-travelling and `-` backward-travelling, and the number is the waveguide. Each component has a transfer matrix `M` with `right = M · left`, acting only on the waveguides it occupies; all other waveguides pass unchanged. The circuit matrix is `T = Bₙ ··· B₂ · B₁`, in the order light meets the components. The outputs are found by solving for the reflected fields under the condition that no light enters from the output end.

### Effective index and group index

The phase light picks up along a waveguide is set by the propagation constant β. Every component with a length uses it: waveguide sections, rings, SLM loops, Y-splitter branches and Bragg gratings.

- **Effective index only** (group index left empty): `β = 2π·n_eff/λ`. Simple, but for real waveguides it puts resonances too far apart, because it ignores dispersion.
- **With a group index n_g:** `β = 2π·n_eff/λ₀ + (2π·n_g/λ₀²)·(λ₀ − λ)`. The effective index sets the phase at the reference wavelength λ₀, and the group index sets the spacing between resonances:
  - ring of length L: `λ²/(n_g·L)`
  - Fabry–Pérot of length L: `λ²/(2·n_g·L)`

  For silicon wire waveguides n_g is about 4.2, and n_eff about 2.4. Use a λ₀ near the middle of your sweep; the formula is most accurate within a few tens of nm of it.

Propagation loss (the loss setting, in dB/cm) applies to the same components.

### Phase and group delay

The Phase chart follows the phase φ of the light at each port continuously across the sweep, starting from its value at the centre wavelength. The Delay chart is the group delay `τ = −dφ/dω = λ²/(2πc) · dφ/dλ`, in picoseconds: how long light of each wavelength takes to reach the port. A plain waveguide of length L gives `n_g·L/c`; a resonator adds a peak at each resonance. Where the light at a port passes through zero its phase flips by π; the Delay chart leaves a gap there.

### Size limits

A sweep has at most 5001 points. A very large circuit (many waveguides and components) automatically uses fewer points so the page stays quick; a note above the spectrum says when, and the exported Python uses the same number.

### Python library

`opticsbox.py` is the same physics as a Python library: every component, a `Circuit` class for any number of waveguides (including Y-splitters with their radiation channels), both models, and wavelength sweeps. Pass `n_g=` and `lam0_nm=` to a component to use a group index, and `loss_db_per_cm=` for the waveguide loss. Run the file directly to self-test it (64 checks):

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
- Every component matrix is reciprocal (its scattering matrix is symmetric). Lossless ones have a unitary scattering matrix; lossy ones never amplify.
- The web app, `opticsbox.py`, and the NumPy and SymPy code the app exports give the same results, in both models and for 1–6 waveguides, on hundreds of randomly generated circuits (the SymPy ones lossless and without dispersion, as that export is).
- The resonant components have been re-derived independently: the Sagnac loop mirror, the ring (with loss and backscatter), the add-drop ring and the CROW (1 to 5 rings) are each rebuilt from couplers, lengths of waveguide and point reflectors, the equations for their internal waves are solved numerically, and the result matches the formulas used.
- The Bragg grating matches a thin-film stack of 600 quarter-wave periods, a different physical model of the same device.
- Whole circuits: the transfer-matrix result matches a separate solver that joins the components' scattering matrices and solves for every internal wave at once.
- Textbook results are reproduced:
  - Bragg grating: peak reflection tanh²(κL), and the right bandwidth.
  - All-pass and add-drop rings: the standard formulas, and resonance spacing λ²/(n_g·L).
  - Fabry–Pérot from two mirrors: the Airy transmission formula, and resonance spacing λ²/(2·n_g·L).
  - Ring backscattering: each resonance splits by the expected amount.
  - Group delay: n_g·L/c for a waveguide, and the standard peak value for a ring on resonance.
- Moving a circuit to different waveguides leaves its results unchanged.

These checks compare the maths with analytic results and between independent implementations. They are not comparisons with measured devices.

**Conventions and modelling choices.** These are self-consistent, but may differ from the conventions in a particular textbook or lab:
- Field basis `[E1+, E1-, E2+, E2-, …]`, with `right = M · left`; the phase convention is e^(−jβL) for forward propagation.
- Directional coupler: `[[τ, −jκ], [−jκ, τ]]` for forward fields, with τ = √(1 − κ²).
- Sagnac loop mirror: transfer matrix `(1/(1 − 2κ²)) · [[e^(−jφ), −2jκτ], [2jκτ, e^(jφ)]]`, which is singular at κ² = ½ (a perfect mirror). When the loop is set by its length, φ = βL and the loop has the waveguide loss too.
- Forward only: each single-waveguide component is reduced to its transmission 1/M₂₂, and reflected light is discarded.
- Y-splitter: a 3-port junction with its own radiation channel. Light on a waveguide that ends at the junction, and out-of-phase light when combining, leaves through that channel and is reported as "Lost at B…". By default the waveguides below shift down one place to make room for the new branch (or up to close the gap when combining). Light already on the bottom waveguide has nowhere to move to and is lost; the app tells you when this happens.
- Partial mirror: lossless and the same at every wavelength. Set it in one of two ways; both reflect the same power, but the reflected light has a different phase, which matters when reflections interfere:
  - **Power reflectance R:** the share of power reflected. The reflected field is j√R from either side.
  - **Field reflection r:** a real amplitude, as at a refractive-index step: r from the left, −r from the right. The power reflected is r².
- Chip edge: loss followed by a lossless mirror, so the reflection is the one seen from the input side.
- Ring backscattering (off by default): sidewall roughness is modelled as one small reflector inside the ring. It couples light travelling clockwise and counter-clockwise, so each resonance splits in two and some light is reflected back along the waveguide.
- Bragg grating: uniform, from coupled-mode theory, with its peak at the Bragg wavelength you set. Its teeth are counted from its left end, so two gratings in a row are in step only when the first is a whole number of periods long (one period is λ_B/(2·n_eff)). The "Phase-shifted Bragg grating" preset uses such a length, which puts its resonance exactly at the Bragg wavelength.
- Components have no length of their own: couplers, MMIs, crossings, mirrors, Y-junctions, and the sideways shift of waveguides at a "make room" Y-splitter add no path length or bend loss unless set. The Y-splitter has optional branch and shift lengths; elsewhere, add a Waveguides section to model real path lengths. This matters only where paths of different length meet again.
- Loss: waveguide sections, rings, add-drop rings, CROWs, MZI and OSSB arms, SLM loops set by a length, Y-splitter path lengths and Bragg gratings use the loss setting (dB/cm). Other components are lossless unless you give them a loss.
- Drawing: rings and couplers are drawn with a narrow gap, as on a real chip. The waveguides never touch there; light crosses the gap, and κ says how much. A Sagnac loop is drawn joined to its waveguide, because the loop is that same waveguide closed on itself by a coupler.
- Light view: the brightness in a waveguide follows the power between components; inside a component it is blended from the power going in to the power coming out. The light shown inside a ring or loop is its circulating power, worked out from the fields arriving at it. The 3D view shows the same light; its chip is an illustration, not to scale.
- SymPy export: β stays a symbol, and ring, add-drop, CROW, MZI, OSSB, SLM-loop and grating formulas are lossless. Lengths are in µm as in the app, so to get numbers give β (and a grating strength) per µm. Simplifying a large bidirectional circuit (for example one with a ring and a grating) can take several minutes.
- The wavelength-dependent coupler uses a simple linear model for how its coupling changes with wavelength.

**Found a mistake or a convention that should change?** Please [open an issue](https://github.com/UnixPenguin/opticsbox/issues) with the circuit (use the Share button to get a link) and the result you expected.

## Citing

If OpticsBox helps your work, a mention or a link back is appreciated.

## License

MIT. Free to use, modify and share. See [LICENSE](LICENSE).
