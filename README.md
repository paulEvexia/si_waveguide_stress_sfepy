# Silicon Waveguide Stress and TE00 DBR Sweep

This repository contains Python scripts for a 2D silicon photonic rib-waveguide
stress simulation and a TE00 effective-index/DBR wavelength-shift sweep. The
current committed files are scripts only; generated CSV, PNG, VTK, and PowerPoint
outputs are intentionally left out of git.

## Scripts

- `src/run_si_waveguide_plane_strain.py`: builds and solves the 2D plane-strain
  mechanical stress model for the silicon/SiO2 waveguide cross-section.
- `src/solve_optical_modes.py`: solves approximate TE-like and TM-like optical
  modes for the rib geometry, optionally using a stress/strain CSV as a
  photoelastic perturbation.
- `src/sweep_reference_temperature_te00_dbr.py`: sweeps reference temperature and
  intrinsic silicon stress, solves the unperturbed and perturbed DBR sections,
  and reports TE00 effective-index and DBR center-wavelength shifts.
- `tools/prepare_sweep_ppt_assets.py`: builds summary plot assets and a manifest
  from completed sweep outputs.
- `tools/build_sweep_pptx_com.ps1`: PowerShell helper for constructing a
  PowerPoint report from the prepared sweep assets.

## Requirements

Use a Python environment with the scientific stack and SfePy available. The
scripts import packages including `numpy`, `scipy`, `matplotlib`, and `sfepy`.

Run commands from the repository root:

```powershell
python src/sweep_reference_temperature_te00_dbr.py
```

## TE00 DBR Sweep

The main sweep command is:

```powershell
python src/sweep_reference_temperature_te00_dbr.py [options]
```

The sweep combines two axes:

- Reference temperature, `T_ref`, in degrees C.
- Intrinsic silicon stress, in MPa.

For each sweep point, the script evaluates two DBR sections:

- `unperturbed`: no SiO2 rib insert, duty cycle 0.50.
- `perturbed`: SiO2 rib insert enabled, duty cycle 0.50.

The reported DBR effective index is the 50/50 weighted average of the two
sections. The stressed DBR center wavelength is calculated relative to a 1550 nm
design wavelength:

```text
n_DBR = 0.5 * n_unperturbed + 0.5 * n_perturbed
lambda_stressed = 1550 nm * n_DBR_stressed / n_DBR_unstressed
```

### Sweep Axis Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--t-ref-start` | `1000.0` | First reference temperature, in degrees C. |
| `--t-ref-stop` | `400.0` | Last reference temperature, in degrees C. |
| `--t-ref-step` | `-50.0` | Temperature increment. Use a negative step for a descending sweep. |
| `--intrinsic-start` | `-250.0` | First intrinsic Si stress value, in MPa. |
| `--intrinsic-stop` | `250.0` | Last intrinsic Si stress value, in MPa. |
| `--intrinsic-step` | `100.0` | Intrinsic-stress increment, in MPa. |

The default sweep evaluates:

```text
T_ref = 1000, 950, ..., 400 C
intrinsic stress = -250, -150, -50, 50, 150, 250 MPa
```

That is 13 temperature values x 6 stress values = 78 DBR sweep points.

The sweep builder includes both endpoints when the step direction reaches the
stop value. A zero step is invalid. If start, stop, and step point in conflicting
directions, the script raises an error because no values are produced.

### Mechanical Mesh and Solver Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--stress-nx` | `200` | Number of mechanical mesh cells in x for the sweep basis solve. |
| `--stress-ny` | `140` | Number of mechanical mesh cells in y for the sweep basis solve. |
| `--mechanical-subprocess` | enabled | Run each mechanical section solve in a fresh Python process. |
| `--no-mechanical-subprocess` | disabled | Run mechanical section solves in the main process. |
| `--mechanical-linear-solver` | `direct` | Mechanical solver: `direct` or `iterative`. |
| `--mechanical-iterative-method` | `cg` | SciPy iterative method used when `--mechanical-linear-solver iterative` is selected. |
| `--mechanical-iterative-i-max` | `5000` | Maximum iterations for the iterative mechanical solver. |
| `--mechanical-iterative-eps-a` | `1e-10` | Absolute tolerance for the iterative mechanical solver. |
| `--mechanical-iterative-eps-r` | `1e-8` | Relative tolerance for the iterative mechanical solver. |
| `--mechanical-basis-eps-star` | `1e-3` | Silicon eigenstrain used for the mechanical basis solve. |

The mechanical part is linear in the scalar silicon eigenstrain used by this
workflow. The sweep solves one mechanical basis field for the unperturbed section
and one for the perturbed section, then scales those fields for every
temperature/stress point. This avoids repeated full mechanical solves across the
entire sweep.

Use the direct solver for the default reduced sweep mesh. For larger meshes that
exceed direct-solver memory, try:

```powershell
python src/sweep_reference_temperature_te00_dbr.py --stress-nx 400 --stress-ny 280 --mechanical-linear-solver iterative
```

### Optical Solver Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--optical-dx` | `0.10` | Target optical grid spacing in x, in microns. |
| `--optical-dy` | `0.10` | Target optical grid spacing in y, in microns. |
| `--modes` | `6` | Number of modes requested from the optical eigensolve. |
| `--include-handle` | off | Include the high-index silicon handle in the optical window. |
| `--strain-source` | `elastic` | Strain field used for photoelastic coupling: `elastic` or `total`. |
| `--stress-scale` | `1.0` | Multiplier applied to the strain-induced index perturbation. |

The default optical window excludes the high-index Si handle so substrate modes
do not dominate the rib modes. Use `--include-handle` only when handle modes are
intentionally part of the analysis.

Use `--strain-source elastic` for the usual photoelastic perturbation so free
thermal/eigenstrain is not double-counted. Use `--strain-source total` only when
you explicitly want the total strain tensor in the perturbation.

For a denser final optical grid:

```powershell
python src/sweep_reference_temperature_te00_dbr.py --optical-dx 0.05 --optical-dy 0.05
```

### Resume and Output-Control Options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--resume` | off | Load the existing sweep CSV and skip completed points with matching run settings. |
| `--save-point-pngs` | off | Save stress, strain, TE00/TM00 intensity, and delta-n PNGs for sweep points. |
| `--point-png-every` | `1` | Save per-point PNGs every Nth sweep point when PNG saving is enabled. |
| `--point-png-dpi` | `150` | DPI for per-point PNG output. Must be at least 50. |
| `--point-png-max-pixels` | `500000` | Maximum plotted array pixels per per-point PNG. Larger arrays are strided for plotting only. |
| `--save-point-mode-fields` | off | Also save signed TE00/TM00 field PNGs, not only intensities. |
| `--keep-point-pngs` | off | Keep existing per-point PNG folders on a non-resume run. |

The sweep CSV is written one completed DBR point at a time. If a long run stops,
continue from completed rows with:

```powershell
python src/sweep_reference_temperature_te00_dbr.py --resume
```

Per-point PNGs can create many files. For a lighter visual audit:

```powershell
python src/sweep_reference_temperature_te00_dbr.py --save-point-pngs --point-png-every 10
```

To reduce plotting memory:

```powershell
python src/sweep_reference_temperature_te00_dbr.py --save-point-pngs --point-png-dpi 100 --point-png-max-pixels 200000
```

### Example Sweeps

Fast smoke test with one temperature and one stress:

```powershell
python src/sweep_reference_temperature_te00_dbr.py --t-ref-start 800 --t-ref-stop 800 --t-ref-step 1 --intrinsic-start 0 --intrinsic-stop 0 --intrinsic-step 1
```

Default sweep:

```powershell
python src/sweep_reference_temperature_te00_dbr.py
```

Finer temperature spacing:

```powershell
python src/sweep_reference_temperature_te00_dbr.py --t-ref-start 1000 --t-ref-stop 400 --t-ref-step -25
```

Wider intrinsic-stress sweep:

```powershell
python src/sweep_reference_temperature_te00_dbr.py --intrinsic-start -500 --intrinsic-stop 500 --intrinsic-step 50
```

Final-resolution optical sweep with resume support:

```powershell
python src/sweep_reference_temperature_te00_dbr.py --optical-dx 0.05 --optical-dy 0.05 --resume
```

## Sweep Outputs

Primary CSV:

```text
results/te00_neff_and_dbr_shift_vs_reference_temperature_and_intrinsic_stress.csv
```

Summary plots:

```text
rslt/optical_modes/te00_delta_neff_vs_reference_temperature.png
rslt/optical_modes/dbr_center_wavelength_shift_vs_reference_temperature.png
rslt/optical_modes/te00_delta_neff_temperature_intrinsic_stress_heatmap.png
rslt/optical_modes/dbr_shift_temperature_intrinsic_stress_heatmap.png
```

When `--save-point-pngs` is enabled, per-point plots are written below:

```text
rslt/optical_modes/sweep_points/
```

Each sweep point has separate `unperturbed` and `perturbed` section folders.

## Optical Mode Solver

Run the standalone optical solver with:

```powershell
python src/solve_optical_modes.py
```

Options:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--dx` | script default | Optical grid spacing in x, in microns. |
| `--dy` | script default | Optical grid spacing in y, in microns. |
| `--modes` | script default | Number of modes requested from the eigensolve. |
| `--include-handle` | off | Include the high-index silicon handle in the optical window. |
| `--stress-coupled` | off | Re-solve modes with strain-optic perturbation from a stress CSV. |
| `--stress-csv` | `results/stress_strain_2d_distribution_for_jmp.csv` | Stress/strain CSV used for stress-coupled optical solves. |
| `--strain-source` | `elastic` | Use `elastic` or `total` strain for the perturbation. |
| `--stress-scale` | `1.0` | Multiplier for stress/strain-induced index perturbation. |

Example stress-coupled solve:

```powershell
python src/solve_optical_modes.py --stress-coupled
```

## Mechanical Stress Solver

Run the standalone mechanical model with:

```powershell
python src/run_si_waveguide_plane_strain.py
```

The mechanical script writes JMP-friendly CSV files under `results/` and stress,
strain, and material-map PNGs under `rslt/`.

The geometry is a silicon rib waveguide with SiO2 BOX/cladding and an optional
SiO2 insert used by the perturbed DBR section. The model uses plane-strain
linear elasticity with thermal mismatch and optional intrinsic silicon stress
represented as eigenstrain.

## PowerPoint Asset Helper

After a sweep with the required PNGs exists, prepare report assets with:

```powershell
python tools/prepare_sweep_ppt_assets.py --section perturbed
```

or:

```powershell
python tools/prepare_sweep_ppt_assets.py --section unperturbed
```

The helper writes summary plots and a manifest under the sweep output folders.
Generated presentation files are not tracked in git.

## Notes

The built-in photoelastic coefficients and material parameters are useful for
workflow development, but process/foundry-specific values should be substituted
before treating absolute `delta_neff` or DBR wavelength shifts as signoff data.
