import argparse
import csv
import html
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST_FILE = ROOT / "results" / "sweep_ppt_manifest.json"
CSV_FILE = ROOT / "results" / "te00_neff_and_dbr_shift_vs_reference_temperature_and_intrinsic_stress.csv"
OUTPUT_FILE = ROOT / "si_waveguide_stress_dbr_sweep_report.html"

DESIGN_WAVELENGTH_NM = 1550.0
T_USE_C = 25.0
ALPHA_SI = 2.6e-6
ALPHA_SIO2 = 0.5e-6
E_SI_MPA = 130_000.0
NU_SI = 0.28
K_SI_MPA = E_SI_MPA / (3.0 * (1.0 - 2.0 * NU_SI))


def rel_uri(path, output_file):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    try:
        rel = path.resolve().relative_to(output_file.resolve().parent)
    except ValueError:
        rel = path.resolve()
    return rel.as_posix().replace(" ", "%20")


def fmt(value, digits=1):
    if value is None:
        return ""
    if isinstance(value, str):
        return html.escape(value)
    if abs(value) >= 100:
        return f"{value:,.{digits}f}"
    if abs(value) >= 1:
        return f"{value:.{digits}f}"
    return f"{value:.4g}"


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    slides = manifest.get("slides", [])
    slides.sort(key=lambda row: (-float(row["T_ref_C"]), float(row["sigma_intrinsic_si_MPa"])))
    return manifest, slides


def read_current_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique_sorted(rows, key, reverse=False):
    return sorted({float(row[key]) for row in rows}, reverse=reverse)


def value_range(rows, key):
    values = [float(row[key]) for row in rows]
    return min(values), max(values)


def endpoint_slope(rows, group_key, x_key, y_key):
    slopes = []
    groups = {}
    for row in rows:
        groups.setdefault(float(row[group_key]), []).append(row)
    for group in groups.values():
        group = sorted(group, key=lambda row: float(row[x_key]))
        first = group[0]
        last = group[-1]
        dx = float(last[x_key]) - float(first[x_key])
        if not math.isclose(dx, 0.0):
            slopes.append((float(last[y_key]) - float(first[y_key])) / dx)
    return sum(slopes) / len(slopes) if slopes else None


def find_extreme(rows, key, reverse=False):
    return sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[0]


def eps_star(t_ref_c, sigma_intrinsic_mpa):
    d_t = T_USE_C - float(t_ref_c)
    eps_cte = (ALPHA_SI - ALPHA_SIO2) * d_t
    eps_intrinsic = float(sigma_intrinsic_mpa) / (3.0 * K_SI_MPA)
    return eps_cte + eps_intrinsic


def image_figure(title, src, caption, output_file, class_name=""):
    path = ROOT / src if not Path(src).is_absolute() else Path(src)
    if not path.exists():
        return ""
    return f"""
        <figure class="figure {html.escape(class_name)}">
          <img src="{html.escape(rel_uri(path, output_file))}" alt="{html.escape(title)}" loading="lazy">
          <figcaption><strong>{html.escape(title)}</strong>{html.escape(caption)}</figcaption>
        </figure>
    """


def figure_grid(figures):
    figures = [figure for figure in figures if figure]
    if not figures:
        return ""
    return '<div class="figure-grid">' + "\n".join(figures) + "</div>"


def existing_representative_point(manifest):
    for slide in manifest.get("slides", []):
        image_paths = [Path(path) for path in slide.get("images", {}).values()]
        if image_paths and all(path.exists() for path in image_paths):
            folder = Path(slide["folder"])
            return slide, folder
    return None, None


def stat_card(label, value, detail=""):
    return f"""
      <div class="stat">
        <span>{html.escape(label)}</span>
        <strong>{html.escape(value)}</strong>
        <small>{html.escape(detail)}</small>
      </div>
    """


def build_rows_table(rows):
    out = []
    for row in rows:
        t_ref = float(row["T_ref_C"])
        sigma = float(row["sigma_intrinsic_si_MPa"])
        shift_pm = float(row["dbr_delta_wavelength_pm"])
        shifted_lambda = DESIGN_WAVELENGTH_NM + shift_pm / 1000.0
        out.append(
            "          <tr>"
            f"<td>{t_ref:.0f}</td>"
            f"<td>{T_USE_C - t_ref:.0f}</td>"
            f"<td>{sigma:.0f}</td>"
            f"<td>{eps_star(t_ref, sigma):.6g}</td>"
            f"<td>{shift_pm:.1f}</td>"
            f"<td>{shifted_lambda:.4f}</td>"
            "</tr>"
        )
    return "\n".join(out)


def build_html(manifest, rows, current_csv_rows, output_file):
    temps = unique_sorted(rows, "T_ref_C", reverse=True)
    sigmas = unique_sorted(rows, "sigma_intrinsic_si_MPa")
    min_shift, max_shift = value_range(rows, "dbr_delta_wavelength_pm")
    max_row = find_extreme(rows, "dbr_delta_wavelength_pm", reverse=True)
    min_row = find_extreme(rows, "dbr_delta_wavelength_pm")
    stress_slope = endpoint_slope(
        rows,
        "T_ref_C",
        "sigma_intrinsic_si_MPa",
        "dbr_delta_wavelength_pm",
    )
    temp_slope = endpoint_slope(
        rows,
        "sigma_intrinsic_si_MPa",
        "T_ref_C",
        "dbr_delta_wavelength_pm",
    )
    zero_rows = [
        row
        for row in rows
        if math.isclose(float(row["sigma_intrinsic_si_MPa"]), 0.0, abs_tol=1e-12)
    ]
    zero_rows = sorted(zero_rows, key=lambda row: float(row["T_ref_C"]), reverse=True)
    zero_high = float(zero_rows[0]["dbr_delta_wavelength_pm"]) if zero_rows else None
    zero_low = float(zero_rows[-1]["dbr_delta_wavelength_pm"]) if zero_rows else None

    rep_slide, rep_folder = existing_representative_point(manifest)
    rep_title = ""
    stress_figures = ""
    optical_figures = ""
    if rep_slide and rep_folder:
        rep_title = (
            f"T_ref = {float(rep_slide['T_ref_C']):.0f} C, "
            f"intrinsic stress = {float(rep_slide['sigma_intrinsic_si_MPa']):.0f} MPa, "
            f"section = {rep_slide['section']}"
        )
        stress_figures = figure_grid(
            [
                image_figure("sigma_xx [MPa]", rep_folder / "stress_sigma_xx_MPa.png", "", output_file),
                image_figure("sigma_yy [MPa]", rep_folder / "stress_sigma_yy_MPa.png", "", output_file),
                image_figure("sigma_xy [MPa]", rep_folder / "stress_sigma_xy_MPa.png", "", output_file),
                image_figure("eps_xx", rep_folder / "strain_elastic_eps_xx.png", "", output_file),
                image_figure("eps_yy", rep_folder / "strain_elastic_eps_yy.png", "", output_file),
                image_figure("eps_xy", rep_folder / "strain_elastic_eps_xy.png", "", output_file),
            ]
        )
        optical_figures = figure_grid(
            [
                image_figure("TE00 intensity", rep_folder / "optical_TE00_intensity.png", "", output_file),
                image_figure("TE delta n", rep_folder / "optical_TE_delta_n.png", "", output_file),
                image_figure("TM00 intensity", rep_folder / "optical_TM00_intensity.png", "", output_file),
                image_figure("TM delta n", rep_folder / "optical_TM_delta_n.png", "", output_file),
            ]
        )

    provenance_note = (
        f"The manifest contains {len(rows)} sweep points. "
        f"The current CSV at results/ has {len(current_csv_rows)} data row(s), "
        "so the table below follows the manifest used by the existing PowerPoint assets."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Silicon Waveguide Stress and TE00 DBR Sweep Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17212b;
      --muted: #5d6975;
      --line: #d7dde3;
      --paper: #f7f9fb;
      --panel: #ffffff;
      --teal: #0f766e;
      --blue: #285c9c;
      --coral: #b94a48;
      --gold: #9a6a12;
      --green: #2f7d45;
    }}

    * {{
      box-sizing: border-box;
    }}

    html {{
      scroll-behavior: smooth;
    }}

    body {{
      margin: 0;
      font-family: "Aptos", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--paper);
      line-height: 1.55;
    }}

    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}

    .wrap {{
      width: min(1180px, calc(100% - 40px));
      margin: 0 auto;
    }}

    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
      gap: 36px;
      padding: 44px 0 34px;
      align-items: end;
    }}

    h1, h2, h3 {{
      margin: 0;
      line-height: 1.12;
      letter-spacing: 0;
    }}

    h1 {{
      font-size: clamp(2rem, 4vw, 4.2rem);
      max-width: 920px;
    }}

    h2 {{
      font-size: clamp(1.5rem, 2vw, 2.2rem);
      margin-bottom: 14px;
    }}

    h3 {{
      font-size: 1.02rem;
      margin-bottom: 8px;
    }}

    p {{
      margin: 0 0 12px;
      color: var(--muted);
    }}

    .lede {{
      max-width: 820px;
      color: #374151;
      font-size: 1.06rem;
      margin-top: 18px;
    }}

    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 0 0 24px;
    }}

    nav a {{
      color: var(--ink);
      text-decoration: none;
      border: 1px solid var(--line);
      background: #ffffff;
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.92rem;
    }}

    section {{
      padding: 32px 0;
      border-bottom: 1px solid var(--line);
    }}

    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}

    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-left: 5px solid var(--teal);
      border-radius: 8px;
      padding: 14px 16px;
      min-height: 112px;
    }}

    .stat:nth-child(2) {{ border-left-color: var(--blue); }}
    .stat:nth-child(3) {{ border-left-color: var(--coral); }}
    .stat:nth-child(4) {{ border-left-color: var(--gold); }}
    .stat:nth-child(5) {{ border-left-color: var(--green); }}

    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .stat strong {{
      display: block;
      margin-top: 8px;
      font-size: 1.45rem;
      line-height: 1.1;
    }}

    .stat small {{
      display: block;
      color: var(--muted);
      margin-top: 8px;
    }}

    .two-col {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 28px;
      align-items: start;
    }}

    .note {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px 18px;
      color: var(--muted);
    }}

    .formula {{
      background: #102033;
      color: #f7fbff;
      padding: 14px 16px;
      border-radius: 8px;
      overflow-x: auto;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 0.92rem;
      line-height: 1.55;
      margin: 14px 0;
    }}

    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}

    .figure-grid .wide {{
      grid-column: 1 / -1;
    }}

    figure {{
      margin: 0;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}

    figure img {{
      display: block;
      width: 100%;
      height: auto;
      background: #ffffff;
    }}

    figcaption {{
      border-top: 1px solid var(--line);
      padding: 10px 12px;
      color: var(--muted);
      font-size: 0.9rem;
    }}

    figcaption strong {{
      color: var(--ink);
      margin-right: 6px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}

    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}

    th:first-child, td:first-child {{
      text-align: left;
    }}

    th {{
      position: sticky;
      top: 0;
      background: #eef3f7;
      color: #26323e;
      z-index: 1;
      font-size: 0.88rem;
    }}

    .table-wrap {{
      max-height: 560px;
      overflow: auto;
      border-radius: 8px;
      margin-top: 18px;
    }}

    .callouts {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}

    .callout {{
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}

    .callout b {{
      color: var(--ink);
      display: block;
      margin-bottom: 6px;
    }}

    footer {{
      color: var(--muted);
      padding: 28px 0 44px;
      font-size: 0.92rem;
    }}

    @media (max-width: 900px) {{
      .hero, .two-col, .figure-grid, .callouts {{
        grid-template-columns: 1fr;
      }}
      .stat-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 560px) {{
      .wrap {{
        width: min(100% - 24px, 1180px);
      }}
      .stat-grid {{
        grid-template-columns: 1fr;
      }}
      section {{
        padding: 24px 0;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <div class="hero">
        <div>
          <h1>Silicon waveguide stress and TE00 DBR sweep</h1>
          <p class="lede">This report summarizes the coupled mechanical and optical workflow for a silicon rib waveguide with SiO2 cladding. The sweep varies reference temperature and intrinsic silicon stress, converts the resulting elastic strain into a photoelastic TE00 effective-index perturbation, and reports the induced DBR center-wavelength shift.</p>
        </div>
        <div class="note">
          <h3>Data source</h3>
          <p>{html.escape(provenance_note)}</p>
        </div>
      </div>
      <nav aria-label="Report sections">
        <a href="#summary">Summary</a>
        <a href="#workflow">Workflow</a>
        <a href="#sweep">Double sweep</a>
        <a href="#plots">Plots</a>
        <a href="#fields">Fields</a>
        <a href="#data">Data</a>
        <a href="#caveats">Caveats</a>
      </nav>
    </div>
  </header>

  <main>
    <section id="summary">
      <div class="wrap">
        <h2>Summary</h2>
        <p>The observed DBR shift spans {min_shift:.1f} pm to {max_shift:.1f} pm over the manifest sweep. In this model, increasing reference temperature means more cooling from the stress-free reference to 25 C, and that stronger cooldown increases the positive wavelength shift. Moving along the positive intrinsic-stress axis offsets that trend and can push the DBR shift through zero.</p>
        <div class="stat-grid">
          {stat_card("Sweep size", f"{len(temps)} x {len(sigmas)} = {len(rows)}", "reference temperatures by intrinsic stresses")}
          {stat_card("Reference temperature", f"{temps[-1]:.0f} to {temps[0]:.0f} C", "T_use fixed at 25 C")}
          {stat_card("Intrinsic Si stress", f"{sigmas[0]:.0f} to {sigmas[-1]:.0f} MPa", "manifest axis values")}
          {stat_card("DBR shift span", f"{min_shift:.1f} to {max_shift:.1f} pm", "relative to 1550 nm")}
          {stat_card("Largest red shift", f"{float(max_row['dbr_delta_wavelength_pm']):.1f} pm", f"T_ref {float(max_row['T_ref_C']):.0f} C, sigma {float(max_row['sigma_intrinsic_si_MPa']):.0f} MPa")}
          {stat_card("Largest blue shift", f"{float(min_row['dbr_delta_wavelength_pm']):.1f} pm", f"T_ref {float(min_row['T_ref_C']):.0f} C, sigma {float(min_row['sigma_intrinsic_si_MPa']):.0f} MPa")}
          {stat_card("Stress trend", f"{stress_slope:.3f} pm/MPa", "average endpoint slope at fixed T_ref")}
          {stat_card("Cooling trend", f"{temp_slope:.3f} pm/C", "average endpoint slope at fixed stress")}
        </div>
        <div class="callouts">
          <div class="callout">
            <b>Cooling-only slice</b>
            <p>At 0 MPa intrinsic stress, the shift is {zero_high:.1f} pm at T_ref = {temps[0]:.0f} C and {zero_low:.1f} pm at T_ref = {temps[-1]:.0f} C.</p>
          </div>
          <div class="callout">
            <b>Intrinsic-stress axis</b>
            <p>At each reference temperature, increasing the intrinsic-stress coordinate makes the DBR shift smaller by roughly {abs(stress_slope):.2f} pm per MPa.</p>
          </div>
          <div class="callout">
            <b>DBR convention</b>
            <p>The DBR uses a 50/50 average of the unperturbed and SiO2-insert perturbed sections at a 1550 nm design wavelength.</p>
          </div>
        </div>
      </div>
    </section>

    <section id="workflow">
      <div class="wrap two-col">
        <div>
          <h2>Project Workflow</h2>
          <p>The project combines a 2D plane-strain mechanical model with an approximate scalar optical mode solve. The geometry is a silicon rib/slab waveguide in SiO2: a 20 um wide simulation window, 5 um silicon handle, 1 um BOX, 3 um total device silicon, 1.8 um slab, 2.6 um rib width, and 5 um top cladding. The perturbed DBR section adds a centered SiO2 insert that is 2.2 um wide and 0.4 um deep.</p>
          <p>The mechanical solve produces elastic strain fields. Those fields are interpolated onto the optical grid, converted into TE-like and TM-like strain-optic index maps, and then used to re-solve the fundamental mode. The report focuses on the stress-coupled TE00 result.</p>
          <div class="formula">delta_inv_n2_TE = p11 * eps_xx + p12 * eps_yy
delta_n_TE = -0.5 * n0^3 * delta_inv_n2_TE * stress_scale</div>
        </div>
        <div>
          <h2>DBR Calculation</h2>
          <p>For every sweep point, the script solves two sections. The unperturbed section has no SiO2 rib insert, while the perturbed section includes the insert. Their TE00 effective indices are averaged with a 0.50 / 0.50 duty cycle.</p>
          <div class="formula">n_DBR = 0.5 * n_unperturbed + 0.5 * n_perturbed
lambda_stressed = 1550 nm * n_DBR_stressed / n_DBR_unstressed
delta_lambda = lambda_stressed - 1550 nm</div>
          <p>This isolates the stress/strain contribution to the DBR center wavelength. It does not include material thermo-optic dispersion or process-specific calibration unless those are added to the model inputs.</p>
        </div>
      </div>
    </section>

    <section id="sweep">
      <div class="wrap two-col">
        <div>
          <h2>Double Sweep</h2>
          <p>The sweep axes are reference temperature and intrinsic silicon stress. The use temperature is fixed at 25 C, so the thermal part is controlled by the cooldown delta T = 25 C - T_ref. A higher T_ref therefore means a larger negative cooldown delta.</p>
          <p>Intrinsic stress enters the starter model as an equivalent isotropic silicon eigenstrain. The total scalar eigenstrain used by the linear mechanical basis scaling is:</p>
          <div class="formula">eps_star = (alpha_Si - alpha_SiO2) * (T_use - T_ref)
         + sigma_intrinsic_Si / (3 * K_Si)</div>
          <p>Because the mechanical equations are linear in this scalar eigenstrain, the sweep solves one mechanical basis field per DBR section and scales it across the temperature/stress grid.</p>
        </div>
        <div>
          <h2>Axis Values</h2>
          <table>
            <tbody>
              <tr><th>Temperature axis</th><td>{html.escape(", ".join(f"{v:.0f}" for v in temps))} C</td></tr>
              <tr><th>Intrinsic stress axis</th><td>{html.escape(", ".join(f"{v:.0f}" for v in sigmas))} MPa</td></tr>
              <tr><th>Strain source</th><td>elastic strain</td></tr>
              <tr><th>Design wavelength</th><td>{DESIGN_WAVELENGTH_NM:.0f} nm</td></tr>
              <tr><th>Photoelastic coefficients</th><td>Si and SiO2 approximations from src/solve_optical_modes.py</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <section id="plots">
      <div class="wrap">
        <h2>Sweep Plots</h2>
        <p>The line plots show the same DBR wavelength shift sliced both ways: shift versus reference temperature for each intrinsic-stress setting, and shift versus intrinsic stress for each reference temperature. The heatmaps show the two primary result surfaces.</p>
        {figure_grid([
          image_figure("DBR shift vs. reference temperature", "rslt/optical_modes/ppt_assets/dbr_shift_pm_vs_reference_temperature_by_stress.png", "Shift in pm, colored by intrinsic stress.", output_file),
          image_figure("DBR shift vs. intrinsic stress", "rslt/optical_modes/ppt_assets/dbr_shift_pm_vs_intrinsic_stress_by_temperature.png", "Shift in pm, colored by reference temperature.", output_file),
          image_figure("DBR wavelength-shift heatmap", "rslt/optical_modes/dbr_shift_temperature_intrinsic_stress_heatmap.png", "Center-wavelength shift over the double sweep.", output_file),
          image_figure("TE00 delta-neff heatmap", "rslt/optical_modes/te00_delta_neff_temperature_intrinsic_stress_heatmap.png", "Effective-index perturbation over the double sweep.", output_file),
          image_figure("Zero-stress TE00 delta neff slice", "rslt/optical_modes/te00_delta_neff_vs_reference_temperature.png", "Unperturbed, perturbed, and 50/50 DBR-average TE00 delta neff.", output_file),
          image_figure("Zero-stress DBR shift slice", "rslt/optical_modes/dbr_center_wavelength_shift_vs_reference_temperature.png", "DBR center-wavelength shift at the near-zero intrinsic-stress slice.", output_file),
        ])}
      </div>
    </section>

    <section id="fields">
      <div class="wrap">
        <h2>Representative Fields</h2>
        <p>{html.escape(rep_title) if rep_title else "No complete representative per-point image folder was found."}</p>
        {figure_grid([
          image_figure("Material map", "rslt/material_map.png", "Mechanical material regions.", output_file),
          image_figure("Optical index map", "rslt/optical_modes/index_map.png", "Unperturbed optical refractive-index map.", output_file),
          image_figure("TE stress-coupled delta n", "rslt/optical_modes/TE_stress_coupled_delta_n.png", "Photoelastic index perturbation for TE-like polarization.", output_file),
          image_figure("TE stress-coupled intensity", "rslt/optical_modes/TE_stress_coupled_fundamental_intensity.png", "Stress-coupled TE fundamental intensity.", output_file),
        ])}
        {stress_figures}
        {optical_figures}
      </div>
    </section>

    <section id="data">
      <div class="wrap">
        <h2>Sweep Data</h2>
        <p>The shifted wavelength column is computed as 1550 nm plus the reported DBR shift. Rows are ordered by descending reference temperature and ascending intrinsic stress.</p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>T_ref [C]</th>
                <th>cooldown dT [C]</th>
                <th>intrinsic stress [MPa]</th>
                <th>eps_star</th>
                <th>DBR shift [pm]</th>
                <th>lambda_stressed [nm]</th>
              </tr>
            </thead>
            <tbody>
{build_rows_table(rows)}
            </tbody>
          </table>
        </div>
      </div>
    </section>

    <section id="caveats">
      <div class="wrap two-col">
        <div>
          <h2>Caveats</h2>
          <p>The material constants and strain-optic coefficients are useful for workflow development, but foundry/process-specific values should replace them before using the absolute wavelength shifts as signoff data.</p>
          <p>The optical model intentionally excludes the high-index silicon handle by default so substrate modes do not dominate the rib modes. The reported response is the elastic-strain photoelastic perturbation, not a full thermal-optic wavelength budget.</p>
        </div>
        <div>
          <h2>Regeneration</h2>
          <p>Rebuild the sweep assets before regenerating this report if the CSV has been overwritten by a partial run.</p>
          <div class="formula">python tools/prepare_sweep_ppt_assets.py --section perturbed
python tools/build_html_report.py</div>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <div class="wrap">
      Generated from {html.escape(str(MANIFEST_FILE.relative_to(ROOT)))}. Report file: {html.escape(str(output_file.relative_to(ROOT)))}.
    </div>
  </footer>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Build the static HTML DBR sweep report.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE)
    parser.add_argument("--csv", type=Path, default=CSV_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    csv_path = args.csv if args.csv.is_absolute() else ROOT / args.csv
    output_file = args.output if args.output.is_absolute() else ROOT / args.output

    manifest, rows = read_manifest(manifest_path)
    if not rows:
        raise RuntimeError(f"No slides found in manifest: {manifest_path}")

    current_csv_rows = read_current_csv(csv_path)
    html_text = build_html(manifest, rows, current_csv_rows, output_file)
    html_text = "\n".join(line.rstrip() for line in html_text.splitlines()) + "\n"
    output_file.write_text(html_text, encoding="utf-8", newline="\n")
    print(f"Saved HTML report: {output_file}")
    print(f"Sweep points in report: {len(rows)}")


if __name__ == "__main__":
    main()
