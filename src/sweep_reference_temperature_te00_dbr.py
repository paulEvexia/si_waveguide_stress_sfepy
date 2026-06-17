import argparse
import csv
import gc
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_si_waveguide_plane_strain as stress
import solve_optical_modes as optical


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RSLT = ROOT / "rslt" / "optical_modes"
RESULTS.mkdir(exist_ok=True)
RSLT.mkdir(parents=True, exist_ok=True)

SWEEP_CSV_FILE = (
    RESULTS
    / "te00_neff_and_dbr_shift_vs_reference_temperature_and_intrinsic_stress.csv"
)
DELTA_NEFF_PNG_FILE = RSLT / "te00_delta_neff_vs_reference_temperature.png"
DBR_SHIFT_PNG_FILE = RSLT / "dbr_center_wavelength_shift_vs_reference_temperature.png"
DELTA_NEFF_HEATMAP_PNG_FILE = RSLT / "te00_delta_neff_temperature_intrinsic_stress_heatmap.png"
DBR_SHIFT_HEATMAP_PNG_FILE = RSLT / "dbr_shift_temperature_intrinsic_stress_heatmap.png"
SWEEP_POINT_PNG_ROOT = RSLT / "sweep_points"

DESIGN_WAVELENGTH_NM = 1550.0
DBR_DUTY_CYCLE_PERTURBED = 0.50
DBR_DUTY_CYCLE_UNPERTURBED = 1.0 - DBR_DUTY_CYCLE_PERTURBED
DEFAULT_MECHANICAL_BASIS_EPS_STAR = 1.0e-3
DEFAULT_SWEEP_STRESS_NX = 200
DEFAULT_SWEEP_STRESS_NY = 140
DEFAULT_POINT_PNG_DPI = 150
DEFAULT_POINT_PNG_MAX_PIXELS = 500_000
STRAIN_PNG_ABS_LIMIT = 3.0e-3
DBR_SECTIONS = [
    {
        "section_name": "unperturbed",
        "enable_sio2_insert": False,
        "duty_cycle": DBR_DUTY_CYCLE_UNPERTURBED,
    },
    {
        "section_name": "perturbed",
        "enable_sio2_insert": True,
        "duty_cycle": DBR_DUTY_CYCLE_PERTURBED,
    },
]


def build_temperature_values(start_c, stop_c, step_c):
    return build_sweep_values(start_c, stop_c, step_c, "Temperature")


def build_intrinsic_stress_values(start_mpa, stop_mpa, step_mpa):
    return build_sweep_values(start_mpa, stop_mpa, step_mpa, "Intrinsic stress")


def build_sweep_values(start, stop, step, label):
    if step == 0:
        raise ValueError(f"{label} step cannot be zero.")

    values = []
    current = start

    if step > 0:
        while current <= stop + 1e-12:
            values.append(float(current))
            current += step
    else:
        while current >= stop - 1e-12:
            values.append(float(current))
            current += step

    if not values:
        raise ValueError(
            f"{label} sweep produced no values. Check start, stop, and step."
        )

    return values


def rows_to_strain_grid(rows, strain_source):
    if strain_source not in {"elastic", "total"}:
        raise ValueError("strain_source must be 'elastic' or 'total'")

    if strain_source == "elastic":
        exx_column = "eps_elastic_xx"
        eyy_column = "eps_elastic_yy"
        exy_column = "eps_elastic_xy"
    else:
        exx_column = "eps_xx"
        eyy_column = "eps_yy"
        exy_column = "eps_xy"

    x_values = sorted({row["x_um"] for row in rows})
    y_values = sorted({row["y_um"] for row in rows})
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    shape = (len(y_values), len(x_values))

    eps_xx = np.zeros(shape, dtype=float)
    eps_yy = np.zeros(shape, dtype=float)
    eps_xy = np.zeros(shape, dtype=float)

    for row in rows:
        j = y_index[row["y_um"]]
        i = x_index[row["x_um"]]
        eps_xx[j, i] = row[exx_column]
        eps_yy[j, i] = row[eyy_column]
        eps_xy[j, i] = row[exy_column]

    return {
        "x": np.array(x_values, dtype=float),
        "y": np.array(y_values, dtype=float),
        "eps_xx": eps_xx,
        "eps_yy": eps_yy,
        "eps_xy": eps_xy,
    }


def solve_te00(grid, modes, analysis_state, strain_source="none", stress_scale=0.0):
    selected, _field = solve_mode(
        grid,
        "TE",
        modes,
        analysis_state=analysis_state,
        strain_source=strain_source,
        stress_scale=stress_scale,
    )
    return selected


def solve_mode(grid, polarization, modes, analysis_state, strain_source="none", stress_scale=0.0):
    solver = optical.solve_te_modes if polarization == "TE" else optical.solve_tm_modes
    _rows, selected, field = optical.solve_and_select(
        grid,
        polarization,
        solver,
        modes,
        analysis_state=analysis_state,
        strain_source=strain_source,
        stress_scale=stress_scale,
    )
    return selected, field


def configure_mechanical_solver(args):
    if args.mechanical_linear_solver == "direct":
        stress.linear_solver_kind = "ls.scipy_direct"
        stress.linear_solver_options = {}
        return

    stress.linear_solver_kind = "ls.scipy_iterative"
    stress.linear_solver_options = {
        "method": args.mechanical_iterative_method,
        "i_max": args.mechanical_iterative_i_max,
        "eps_a": args.mechanical_iterative_eps_a,
        "eps_r": args.mechanical_iterative_eps_r,
    }


def run_mechanical_worker(args):
    configure_mechanical_solver(args)
    stress.enable_sio2_insert = bool(args.worker_enable_sio2_insert)
    stress.T_ref_C = float(args.worker_t_ref)
    stress.sigma_intrinsic_si_MPa = float(args.worker_intrinsic_stress)

    strain_grid = stress.solve_case_strain_grid(
        nx_cells=args.stress_nx,
        ny_cells=args.stress_ny,
        strain_source=args.strain_source,
    )

    np.savez_compressed(
        args.worker_output,
        x=strain_grid["x"],
        y=strain_grid["y"],
        eps_xx=strain_grid["eps_xx"],
        eps_yy=strain_grid["eps_yy"],
        eps_xy=strain_grid["eps_xy"],
    )


def solve_strain_grid_in_subprocess(section, t_ref_c, intrinsic_stress_mpa, args):
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        output_path = Path(tmp.name)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mechanical-worker",
        "--worker-t-ref",
        str(float(t_ref_c)),
        "--worker-intrinsic-stress",
        str(float(intrinsic_stress_mpa)),
        "--worker-enable-sio2-insert",
        "1" if section["enable_sio2_insert"] else "0",
        "--worker-output",
        str(output_path),
        "--stress-nx",
        str(args.stress_nx),
        "--stress-ny",
        str(args.stress_ny),
        "--strain-source",
        args.strain_source,
        "--mechanical-linear-solver",
        args.mechanical_linear_solver,
        "--mechanical-iterative-method",
        args.mechanical_iterative_method,
        "--mechanical-iterative-i-max",
        str(args.mechanical_iterative_i_max),
        "--mechanical-iterative-eps-a",
        str(args.mechanical_iterative_eps_a),
        "--mechanical-iterative-eps-r",
        str(args.mechanical_iterative_eps_r),
    ]

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr)
            raise RuntimeError(
                "Mechanical worker failed for "
                f"{section['section_name']} section, "
                f"T_ref={t_ref_c:.1f} C, "
                f"intrinsic stress={intrinsic_stress_mpa:.1f} MPa"
            )

        with np.load(output_path) as data:
            return {
                "x": data["x"].copy(),
                "y": data["y"].copy(),
                "eps_xx": data["eps_xx"].copy(),
                "eps_yy": data["eps_yy"].copy(),
                "eps_xy": data["eps_xy"].copy(),
            }

    finally:
        output_path.unlink(missing_ok=True)


def solve_strain_grid_in_process(t_ref_c, intrinsic_stress_mpa, args):
    stress.T_ref_C = float(t_ref_c)
    stress.sigma_intrinsic_si_MPa = float(intrinsic_stress_mpa)
    configure_mechanical_solver(args)
    strain_grid = stress.solve_case_strain_grid(
        nx_cells=args.stress_nx,
        ny_cells=args.stress_ny,
        strain_source=args.strain_source,
    )
    gc.collect()
    return strain_grid


def eps_star_for_sweep_point(t_ref_c, intrinsic_stress_mpa):
    dT = stress.T_use_C - float(t_ref_c)
    eps_cte = (stress.alpha_si - stress.alpha_sio2) * dT
    eps_intrinsic = float(intrinsic_stress_mpa) / (3.0 * stress.K_si)
    return eps_cte + eps_intrinsic


def basis_intrinsic_stress_mpa(basis_eps_star):
    return float(basis_eps_star) * 3.0 * stress.K_si


def scale_strain_grid(strain_grid, scale):
    return {
        "x": strain_grid["x"],
        "y": strain_grid["y"],
        "eps_xx": strain_grid["eps_xx"] * scale,
        "eps_yy": strain_grid["eps_yy"] * scale,
        "eps_xy": strain_grid["eps_xy"] * scale,
    }


def file_token(value):
    text = f"{float(value):.1f}"
    return text.replace("-", "m").replace(".", "p")


def sweep_point_dir(t_ref_c, intrinsic_stress_mpa, section_name):
    return (
        SWEEP_POINT_PNG_ROOT
        / f"Tref_{file_token(t_ref_c)}C_sigma_{file_token(intrinsic_stress_mpa)}MPa"
        / section_name
    )


def decimate_plot_array(x, y, values, max_pixels):
    if max_pixels <= 0 or values.size <= max_pixels:
        return x, y, values, 1

    stride = int(np.ceil(np.sqrt(values.size / max_pixels)))
    return x[::stride], y[::stride], values[::stride, ::stride], stride


def plot_array_png(
    x,
    y,
    values,
    label,
    title,
    path,
    cmap,
    symmetric=True,
    fixed_abs_limit=None,
    dpi=DEFAULT_POINT_PNG_DPI,
    max_pixels=DEFAULT_POINT_PNG_MAX_PIXELS,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    x_plot, y_plot, values_plot, stride = decimate_plot_array(
        x,
        y,
        values,
        max_pixels,
    )
    color_limits = {}

    if fixed_abs_limit is not None:
        fixed_abs_limit = float(fixed_abs_limit)
        if fixed_abs_limit <= 0.0:
            raise ValueError("fixed_abs_limit must be positive.")
        color_limits = {"vmin": -fixed_abs_limit, "vmax": fixed_abs_limit}
    elif symmetric:
        max_abs = np.nanmax(np.abs(values_plot))
        if np.isfinite(max_abs) and max_abs > 0.0:
            color_limits = {"vmin": -max_abs, "vmax": max_abs}

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        im = ax.imshow(
            values_plot,
            extent=[
                float(np.min(x_plot)),
                float(np.max(x_plot)),
                float(np.min(y_plot)),
                float(np.max(y_plot)),
            ],
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            cmap=cmap,
            **color_limits,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(label)
        ax.set_xlabel("x [um]")
        ax.set_ylabel("y [um]")
        if stride > 1:
            title = f"{title} (plot stride {stride})"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
    finally:
        if fig is not None:
            plt.close(fig)
        plt.close("all")
        del values, values_plot
        gc.collect()


def mechanical_si_mask(section, strain_grid):
    original_insert = stress.enable_sio2_insert
    xx, yy = np.meshgrid(strain_grid["x"], strain_grid["y"])
    mask = np.zeros(xx.shape, dtype=bool)

    try:
        stress.enable_sio2_insert = section["enable_sio2_insert"]

        for index, _ in np.ndenumerate(xx):
            mask[index] = stress.inside_si_region(float(xx[index]), float(yy[index]))

    finally:
        stress.enable_sio2_insert = original_insert

    return mask


def elastic_strain_for_stress(section, strain_grid, eps_star, strain_source):
    eps_xx = np.array(strain_grid["eps_xx"], dtype=float, copy=True)
    eps_yy = np.array(strain_grid["eps_yy"], dtype=float, copy=True)
    eps_xy = np.array(strain_grid["eps_xy"], dtype=float, copy=True)

    if strain_source == "total":
        si_mask = mechanical_si_mask(section, strain_grid)
        eps_xx[si_mask] -= eps_star
        eps_yy[si_mask] -= eps_star

    return eps_xx, eps_yy, eps_xy


def stress_components_from_strain(section, strain_grid, eps_star, strain_source):
    eps_xx, eps_yy, eps_xy = elastic_strain_for_stress(
        section,
        strain_grid,
        eps_star,
        strain_source,
    )
    si_mask = mechanical_si_mask(section, strain_grid)
    gamma_xy = 2.0 * eps_xy

    sigma_xx = np.zeros_like(eps_xx)
    sigma_yy = np.zeros_like(eps_yy)
    sigma_xy = np.zeros_like(eps_xy)

    for mask, D in [(si_mask, stress.D_si), (~si_mask, stress.D_sio2)]:
        sigma_xx[mask] = (
            D[0, 0] * eps_xx[mask]
            + D[0, 1] * eps_yy[mask]
            + D[0, 2] * gamma_xy[mask]
        )
        sigma_yy[mask] = (
            D[1, 0] * eps_xx[mask]
            + D[1, 1] * eps_yy[mask]
            + D[1, 2] * gamma_xy[mask]
        )
        sigma_xy[mask] = (
            D[2, 0] * eps_xx[mask]
            + D[2, 1] * eps_yy[mask]
            + D[2, 2] * gamma_xy[mask]
        )

    return {
        "sigma_xx_MPa": sigma_xx,
        "sigma_yy_MPa": sigma_yy,
        "sigma_xy_MPa": sigma_xy,
    }


def save_mechanical_point_pngs(
    section,
    t_ref_c,
    intrinsic_stress_mpa,
    strain_grid,
    eps_star,
    args,
):
    out_dir = sweep_point_dir(t_ref_c, intrinsic_stress_mpa, section["section_name"])
    title_prefix = (
        f"{section['section_name']}, T_ref={t_ref_c:.1f} C, "
        f"sigma_i={intrinsic_stress_mpa:.1f} MPa"
    )
    stress_fields = stress_components_from_strain(
        section,
        strain_grid,
        eps_star,
        args.strain_source,
    )

    for metric, values in stress_fields.items():
        plot_array_png(
            strain_grid["x"],
            strain_grid["y"],
            values,
            metric,
            f"{title_prefix}: {metric}",
            out_dir / f"stress_{metric}.png",
            stress.RGB_DISTRIBUTION_CMAP,
            symmetric=True,
            dpi=args.point_png_dpi,
            max_pixels=args.point_png_max_pixels,
        )

    for metric in ["eps_xx", "eps_yy", "eps_xy"]:
        plot_array_png(
            strain_grid["x"],
            strain_grid["y"],
            strain_grid[metric],
            f"{args.strain_source} strain {metric}",
            f"{title_prefix}: {args.strain_source} strain {metric}",
            out_dir / f"strain_{args.strain_source}_{metric}.png",
            stress.RGB_DISTRIBUTION_CMAP,
            symmetric=True,
            fixed_abs_limit=STRAIN_PNG_ABS_LIMIT,
            dpi=args.point_png_dpi,
            max_pixels=args.point_png_max_pixels,
        )


def save_optical_point_pngs(
    section,
    t_ref_c,
    intrinsic_stress_mpa,
    coupled,
    te_field,
    tm_field,
    te_selected,
    tm_selected,
    args,
):
    out_dir = sweep_point_dir(t_ref_c, intrinsic_stress_mpa, section["section_name"])
    title_prefix = (
        f"{section['section_name']}, T_ref={t_ref_c:.1f} C, "
        f"sigma_i={intrinsic_stress_mpa:.1f} MPa"
    )

    for polarization, field, selected, grid in [
        ("TE", te_field, te_selected, coupled["TE"]),
        ("TM", tm_field, tm_selected, coupled["TM"]),
    ]:
        intensity = np.abs(optical.normalize_field(field)) ** 2
        plot_array_png(
            grid["x"],
            grid["y"],
            intensity,
            "Normalized intensity",
            f"{title_prefix}: {polarization}00 intensity, neff={selected['neff']:.6f}",
            out_dir / f"optical_{polarization}00_intensity.png",
            "inferno",
            symmetric=False,
            dpi=args.point_png_dpi,
            max_pixels=args.point_png_max_pixels,
        )

        if args.save_point_mode_fields:
            normalized_field = optical.normalize_field(field)
            plot_array_png(
                grid["x"],
                grid["y"],
                normalized_field,
                "Normalized field",
                f"{title_prefix}: {polarization}00 field, neff={selected['neff']:.6f}",
                out_dir / f"optical_{polarization}00_field.png",
                "RdBu_r",
                symmetric=True,
                dpi=args.point_png_dpi,
                max_pixels=args.point_png_max_pixels,
            )

    for polarization, delta_n in [
        ("TE", coupled["delta_n_te"]),
        ("TM", coupled["delta_n_tm"]),
    ]:
        plot_array_png(
            coupled[polarization]["x"],
            coupled[polarization]["y"],
            delta_n,
            f"delta n, {polarization}-like",
            f"{title_prefix}: {polarization}-like delta n",
            out_dir / f"optical_{polarization}_delta_n.png",
            "RdBu_r",
            symmetric=True,
            dpi=args.point_png_dpi,
            max_pixels=args.point_png_max_pixels,
        )


def should_save_point_pngs(args, point_index):
    if not args.save_point_pngs:
        return False

    return (point_index - 1) % args.point_png_every == 0


def solve_section_basis_strain_grid(section, args, basis_cache):
    cache_key = section["section_name"]
    if cache_key in basis_cache:
        return basis_cache[cache_key]

    basis_eps_star = args.mechanical_basis_eps_star
    if basis_eps_star == 0:
        raise ValueError("--mechanical-basis-eps-star cannot be zero.")

    basis_t_ref_c = stress.T_use_C
    basis_intrinsic_mpa = basis_intrinsic_stress_mpa(basis_eps_star)
    print(
        f"  mechanical basis solve: section = {section['section_name']}, "
        f"insert = {section['enable_sio2_insert']}, "
        f"eps_star = {basis_eps_star:g}"
    )

    if args.mechanical_subprocess:
        basis_grid = solve_strain_grid_in_subprocess(
            section,
            basis_t_ref_c,
            basis_intrinsic_mpa,
            args,
        )
    else:
        original_stress_insert = stress.enable_sio2_insert
        try:
            stress.enable_sio2_insert = section["enable_sio2_insert"]
            basis_grid = solve_strain_grid_in_process(
                basis_t_ref_c,
                basis_intrinsic_mpa,
                args,
            )
        finally:
            stress.enable_sio2_insert = original_stress_insert

    basis_cache[cache_key] = basis_grid
    return basis_grid


def solve_section_te00(
    section,
    t_ref_c,
    intrinsic_stress_mpa,
    point_index,
    args,
    unstressed_cache,
    basis_cache,
):
    original_stress_insert = stress.enable_sio2_insert
    original_optical_insert = optical.enable_sio2_insert
    original_t_ref = stress.T_ref_C
    original_intrinsic_stress = stress.sigma_intrinsic_si_MPa

    try:
        stress.enable_sio2_insert = section["enable_sio2_insert"]
        optical.enable_sio2_insert = section["enable_sio2_insert"]

        optical_grid = optical.build_grid(
            args.optical_dx,
            args.optical_dy,
            include_handle=args.include_handle,
        )
        cache_key = section["section_name"]

        if cache_key not in unstressed_cache:
            unstressed_cache[cache_key] = solve_te00(
                optical_grid,
                args.modes,
                analysis_state="unstressed",
            )

        unstressed = unstressed_cache[cache_key]
        basis_grid = solve_section_basis_strain_grid(section, args, basis_cache)
        eps_star = eps_star_for_sweep_point(t_ref_c, intrinsic_stress_mpa)
        strain_grid = scale_strain_grid(
            basis_grid,
            eps_star / args.mechanical_basis_eps_star,
        )

        coupled = optical.build_stress_coupled_grids_from_strain_grid(
            optical_grid,
            strain_grid,
            args.stress_scale,
        )
        stressed, te_field = solve_mode(
            coupled["TE"],
            "TE",
            args.modes,
            analysis_state="stress_coupled",
            strain_source=args.strain_source,
            stress_scale=args.stress_scale,
        )
        tm_selected = None
        tm_field = None

        if should_save_point_pngs(args, point_index):
            tm_selected, tm_field = solve_mode(
                coupled["TM"],
                "TM",
                args.modes,
                analysis_state="stress_coupled",
                strain_source=args.strain_source,
                stress_scale=args.stress_scale,
            )
            save_mechanical_point_pngs(
                section,
                t_ref_c,
                intrinsic_stress_mpa,
                strain_grid,
                eps_star,
                args,
            )
            save_optical_point_pngs(
                section,
                t_ref_c,
                intrinsic_stress_mpa,
                coupled,
                te_field,
                tm_field,
                stressed,
                tm_selected,
                args,
            )

        result = {
            "section_name": section["section_name"],
            "enable_sio2_insert": section["enable_sio2_insert"],
            "duty_cycle": section["duty_cycle"],
            "optical_grid": optical_grid,
            "unstressed": unstressed,
            "stressed": stressed,
            "neff_unstressed": unstressed["neff"],
            "neff_stressed": stressed["neff"],
            "delta_neff": stressed["neff"] - unstressed["neff"],
            "eps_star": eps_star,
        }
        del strain_grid, coupled, te_field, tm_field
        gc.collect()
        return result

    finally:
        stress.enable_sio2_insert = original_stress_insert
        optical.enable_sio2_insert = original_optical_insert
        stress.T_ref_C = original_t_ref
        stress.sigma_intrinsic_si_MPa = original_intrinsic_stress


def sweep_key_from_values(t_ref_c, intrinsic_stress_mpa):
    return (round(float(t_ref_c), 9), round(float(intrinsic_stress_mpa), 9))


def sweep_key_from_row(row):
    return sweep_key_from_values(row["T_ref_C"], row["sigma_intrinsic_si_MPa"])


def coerce_csv_value(value):
    if value == "":
        return value

    try:
        as_float = float(value)
    except ValueError:
        return value

    if as_float.is_integer() and value.strip().lower() not in {"nan", "inf", "-inf"}:
        return int(as_float)

    return as_float


def coerce_csv_row(row):
    return {key: coerce_csv_value(value) for key, value in row.items()}


def read_existing_sweep_rows():
    if not SWEEP_CSV_FILE.exists() or SWEEP_CSV_FILE.stat().st_size == 0:
        return []

    with open(SWEEP_CSV_FILE, newline="", encoding="utf-8") as f:
        return [coerce_csv_row(row) for row in csv.DictReader(f)]


def values_match(value_a, value_b, tol=1e-12):
    try:
        return abs(float(value_a) - float(value_b)) <= tol
    except (TypeError, ValueError):
        return value_a == value_b


def row_matches_run_settings(row, args):
    required_columns = {
        "mechanical_basis_eps_star",
        "eps_star_total",
        "stress_mesh_nx",
        "stress_mesh_ny",
        "optical_dx_um",
        "optical_dy_um",
        "strain_source",
        "stress_scale",
        "include_handle",
    }

    if not required_columns.issubset(row):
        return False

    return (
        int(row["stress_mesh_nx"]) == int(args.stress_nx)
        and int(row["stress_mesh_ny"]) == int(args.stress_ny)
        and values_match(row["optical_dx_um"], args.optical_dx)
        and values_match(row["optical_dy_um"], args.optical_dy)
        and row["strain_source"] == args.strain_source
        and values_match(row["stress_scale"], args.stress_scale)
        and int(row["include_handle"]) == int(args.include_handle)
        and values_match(
            row["mechanical_basis_eps_star"],
            args.mechanical_basis_eps_star,
        )
    )


def append_sweep_csv_row(row):
    write_header = not SWEEP_CSV_FILE.exists() or SWEEP_CSV_FILE.stat().st_size == 0

    with open(SWEEP_CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_temperature_intrinsic_stress_sweep(
    temperatures,
    intrinsic_stresses,
    args,
    existing_rows=None,
):
    rows = list(existing_rows or [])
    completed_keys = {sweep_key_from_row(row) for row in rows}
    unstressed_cache = {}
    basis_cache = {}
    point_index = 0

    for t_ref_c in temperatures:
        for intrinsic_stress_mpa in intrinsic_stresses:
            point_index += 1
            key = sweep_key_from_values(t_ref_c, intrinsic_stress_mpa)
            if key in completed_keys:
                print(
                    f"Skipping completed point: T_ref = {t_ref_c:.1f} C, "
                    f"intrinsic Si stress = {intrinsic_stress_mpa:.1f} MPa"
                )
                continue

            print(
                f"DBR period: T_ref = {t_ref_c:.1f} C, "
                f"intrinsic Si stress = {intrinsic_stress_mpa:.1f} MPa"
            )
            section_results = {}

            for section in DBR_SECTIONS:
                print(
                    f"  section = {section['section_name']}, "
                    f"insert = {section['enable_sio2_insert']}, "
                    f"duty = {section['duty_cycle']:.3f}"
                )
                result = solve_section_te00(
                    section,
                    t_ref_c,
                    intrinsic_stress_mpa,
                    point_index,
                    args,
                    unstressed_cache,
                    basis_cache,
                )
                section_results[section["section_name"]] = result

            unperturbed = section_results["unperturbed"]
            perturbed = section_results["perturbed"]
            neff_average_unstressed = (
                DBR_DUTY_CYCLE_UNPERTURBED * unperturbed["neff_unstressed"]
                + DBR_DUTY_CYCLE_PERTURBED * perturbed["neff_unstressed"]
            )
            neff_average_stressed = (
                DBR_DUTY_CYCLE_UNPERTURBED * unperturbed["neff_stressed"]
                + DBR_DUTY_CYCLE_PERTURBED * perturbed["neff_stressed"]
            )
            delta_neff_average = neff_average_stressed - neff_average_unstressed
            dbr_period_um = (
                (DESIGN_WAVELENGTH_NM / 1000.0)
                / (2.0 * neff_average_unstressed)
            )
            dbr_center_wavelength_stressed_nm = (
                DESIGN_WAVELENGTH_NM
                * neff_average_stressed
                / neff_average_unstressed
            )
            dbr_delta_wavelength_nm = (
                dbr_center_wavelength_stressed_nm - DESIGN_WAVELENGTH_NM
            )

            optical_grid = unperturbed["optical_grid"]
            row = {
                "T_ref_C": t_ref_c,
                "sigma_intrinsic_si_MPa": intrinsic_stress_mpa,
                "T_use_C": stress.T_use_C,
                "cooldown_delta_T_C": stress.T_use_C - t_ref_c,
                "strain_source": args.strain_source,
                "stress_scale": args.stress_scale,
                "mechanical_basis_eps_star": args.mechanical_basis_eps_star,
                "eps_star_total": eps_star_for_sweep_point(
                    t_ref_c,
                    intrinsic_stress_mpa,
                ),
                "stress_mesh_nx": args.stress_nx,
                "stress_mesh_ny": args.stress_ny,
                "stress_mesh_dx_um": stress.Lx / args.stress_nx,
                "stress_mesh_dy_um": stress.Ly / args.stress_ny,
                "optical_dx_um": optical_grid["dx_um"],
                "optical_dy_um": optical_grid["dy_um"],
                "optical_grid_nx": optical_grid["nx"],
                "optical_grid_ny": optical_grid["ny"],
                "include_handle": int(args.include_handle),
                "dbr_duty_cycle_unperturbed": DBR_DUTY_CYCLE_UNPERTURBED,
                "dbr_duty_cycle_perturbed": DBR_DUTY_CYCLE_PERTURBED,
                "unperturbed_te00_neff_unstressed": unperturbed["neff_unstressed"],
                "unperturbed_te00_neff_stress_coupled": unperturbed["neff_stressed"],
                "unperturbed_te00_delta_neff": unperturbed["delta_neff"],
                "unperturbed_te00_unstressed_mode_index": unperturbed["unstressed"]["mode_index"],
                "unperturbed_te00_stressed_mode_index": unperturbed["stressed"]["mode_index"],
                "perturbed_te00_neff_unstressed": perturbed["neff_unstressed"],
                "perturbed_te00_neff_stress_coupled": perturbed["neff_stressed"],
                "perturbed_te00_delta_neff": perturbed["delta_neff"],
                "perturbed_te00_unstressed_mode_index": perturbed["unstressed"]["mode_index"],
                "perturbed_te00_stressed_mode_index": perturbed["stressed"]["mode_index"],
                "dbr_average_te00_neff_unstressed": neff_average_unstressed,
                "dbr_average_te00_neff_stress_coupled": neff_average_stressed,
                "dbr_average_te00_delta_neff": delta_neff_average,
                "dbr_average_te00_relative_delta_neff": (
                    delta_neff_average / neff_average_unstressed
                ),
                "dbr_design_wavelength_nm": DESIGN_WAVELENGTH_NM,
                "dbr_period_um": dbr_period_um,
                "dbr_center_wavelength_stressed_nm": dbr_center_wavelength_stressed_nm,
                "dbr_delta_wavelength_nm": dbr_delta_wavelength_nm,
                "dbr_delta_wavelength_pm": dbr_delta_wavelength_nm * 1000.0,
            }
            rows.append(row)
            completed_keys.add(key)
            append_sweep_csv_row(row)
            print(f"  saved completed point to {SWEEP_CSV_FILE}")
            del section_results, unperturbed, perturbed, optical_grid
            gc.collect()

    return rows


def write_sweep_csv(rows):
    if not rows:
        SWEEP_CSV_FILE.unlink(missing_ok=True)
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(SWEEP_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_sweep(rows):
    plot_zero_intrinsic_slice(rows)
    plot_heatmap(
        rows,
        "dbr_average_te00_delta_neff",
        DELTA_NEFF_HEATMAP_PNG_FILE,
        "TE00 Delta neff vs. Reference Temperature and Intrinsic Stress",
        "50/50 DBR average TE00 delta neff",
    )
    plot_heatmap(
        rows,
        "dbr_delta_wavelength_nm",
        DBR_SHIFT_HEATMAP_PNG_FILE,
        "DBR Center-Wavelength Shift vs. Reference Temperature and Intrinsic Stress",
        "DBR center wavelength shift [nm]",
    )


def axis_edges(values):
    values = np.array(sorted(values), dtype=float)

    if len(values) == 1:
        half_width = max(abs(values[0]) * 0.05, 1.0)
        return np.array([values[0] - half_width, values[0] + half_width])

    midpoints = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate(([first], midpoints, [last]))


def sweep_matrix(rows, value_key):
    temperatures = sorted({row["T_ref_C"] for row in rows})
    intrinsic_stresses = sorted({row["sigma_intrinsic_si_MPa"] for row in rows})
    temp_index = {value: index for index, value in enumerate(temperatures)}
    stress_index = {value: index for index, value in enumerate(intrinsic_stresses)}
    data = np.full((len(intrinsic_stresses), len(temperatures)), np.nan, dtype=float)

    for row in rows:
        j = stress_index[row["sigma_intrinsic_si_MPa"]]
        i = temp_index[row["T_ref_C"]]
        data[j, i] = row[value_key]

    return temperatures, intrinsic_stresses, data


def plot_heatmap(rows, value_key, png_file, title, colorbar_label):
    temperatures, intrinsic_stresses, data = sweep_matrix(rows, value_key)
    temp_edges = axis_edges(temperatures)
    stress_edges = axis_edges(intrinsic_stresses)

    max_abs = np.nanmax(np.abs(data))
    color_limits = {}
    if np.isfinite(max_abs) and max_abs > 0:
        color_limits = {"vmin": -max_abs, "vmax": max_abs}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.pcolormesh(
        temp_edges,
        stress_edges,
        data,
        shading="auto",
        cmap="coolwarm",
        **color_limits,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    ax.set_xlabel("Reference temperature [C]")
    ax.set_ylabel("Intrinsic Si stress [MPa]")
    ax.set_title(title)
    ax.set_xlim(min(temp_edges), max(temp_edges))
    ax.set_ylim(min(stress_edges), max(stress_edges))
    fig.tight_layout()
    fig.savefig(png_file, dpi=300)
    plt.close(fig)


def plot_zero_intrinsic_slice(rows):
    intrinsic_stresses = sorted({row["sigma_intrinsic_si_MPa"] for row in rows})
    slice_stress = min(intrinsic_stresses, key=abs)
    zero_rows = sorted(
        [
            row
            for row in rows
            if np.isclose(row["sigma_intrinsic_si_MPa"], slice_stress, atol=1e-12)
        ],
        key=lambda row: row["T_ref_C"],
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [row["T_ref_C"] for row in zero_rows],
        [row["dbr_average_te00_delta_neff"] for row in zero_rows],
        marker="o",
        label="50/50 DBR average",
    )
    ax.plot(
        [row["T_ref_C"] for row in zero_rows],
        [row["unperturbed_te00_delta_neff"] for row in zero_rows],
        marker="s",
        linestyle="--",
        label="unperturbed section",
    )
    ax.plot(
        [row["T_ref_C"] for row in zero_rows],
        [row["perturbed_te00_delta_neff"] for row in zero_rows],
        marker="^",
        linestyle="--",
        label="perturbed section",
    )

    ax.set_xlabel("Reference temperature [C]")
    ax.set_ylabel("TE00 delta neff")
    ax.set_title(
        "Stress/Strain-Induced TE00 Effective-Index Shift "
        f"at {slice_stress:.1f} MPa Intrinsic Stress"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DELTA_NEFF_PNG_FILE, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [row["T_ref_C"] for row in zero_rows],
        [row["dbr_delta_wavelength_nm"] for row in zero_rows],
        marker="o",
        label="50/50 DBR period",
    )

    ax.set_xlabel("Reference temperature [C]")
    ax.set_ylabel("DBR center wavelength shift [nm]")
    ax.set_title(
        "50/50 DBR Center-Wavelength Shift From Stress/Strain "
        f"at {slice_stress:.1f} MPa Intrinsic Stress"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DBR_SHIFT_PNG_FILE, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep reference temperature and intrinsic Si stress, then compute "
            "TE00 neff/DBR shift."
        )
    )
    parser.add_argument("--t-ref-start", type=float, default=1000.0)
    parser.add_argument("--t-ref-stop", type=float, default=400.0)
    parser.add_argument("--t-ref-step", type=float, default=-50.0)
    parser.add_argument("--intrinsic-start", type=float, default=-250.0)
    parser.add_argument("--intrinsic-stop", type=float, default=250.0)
    parser.add_argument("--intrinsic-step", type=float, default=100.0)
    parser.add_argument("--stress-nx", type=int, default=DEFAULT_SWEEP_STRESS_NX)
    parser.add_argument("--stress-ny", type=int, default=DEFAULT_SWEEP_STRESS_NY)
    parser.add_argument("--optical-dx", type=float, default=0.10)
    parser.add_argument("--optical-dy", type=float, default=0.10)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument(
        "--strain-source",
        choices=["elastic", "total"],
        default="elastic",
    )
    parser.add_argument("--stress-scale", type=float, default=1.0)
    parser.add_argument("--include-handle", action="store_true")
    parser.add_argument(
        "--save-point-pngs",
        action="store_true",
        help=(
            "Save stress, strain, TE00, and TM00 PNGs for sweep points under "
            "rslt/optical_modes/sweep_points/."
        ),
    )
    parser.add_argument(
        "--point-png-every",
        type=int,
        default=1,
        help="Save per-point PNGs every Nth sweep point when --save-point-pngs is used.",
    )
    parser.add_argument(
        "--point-png-dpi",
        type=int,
        default=DEFAULT_POINT_PNG_DPI,
        help="DPI for per-point PNGs.",
    )
    parser.add_argument(
        "--point-png-max-pixels",
        type=int,
        default=DEFAULT_POINT_PNG_MAX_PIXELS,
        help=(
            "Maximum array pixels rendered per per-point PNG. Larger arrays "
            "are strided for plotting only; calculations are not downsampled."
        ),
    )
    parser.add_argument(
        "--save-point-mode-fields",
        action="store_true",
        help=(
            "Also save signed TE00/TM00 field PNGs for each point. "
            "By default only mode intensity PNGs are saved."
        ),
    )
    parser.add_argument(
        "--keep-point-pngs",
        action="store_true",
        help=(
            "Keep existing rslt/optical_modes/sweep_points PNG folders when "
            "starting a non-resume run."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the existing sweep CSV and skip completed T/stress points.",
    )
    parser.add_argument(
        "--mechanical-subprocess",
        dest="mechanical_subprocess",
        action="store_true",
        help=(
            "Run each mechanical section solve in a fresh Python process "
            "to release SfePy memory between solves."
        ),
    )
    parser.add_argument(
        "--no-mechanical-subprocess",
        dest="mechanical_subprocess",
        action="store_false",
        help="Run mechanical solves in the main process.",
    )
    parser.set_defaults(mechanical_subprocess=True)
    parser.add_argument(
        "--mechanical-linear-solver",
        choices=["iterative", "direct"],
        default="direct",
        help=(
            "Mechanical linear solver used by the sweep. The default direct "
            "solver is reliable with the sweep's reduced basis mesh; use "
            "iterative for larger meshes that exceed direct-solver memory."
        ),
    )
    parser.add_argument(
        "--mechanical-iterative-method",
        default="cg",
        help="SciPy iterative method for the mechanical solve, for example cg or bicgstab.",
    )
    parser.add_argument("--mechanical-iterative-i-max", type=int, default=5000)
    parser.add_argument("--mechanical-iterative-eps-a", type=float, default=1e-10)
    parser.add_argument("--mechanical-iterative-eps-r", type=float, default=1e-8)
    parser.add_argument(
        "--mechanical-basis-eps-star",
        type=float,
        default=DEFAULT_MECHANICAL_BASIS_EPS_STAR,
        help=(
            "Si eigenstrain used for the mechanical basis solve. The sweep "
            "scales this basis field linearly for each temperature/stress point."
        ),
    )
    parser.add_argument(
        "--mechanical-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-t-ref",
        type=float,
        default=stress.T_ref_C,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-intrinsic-stress",
        type=float,
        default=stress.sigma_intrinsic_si_MPa,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-enable-sio2-insert",
        type=int,
        default=int(stress.enable_sio2_insert),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-output",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.point_png_every < 1:
        raise ValueError("--point-png-every must be >= 1.")
    if args.point_png_dpi < 50:
        raise ValueError("--point-png-dpi must be >= 50.")
    if args.point_png_max_pixels < 10_000:
        raise ValueError("--point-png-max-pixels must be >= 10000.")

    if args.mechanical_worker:
        if not args.worker_output:
            raise ValueError("--worker-output is required in mechanical-worker mode.")
        run_mechanical_worker(args)
        return

    if args.resume:
        all_existing_rows = read_existing_sweep_rows()
        existing_rows = [
            row for row in all_existing_rows if row_matches_run_settings(row, args)
        ]
        if len(existing_rows) != len(all_existing_rows):
            skipped = len(all_existing_rows) - len(existing_rows)
            print(
                f"Resume mode: ignored {skipped} existing row(s) with different "
                "mesh/optical settings or older schema."
            )
            write_sweep_csv(existing_rows)
    else:
        SWEEP_CSV_FILE.unlink(missing_ok=True)
        existing_rows = []
        if args.save_point_pngs and SWEEP_POINT_PNG_ROOT.exists() and not args.keep_point_pngs:
            try:
                shutil.rmtree(SWEEP_POINT_PNG_ROOT)
            except PermissionError as exc:
                print(
                    "WARNING: Could not clear existing per-point PNG folder because "
                    "Windows reports it is in use. Continuing and overwriting matching "
                    "PNG files where possible."
                )
                print(f"         Folder: {SWEEP_POINT_PNG_ROOT}")
                print(f"         Detail: {exc}")

    temperatures = build_temperature_values(
        args.t_ref_start,
        args.t_ref_stop,
        args.t_ref_step,
    )
    intrinsic_stresses = build_intrinsic_stress_values(
        args.intrinsic_start,
        args.intrinsic_stop,
        args.intrinsic_step,
    )
    print("TE00 stress/DBR reference-temperature and intrinsic-stress sweep")
    print("----------------------------------------------------------------")
    print(f"T_ref values [C] = {temperatures}")
    print(
        "Intrinsic Si stress values [MPa] = "
        f"{intrinsic_stresses[0]:.1f} to {intrinsic_stresses[-1]:.1f} "
        f"({len(intrinsic_stresses)} values)"
    )
    print(
        "Sweep points = "
        f"{len(temperatures) * len(intrinsic_stresses)} "
        f"({len(temperatures)} temperatures x {len(intrinsic_stresses)} stresses)"
    )
    print(
        f"Mechanical basis solves = {len(DBR_SECTIONS)} "
        "(one per DBR section, then linear strain scaling)"
    )
    print(f"Stress mesh = {args.stress_nx} x {args.stress_ny}")
    print(f"Optical grid target = {args.optical_dx} um x {args.optical_dy} um")
    if args.save_point_pngs:
        print(
            f"Per-point PNGs = enabled, every {args.point_png_every} point(s), "
            f"dpi = {args.point_png_dpi}, "
            f"max plotted pixels = {args.point_png_max_pixels}, "
            f"output root = {SWEEP_POINT_PNG_ROOT}"
        )
    else:
        print("Per-point PNGs = disabled")
    print(
        "Mechanical solve mode = "
        f"{'subprocess per section' if args.mechanical_subprocess else 'main process'}"
    )
    print(
        "Mechanical linear solver = "
        f"{args.mechanical_linear_solver}"
        + (
            ""
            if args.mechanical_linear_solver == "direct"
            else (
                f" ({args.mechanical_iterative_method}, "
                f"eps_r={args.mechanical_iterative_eps_r:g}, "
                f"i_max={args.mechanical_iterative_i_max})"
            )
        )
    )
    if args.resume:
        print(f"Resume mode: loaded {len(existing_rows)} existing rows.")
    print(f"Design DBR center wavelength = {DESIGN_WAVELENGTH_NM:.1f} nm")
    print(
        "DBR duty cycle: "
        f"{DBR_DUTY_CYCLE_UNPERTURBED:.2f} unperturbed / "
        f"{DBR_DUTY_CYCLE_PERTURBED:.2f} perturbed"
    )

    rows = run_temperature_intrinsic_stress_sweep(
        temperatures,
        intrinsic_stresses,
        args,
        existing_rows=existing_rows,
    )

    write_sweep_csv(rows)
    plot_sweep(rows)
    print(f"Saved sweep CSV: {SWEEP_CSV_FILE}")
    print(f"Saved zero-intrinsic-stress delta neff plot: {DELTA_NEFF_PNG_FILE}")
    print(f"Saved zero-intrinsic-stress DBR shift plot: {DBR_SHIFT_PNG_FILE}")
    print(f"Saved delta neff heatmap: {DELTA_NEFF_HEATMAP_PNG_FILE}")
    print(f"Saved DBR wavelength-shift heatmap: {DBR_SHIFT_HEATMAP_PNG_FILE}")


if __name__ == "__main__":
    main()
