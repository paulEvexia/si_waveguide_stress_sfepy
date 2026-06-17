import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import eigsh


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RSLT = ROOT / "rslt" / "optical_modes"
RESULTS.mkdir(exist_ok=True)
RSLT.mkdir(parents=True, exist_ok=True)

OPTICAL_MODES_CSV_FILE = RESULTS / "optical_modes_summary.csv"
OPTICAL_FIELD_CSV_FILE = RESULTS / "optical_fundamental_mode_fields.csv"
OPTICAL_STRESS_COUPLED_CSV_FILE = RESULTS / "optical_stress_coupled_neff_summary.csv"
OPTICAL_STRESS_FIELD_CSV_FILE = RESULTS / "optical_stress_coupled_mode_fields.csv"
DEFAULT_STRESS_STRAIN_CSV_FILE = RESULTS / "stress_strain_2d_distribution_for_jmp.csv"

# Geometry in micrometers. Keep these synchronized with run_si_waveguide_plane_strain.py.
Lx = 20.0
x_min = -Lx / 2.0
x_max = Lx / 2.0

box_thickness = 1.0
si_handle_thickness = 5.0
si_total_thickness = 3.0
si_slab_thickness = 1.8
top_cladding_thickness = 5.0

si_rib_width = 2.6
si_handle_bottom_y = 0.0
si_handle_top_y = si_handle_bottom_y + si_handle_thickness
box_bottom_y = si_handle_top_y
box_top_y = box_bottom_y + box_thickness
si_bottom_y = box_top_y
si_slab_top_y = si_bottom_y + si_slab_thickness
si_top_y = si_bottom_y + si_total_thickness
y_min = si_handle_bottom_y
y_max = si_top_y + top_cladding_thickness

enable_sio2_insert = True
sio2_insert_width = 2.2
sio2_insert_depth = 0.4
sio2_insert_center_x = 0.0
sio2_insert_top_y = si_top_y
sio2_insert_bottom_y = sio2_insert_top_y - sio2_insert_depth
sio2_insert_x_min = sio2_insert_center_x - sio2_insert_width / 2.0
sio2_insert_x_max = sio2_insert_center_x + sio2_insert_width / 2.0

# Optical constants at 1550 nm. Adjust if you need material dispersion.
wavelength_um = 1.55
n_si = 3.476
n_sio2 = 1.444

# Approximate strain-optic coefficients. Replace with process/foundry-specific
# values before using the absolute delta_neff as a design signoff number.
si_p11 = -0.09
si_p12 = 0.017
sio2_p11 = 0.121
sio2_p12 = 0.270

# The full stack includes a high-index Si handle. For rib modes, default to the
# optical window from BOX bottom through top cladding so substrate modes do not
# appear before the waveguide modes.
default_optical_dx_um = 0.05
default_optical_dy_um = 0.05
default_num_modes = 8
default_include_handle = False
min_device_si_fraction = 0.10
max_handle_fraction = 0.50


def inside_sio2_insert(x, y):
    if not enable_sio2_insert:
        return False

    return (
        sio2_insert_x_min <= x <= sio2_insert_x_max
        and sio2_insert_bottom_y <= y <= sio2_insert_top_y
    )


def inside_handle(x, y):
    return si_handle_bottom_y <= y <= si_handle_top_y


def inside_slab(x, y):
    return si_bottom_y <= y <= si_slab_top_y


def inside_rib(x, y):
    return (
        abs(x) <= si_rib_width / 2.0
        and si_slab_top_y < y <= si_top_y
        and not inside_sio2_insert(x, y)
    )


def inside_device_si(x, y):
    return inside_slab(x, y) or inside_rib(x, y)


def region_name(x, y, include_handle=False):
    if include_handle and inside_handle(x, y):
        return "Si handle"

    if box_bottom_y <= y <= box_top_y:
        return "BOX SiO2"

    if inside_slab(x, y):
        return "Si slab"

    if inside_sio2_insert(x, y):
        return "SiO2 insert"

    if inside_rib(x, y):
        return "Si rib"

    return "SiO2 cladding"


def refractive_index_at(x, y, include_handle=False):
    if include_handle and inside_handle(x, y):
        return n_si

    if inside_device_si(x, y):
        return n_si

    return n_sio2


def copy_grid_with_n_map(grid, n_map):
    copied = dict(grid)
    copied["n_map"] = n_map
    copied["epsilon"] = n_map**2
    return copied


def build_grid(dx_um, dy_um, include_handle=False):
    optical_y_min = y_min if include_handle else box_bottom_y
    optical_y_max = y_max

    nx_cells = int(np.ceil((x_max - x_min) / dx_um))
    ny_cells = int(np.ceil((optical_y_max - optical_y_min) / dy_um))
    x = np.linspace(x_min, x_max, nx_cells + 1)
    y = np.linspace(optical_y_min, optical_y_max, ny_cells + 1)

    x_interior = x[1:-1]
    y_interior = y[1:-1]
    xx, yy = np.meshgrid(x_interior, y_interior)

    n_map = np.empty_like(xx, dtype=float)
    region_map = np.empty(xx.shape, dtype=object)
    handle_mask = np.zeros(xx.shape, dtype=bool)
    device_si_mask = np.zeros(xx.shape, dtype=bool)
    rib_mask = np.zeros(xx.shape, dtype=bool)
    slab_mask = np.zeros(xx.shape, dtype=bool)
    insert_mask = np.zeros(xx.shape, dtype=bool)
    box_mask = np.zeros(xx.shape, dtype=bool)
    cladding_mask = np.zeros(xx.shape, dtype=bool)

    for index, _ in np.ndenumerate(xx):
        xi = float(xx[index])
        yi = float(yy[index])
        name = region_name(xi, yi, include_handle=include_handle)
        region_map[index] = name
        n_map[index] = refractive_index_at(xi, yi, include_handle=include_handle)
        handle_mask[index] = name == "Si handle"
        device_si_mask[index] = name in {"Si slab", "Si rib"}
        rib_mask[index] = name == "Si rib"
        slab_mask[index] = name == "Si slab"
        insert_mask[index] = name == "SiO2 insert"
        box_mask[index] = name == "BOX SiO2"
        cladding_mask[index] = name == "SiO2 cladding"

    return {
        "x": x_interior,
        "y": y_interior,
        "xx": xx,
        "yy": yy,
        "dx_um": float(x[1] - x[0]),
        "dy_um": float(y[1] - y[0]),
        "n_map": n_map,
        "epsilon": n_map**2,
        "region_map": region_map,
        "masks": {
            "handle": handle_mask,
            "device_si": device_si_mask,
            "rib": rib_mask,
            "slab": slab_mask,
            "sio2_insert": insert_mask,
            "box": box_mask,
            "cladding": cladding_mask,
        },
        "include_handle": include_handle,
        "optical_y_min": optical_y_min,
        "optical_y_max": optical_y_max,
        "nx": len(x_interior),
        "ny": len(y_interior),
    }


def read_strain_grid(csv_file, strain_source):
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

    rows = []

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "x_um": float(row["x_um"]),
                    "y_um": float(row["y_um"]),
                    "eps_xx": float(row[exx_column]),
                    "eps_yy": float(row[eyy_column]),
                    "eps_xy": float(row[exy_column]),
                }
            )

    if not rows:
        raise ValueError(f"No rows found in {csv_file}")

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
        eps_xx[j, i] = row["eps_xx"]
        eps_yy[j, i] = row["eps_yy"]
        eps_xy[j, i] = row["eps_xy"]

    return {
        "x": np.array(x_values, dtype=float),
        "y": np.array(y_values, dtype=float),
        "eps_xx": eps_xx,
        "eps_yy": eps_yy,
        "eps_xy": eps_xy,
    }


def interpolate_strain_to_optical_grid(grid, strain_grid):
    points = np.column_stack([grid["yy"].ravel(), grid["xx"].ravel()])
    interpolated = {}

    for metric in ["eps_xx", "eps_yy", "eps_xy"]:
        interpolator = RegularGridInterpolator(
            (strain_grid["y"], strain_grid["x"]),
            strain_grid[metric],
            bounds_error=False,
            fill_value=0.0,
        )
        interpolated[metric] = interpolator(points).reshape(grid["xx"].shape)

    return interpolated


def get_photoelastic_maps(grid):
    p11 = np.full(grid["xx"].shape, sio2_p11, dtype=float)
    p12 = np.full(grid["xx"].shape, sio2_p12, dtype=float)
    si_mask = grid["masks"]["device_si"] | grid["masks"]["handle"]
    p11[si_mask] = si_p11
    p12[si_mask] = si_p12
    return p11, p12


def build_stress_coupled_grids(grid, stress_csv_file, strain_source, stress_scale):
    strain_grid = read_strain_grid(stress_csv_file, strain_source)
    return build_stress_coupled_grids_from_strain_grid(grid, strain_grid, stress_scale)


def build_stress_coupled_grids_from_strain_grid(grid, strain_grid, stress_scale):
    strain = interpolate_strain_to_optical_grid(grid, strain_grid)
    p11, p12 = get_photoelastic_maps(grid)
    n0 = grid["n_map"]

    # Dominant-field scalar approximation:
    # TE-like field is treated as mostly Ex, TM-like field as mostly Ey.
    delta_inv_n2_te = p11 * strain["eps_xx"] + p12 * strain["eps_yy"]
    delta_inv_n2_tm = p12 * strain["eps_xx"] + p11 * strain["eps_yy"]

    delta_n_te = -0.5 * n0**3 * delta_inv_n2_te * stress_scale
    delta_n_tm = -0.5 * n0**3 * delta_inv_n2_tm * stress_scale
    n_te = np.maximum(n0 + delta_n_te, 1.0)
    n_tm = np.maximum(n0 + delta_n_tm, 1.0)

    return {
        "TE": copy_grid_with_n_map(grid, n_te),
        "TM": copy_grid_with_n_map(grid, n_tm),
        "delta_n_te": delta_n_te,
        "delta_n_tm": delta_n_tm,
        "strain": strain,
    }


def build_laplacian(nx_points, ny_points, dx_um, dy_um):
    lx = diags(
        [np.ones(nx_points - 1), -2.0 * np.ones(nx_points), np.ones(nx_points - 1)],
        [-1, 0, 1],
        shape=(nx_points, nx_points),
        format="csr",
    ) / dx_um**2
    ly = diags(
        [np.ones(ny_points - 1), -2.0 * np.ones(ny_points), np.ones(ny_points - 1)],
        [-1, 0, 1],
        shape=(ny_points, ny_points),
        format="csr",
    ) / dy_um**2

    return (
        kron(identity(ny_points, format="csr"), lx, format="csr")
        + kron(ly, identity(nx_points, format="csr"), format="csr")
    )


def build_weighted_divergence(epsilon, dx_um, dy_um):
    ny_points, nx_points = epsilon.shape
    inv_eps = 1.0 / epsilon
    rows = []
    cols = []
    data = []

    def flat_index(j, i):
        return j * nx_points + i

    for j in range(ny_points):
        for i in range(nx_points):
            p = flat_index(j, i)
            diag = 0.0

            if i + 1 < nx_points:
                coef = 0.5 * (inv_eps[j, i] + inv_eps[j, i + 1]) / dx_um**2
                rows.append(p)
                cols.append(flat_index(j, i + 1))
                data.append(coef)
            else:
                coef = inv_eps[j, i] / dx_um**2
            diag -= coef

            if i - 1 >= 0:
                coef = 0.5 * (inv_eps[j, i] + inv_eps[j, i - 1]) / dx_um**2
                rows.append(p)
                cols.append(flat_index(j, i - 1))
                data.append(coef)
            else:
                coef = inv_eps[j, i] / dx_um**2
            diag -= coef

            if j + 1 < ny_points:
                coef = 0.5 * (inv_eps[j, i] + inv_eps[j + 1, i]) / dy_um**2
                rows.append(p)
                cols.append(flat_index(j + 1, i))
                data.append(coef)
            else:
                coef = inv_eps[j, i] / dy_um**2
            diag -= coef

            if j - 1 >= 0:
                coef = 0.5 * (inv_eps[j, i] + inv_eps[j - 1, i]) / dy_um**2
                rows.append(p)
                cols.append(flat_index(j - 1, i))
                data.append(coef)
            else:
                coef = inv_eps[j, i] / dy_um**2
            diag -= coef

            rows.append(p)
            cols.append(p)
            data.append(diag)

    from scipy.sparse import coo_matrix

    return coo_matrix(
        (data, (rows, cols)),
        shape=(nx_points * ny_points, nx_points * ny_points),
    ).tocsr()


def normalize_field(field):
    max_abs = np.max(np.abs(field))
    if max_abs == 0.0:
        return field
    return field / max_abs


def confinement_fraction(intensity, mask):
    total = np.sum(intensity)
    if total == 0.0:
        return 0.0
    return float(np.sum(intensity[mask]) / total)


def summarize_mode(
    polarization,
    mode_index,
    neff,
    field,
    grid,
    analysis_state="unstressed",
    strain_source="none",
    stress_scale=0.0,
):
    field = normalize_field(field)
    intensity = np.abs(field) ** 2
    masks = grid["masks"]

    return {
        "analysis_state": analysis_state,
        "polarization": polarization,
        "mode_index": mode_index,
        "neff": float(neff),
        "wavelength_um": wavelength_um,
        "strain_source": strain_source,
        "stress_scale": stress_scale,
        "grid_dx_um": grid["dx_um"],
        "grid_dy_um": grid["dy_um"],
        "grid_nx": grid["nx"],
        "grid_ny": grid["ny"],
        "include_handle": int(grid["include_handle"]),
        "device_si_fraction": confinement_fraction(intensity, masks["device_si"]),
        "rib_fraction": confinement_fraction(intensity, masks["rib"]),
        "slab_fraction": confinement_fraction(intensity, masks["slab"]),
        "handle_fraction": confinement_fraction(intensity, masks["handle"]),
        "box_fraction": confinement_fraction(intensity, masks["box"]),
        "cladding_fraction": confinement_fraction(intensity, masks["cladding"]),
        "sio2_insert_fraction": confinement_fraction(intensity, masks["sio2_insert"]),
        "selected_as_fundamental": 0,
    }


def solve_te_modes(grid, num_modes):
    k0 = 2.0 * np.pi / wavelength_um
    epsilon = grid["epsilon"]
    ny_points, nx_points = epsilon.shape
    laplacian = build_laplacian(nx_points, ny_points, grid["dx_um"], grid["dy_um"])
    operator = laplacian + diags((k0**2 * epsilon).ravel(), 0, format="csr")

    eigenvalues, eigenvectors = eigsh(
        operator,
        k=num_modes,
        which="LA",
        tol=1e-8,
        maxiter=5000,
    )

    order = np.argsort(eigenvalues)[::-1]
    modes = []

    for out_index, eig_index in enumerate(order):
        beta2 = float(eigenvalues[eig_index])
        neff = np.sqrt(max(beta2, 0.0)) / k0
        field = eigenvectors[:, eig_index].reshape((ny_points, nx_points))
        modes.append((out_index, neff, normalize_field(field)))

    return modes


def solve_tm_modes(grid, num_modes):
    k0 = 2.0 * np.pi / wavelength_um
    epsilon = grid["epsilon"]
    ny_points, nx_points = epsilon.shape
    divergence = build_weighted_divergence(epsilon, grid["dx_um"], grid["dy_um"])
    operator = divergence + identity(nx_points * ny_points, format="csr") * k0**2
    mass = diags((1.0 / epsilon).ravel(), 0, format="csr")

    eigenvalues, eigenvectors = eigsh(
        operator,
        M=mass,
        k=num_modes,
        which="LA",
        tol=1e-8,
        maxiter=5000,
    )

    order = np.argsort(eigenvalues)[::-1]
    modes = []

    for out_index, eig_index in enumerate(order):
        beta2 = float(eigenvalues[eig_index])
        neff = np.sqrt(max(beta2, 0.0)) / k0
        field = eigenvectors[:, eig_index].reshape((ny_points, nx_points))
        modes.append((out_index, neff, normalize_field(field)))

    return modes


def select_fundamental_mode(mode_rows):
    candidates = [
        row for row in mode_rows
        if row["device_si_fraction"] >= min_device_si_fraction
        and row["handle_fraction"] <= max_handle_fraction
    ]

    if not candidates:
        candidates = sorted(
            mode_rows,
            key=lambda row: (row["device_si_fraction"], row["neff"]),
            reverse=True,
        )
    else:
        candidates = sorted(
            candidates,
            key=lambda row: row["neff"],
            reverse=True,
        )

    return candidates[0]


def plot_index_map(grid, filename="index_map.png", title="Optical Index Map"):
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        grid["n_map"],
        extent=[
            grid["x"].min(),
            grid["x"].max(),
            grid["y"].min(),
            grid["y"].max(),
        ],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="viridis",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Refractive index")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RSLT / filename, dpi=300)
    plt.close(fig)


def plot_delta_n(grid, delta_n, polarization):
    max_abs = np.max(np.abs(delta_n))

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        delta_n,
        extent=[
            grid["x"].min(),
            grid["x"].max(),
            grid["y"].min(),
            grid["y"].max(),
        ],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-max_abs if max_abs > 0.0 else None,
        vmax=max_abs if max_abs > 0.0 else None,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"delta n, {polarization}-like")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(f"Stress/Strain-Induced Index Perturbation: {polarization}")
    fig.tight_layout()
    fig.savefig(RSLT / f"{polarization}_stress_coupled_delta_n.png", dpi=300)
    plt.close(fig)


def plot_mode(grid, field, polarization, analysis_state="unstressed"):
    intensity = np.abs(normalize_field(field)) ** 2
    filename_prefix = (
        polarization
        if analysis_state == "unstressed"
        else f"{polarization}_{analysis_state}"
    )
    title_prefix = (
        f"Fundamental {polarization}-like"
        if analysis_state == "unstressed"
        else f"Fundamental {polarization}-like, {analysis_state}"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        intensity,
        extent=[
            grid["x"].min(),
            grid["x"].max(),
            grid["y"].min(),
            grid["y"].max(),
        ],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="inferno",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized intensity")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(f"{title_prefix} Mode Intensity")
    fig.tight_layout()
    fig.savefig(RSLT / f"{filename_prefix}_fundamental_intensity.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    max_abs = np.max(np.abs(field))
    im = ax.imshow(
        field,
        extent=[
            grid["x"].min(),
            grid["x"].max(),
            grid["y"].min(),
            grid["y"].max(),
        ],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized field")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(f"{title_prefix} Mode Field")
    fig.tight_layout()
    fig.savefig(RSLT / f"{filename_prefix}_fundamental_field.png", dpi=300)
    plt.close(fig)


def write_mode_summary(rows):
    with open(OPTICAL_MODES_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_stress_coupled_summary(selected_rows):
    rows = []

    for polarization in ["TE", "TM"]:
        unstressed = selected_rows.get(("unstressed", polarization))
        stressed = selected_rows.get(("stress_coupled", polarization))

        if unstressed is None or stressed is None:
            continue

        rows.append(
            {
                "polarization": polarization,
                "wavelength_um": wavelength_um,
                "strain_source": stressed["strain_source"],
                "stress_scale": stressed["stress_scale"],
                "neff_unstressed": unstressed["neff"],
                "neff_stress_coupled": stressed["neff"],
                "delta_neff": stressed["neff"] - unstressed["neff"],
                "relative_delta_neff": (
                    (stressed["neff"] - unstressed["neff"]) / unstressed["neff"]
                    if unstressed["neff"] != 0.0
                    else 0.0
                ),
                "unstressed_mode_index": unstressed["mode_index"],
                "stress_coupled_mode_index": stressed["mode_index"],
                "unstressed_device_si_fraction": unstressed["device_si_fraction"],
                "stress_coupled_device_si_fraction": stressed["device_si_fraction"],
                "stress_coupled_rib_fraction": stressed["rib_fraction"],
                "stress_coupled_slab_fraction": stressed["slab_fraction"],
                "stress_coupled_sio2_insert_fraction": stressed["sio2_insert_fraction"],
            }
        )

    if rows:
        with open(OPTICAL_STRESS_COUPLED_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def solve_and_select(
    grid,
    polarization,
    solver,
    num_modes,
    analysis_state,
    strain_source="none",
    stress_scale=0.0,
):
    modes = solver(grid, num_modes)
    rows = []
    fields_by_index = {}

    for mode_index, neff, field in modes:
        row = summarize_mode(
            polarization,
            mode_index,
            neff,
            field,
            grid,
            analysis_state=analysis_state,
            strain_source=strain_source,
            stress_scale=stress_scale,
        )
        rows.append(row)
        fields_by_index[mode_index] = field

    selected = select_fundamental_mode(rows)
    selected["selected_as_fundamental"] = 1
    selected_field = fields_by_index[selected["mode_index"]]

    return rows, selected, selected_field


def write_fundamental_fields(
    grid,
    selected_modes,
    path=OPTICAL_FIELD_CSV_FILE,
    te_n_map=None,
    tm_n_map=None,
):
    te_field = selected_modes["TE"]
    tm_field = selected_modes["TM"]
    te_intensity = np.abs(te_field) ** 2
    tm_intensity = np.abs(tm_field) ** 2
    te_n_map = grid["n_map"] if te_n_map is None else te_n_map
    tm_n_map = grid["n_map"] if tm_n_map is None else tm_n_map

    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "x_um",
            "y_um",
            "n",
            "n_te",
            "n_tm",
            "region_name",
            "te_field",
            "te_intensity",
            "tm_field",
            "tm_intensity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for j in range(grid["ny"]):
            for i in range(grid["nx"]):
                writer.writerow(
                    {
                        "x_um": grid["xx"][j, i],
                        "y_um": grid["yy"][j, i],
                        "n": grid["n_map"][j, i],
                        "n_te": te_n_map[j, i],
                        "n_tm": tm_n_map[j, i],
                        "region_name": grid["region_map"][j, i],
                        "te_field": te_field[j, i],
                        "te_intensity": te_intensity[j, i],
                        "tm_field": tm_field[j, i],
                        "tm_intensity": tm_intensity[j, i],
                    }
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finite-difference optical mode solve for the Si rib geometry."
    )
    parser.add_argument("--dx", type=float, default=default_optical_dx_um)
    parser.add_argument("--dy", type=float, default=default_optical_dy_um)
    parser.add_argument("--modes", type=int, default=default_num_modes)
    parser.add_argument(
        "--include-handle",
        action="store_true",
        default=default_include_handle,
        help="Include the high-index Si handle in the optical window.",
    )
    parser.add_argument(
        "--stress-coupled",
        action="store_true",
        help="Re-solve modes with strain-optic index perturbation from the stress CSV.",
    )
    parser.add_argument(
        "--stress-csv",
        type=Path,
        default=DEFAULT_STRESS_STRAIN_CSV_FILE,
        help="Stress/strain CSV from run_si_waveguide_plane_strain.py.",
    )
    parser.add_argument(
        "--strain-source",
        choices=["elastic", "total"],
        default="elastic",
        help="Use elastic or total strain for photoelastic perturbation.",
    )
    parser.add_argument(
        "--stress-scale",
        type=float,
        default=1.0,
        help="Multiplier for the stress/strain-induced index perturbation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    grid = build_grid(args.dx, args.dy, include_handle=args.include_handle)

    print("Si rib optical mode solve")
    print("-------------------------")
    print(f"wavelength = {wavelength_um:.4f} um")
    print(f"n_Si = {n_si:.4f}")
    print(f"n_SiO2 = {n_sio2:.4f}")
    print(f"grid = {grid['nx']} x {grid['ny']} interior points")
    print(f"dx = {grid['dx_um']:.5f} um, dy = {grid['dy_um']:.5f} um")
    print(f"include Si handle = {args.include_handle}")
    print(f"requested modes per polarization = {args.modes}")
    print(f"stress-coupled solve = {args.stress_coupled}")

    plot_index_map(grid)

    all_rows = []
    selected_rows = {}
    unstressed_fields = {}

    for polarization, solver in [
        ("TE", solve_te_modes),
        ("TM", solve_tm_modes),
    ]:
        print(f"Solving {polarization}-like modes...")
        rows, selected, selected_field = solve_and_select(
            grid,
            polarization,
            solver,
            args.modes,
            analysis_state="unstressed",
        )
        selected_rows[("unstressed", polarization)] = selected
        unstressed_fields[polarization] = selected_field
        plot_mode(grid, selected_field, polarization)
        all_rows.extend(rows)
        print(
            f"Selected {polarization} mode {selected['mode_index']} "
            f"with neff = {selected['neff']:.6f}, "
            f"device Si fraction = {selected['device_si_fraction']:.4f}"
        )

    stress_coupled_fields = {}

    if args.stress_coupled:
        print("Building stress/strain-coupled optical index maps...")
        perturbation = build_stress_coupled_grids(
            grid,
            args.stress_csv,
            args.strain_source,
            args.stress_scale,
        )
        plot_delta_n(grid, perturbation["delta_n_te"], "TE")
        plot_delta_n(grid, perturbation["delta_n_tm"], "TM")
        plot_index_map(
            perturbation["TE"],
            filename="TE_stress_coupled_index_map.png",
            title="TE-Like Stress/Strain-Coupled Index Map",
        )
        plot_index_map(
            perturbation["TM"],
            filename="TM_stress_coupled_index_map.png",
            title="TM-Like Stress/Strain-Coupled Index Map",
        )

        for polarization, solver in [
            ("TE", solve_te_modes),
            ("TM", solve_tm_modes),
        ]:
            print(f"Solving stress-coupled {polarization}-like modes...")
            rows, selected, selected_field = solve_and_select(
                perturbation[polarization],
                polarization,
                solver,
                args.modes,
                analysis_state="stress_coupled",
                strain_source=args.strain_source,
                stress_scale=args.stress_scale,
            )
            selected_rows[("stress_coupled", polarization)] = selected
            stress_coupled_fields[polarization] = selected_field
            plot_mode(
                perturbation[polarization],
                selected_field,
                polarization,
                analysis_state="stress_coupled",
            )
            all_rows.extend(rows)
            print(
                f"Selected stress-coupled {polarization} mode "
                f"{selected['mode_index']} with neff = {selected['neff']:.6f}"
            )

        write_stress_coupled_summary(selected_rows)
        write_fundamental_fields(
            grid,
            stress_coupled_fields,
            path=OPTICAL_STRESS_FIELD_CSV_FILE,
            te_n_map=perturbation["TE"]["n_map"],
            tm_n_map=perturbation["TM"]["n_map"],
        )

    write_mode_summary(all_rows)
    write_fundamental_fields(grid, unstressed_fields)
    print(f"Saved optical mode summary: {OPTICAL_MODES_CSV_FILE}")
    print(f"Saved fundamental mode fields: {OPTICAL_FIELD_CSV_FILE}")
    if args.stress_coupled:
        print(f"Saved stress-coupled neff summary: {OPTICAL_STRESS_COUPLED_CSV_FILE}")
        print(f"Saved stress-coupled mode fields: {OPTICAL_STRESS_FIELD_CSV_FILE}")
    print(f"Saved optical PNGs under: {RSLT}")


if __name__ == "__main__":
    main()
