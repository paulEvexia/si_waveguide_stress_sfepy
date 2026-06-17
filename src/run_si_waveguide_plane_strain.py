import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from sfepy.base.base import IndexedStruct
from sfepy.base.conf import ProblemConf
from sfepy.discrete import Problem
from sfepy.discrete.fem import Mesh
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
RESULTS = ROOT / "results"
RSLT = ROOT / "rslt"
RESULTS.mkdir(exist_ok=True)
RSLT.mkdir(exist_ok=True)

MESH_FILE = SRC / "si_waveguide_sio2_mesh.vtk"
PROBE_CSV_FILE = RESULTS / "probe_result.csv"
DISTRIBUTION_WIDE_CSV_FILE = RESULTS / "distribution_results_for_jmp.csv"
DISTRIBUTION_LONG_CSV_FILE = RESULTS / "distribution_results_long_for_jmp.csv"
STRESS_STRAIN_2D_CSV_FILE = RESULTS / "stress_strain_2d_distribution_for_jmp.csv"
STRESS_STRAIN_2D_LONG_CSV_FILE = RESULTS / "stress_strain_2d_distribution_long_for_jmp.csv"
MESH_STUDY_CSV_FILE = RESULTS / "mesh_convergence_probe_points.csv"
MESH_STUDY_SUMMARY_CSV_FILE = RESULTS / "mesh_convergence_summary.csv"
PROBE_CONVERGENCE_CSV_FILE = RESULTS / "probe_convergence_study.csv"
MATERIAL_MAP_PNG_FILE = RSLT / "material_map.png"

RGB_DISTRIBUTION_CMAP = LinearSegmentedColormap.from_list(
    "rgb_distribution",
    ["blue", "green", "red"],
)
RGB_MATERIAL_CMAP = ListedColormap(["blue", "red"])
STRAIN_PNG_ABS_LIMIT = 3.0e-3

# Geometry in micrometers.
# Stack, bottom to top: Si handle, SiO2 BOX, Si rib/slab, conformal SiO2 cladding.
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
Ly = y_max - y_min

enable_sio2_insert = True
sio2_insert_width = 2.2
sio2_insert_depth = 0.4
sio2_insert_center_x = 0.0
sio2_insert_top_y = si_top_y
sio2_insert_bottom_y = sio2_insert_top_y - sio2_insert_depth
sio2_insert_x_min = sio2_insert_center_x - sio2_insert_width / 2.0
sio2_insert_x_max = sio2_insert_center_x + sio2_insert_width / 2.0

probe_x = 0.0
probe_y = 0.5 * (si_slab_top_y + si_top_y)

nx = 400
ny = 280

# Isotropic approximations in MPa and 1/K.
E_sio2 = 72_000.0
nu_sio2 = 0.17
alpha_sio2 = 0.5e-6

E_si = 130_000.0
nu_si = 0.28
alpha_si = 2.6e-6

T_ref_C = 400.0
T_use_C = 25.0

# Positive intrinsic stress maps to positive isotropic eigenstrain in this starter model.
sigma_intrinsic_si_MPa = 0.0

# The single-case stress script defaults to the direct sparse solver. Long
# parameter sweeps can override this to use an iterative solver with lower
# peak memory.
linear_solver_kind = "ls.scipy_direct"
linear_solver_options = {}

JMP_BASE_COLUMNS = [
    "case_id",
    "T_ref_C",
    "T_use_C",
    "si_width_um",
    "si_height_um",
    "si_rib_width_um",
    "si_total_thickness_um",
    "si_slab_thickness_um",
    "si_handle_thickness_um",
    "box_thickness_um",
    "top_cladding_thickness_um",
    "sio2_insert_enabled",
    "sio2_insert_width_um",
    "sio2_insert_depth_um",
    "sio2_insert_center_x_um",
    "sio2_insert_top_y_um",
    "sio2_insert_bottom_y_um",
    "sigma_intrinsic_si_MPa",
    "eps_cte",
    "eps_intrinsic",
    "eps_star_total",
    "mesh_nx",
    "mesh_ny",
    "mesh_dx_um",
    "mesh_dy_um",
    "cell_id",
    "cell_i",
    "cell_j",
    "x_um",
    "y_um",
    "material_id",
    "material_name",
    "region_name",
]

JMP_STRESS_STRAIN_COLUMNS = [
    "sigma_xx_MPa",
    "sigma_yy_MPa",
    "sigma_xy_MPa",
    "eps_xx",
    "eps_yy",
    "eps_xy",
    "eps_eigen_xx",
    "eps_eigen_yy",
    "eps_eigen_xy",
    "eps_elastic_xx",
    "eps_elastic_yy",
    "eps_elastic_xy",
]

JMP_FIELD_COLUMNS = JMP_STRESS_STRAIN_COLUMNS + [
    "ux_um",
    "uy_um",
]

PNG_DISTRIBUTION_SPECS = [
    ("sigma_xx_MPa", "sigma_xx [MPa]", "stress_sigma_xx_MPa.png"),
    ("sigma_yy_MPa", "sigma_yy [MPa]", "stress_sigma_yy_MPa.png"),
    ("sigma_xy_MPa", "sigma_xy [MPa]", "stress_sigma_xy_MPa.png"),
    ("eps_xx", "total strain eps_xx", "strain_total_eps_xx.png"),
    ("eps_yy", "total strain eps_yy", "strain_total_eps_yy.png"),
    ("eps_xy", "total strain eps_xy", "strain_total_eps_xy.png"),
    ("eps_elastic_xx", "elastic strain eps_xx", "strain_elastic_eps_xx.png"),
    ("eps_elastic_yy", "elastic strain eps_yy", "strain_elastic_eps_yy.png"),
    ("eps_elastic_xy", "elastic strain eps_xy", "strain_elastic_eps_xy.png"),
    ("eps_eigen_xx", "eigenstrain eps_xx", "strain_eigen_eps_xx.png"),
    ("eps_eigen_yy", "eigenstrain eps_yy", "strain_eigen_eps_yy.png"),
    ("eps_eigen_xy", "eigenstrain eps_xy", "strain_eigen_eps_xy.png"),
]

enable_mesh_convergence_study = False
mesh_study_target_h_values_um = [0.15, 0.10, 0.075, 0.05]
mesh_study_sample_radius_um = 0.15
mesh_study_convergence_tolerance = 0.05
mesh_study_metrics = [
    "sigma_xx_MPa",
    "sigma_yy_MPa",
    "sigma_xy_MPa",
    "eps_xx",
    "eps_yy",
    "eps_xy",
]
mesh_study_user_probe_points = [
    {
        "point_name": "user_probe_waveguide_center",
        "x_um": 0.0,
        "y_um": 4.5,
    },
]


def build_mesh(nx_cells=nx, ny_cells=ny, mesh_file=MESH_FILE):
    xs = np.linspace(x_min, x_max, nx_cells + 1)
    ys = np.linspace(y_min, y_max, ny_cells + 1)
    coors = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)

    conn = []
    mat_ids = []

    for j in range(ny_cells):
        for i in range(nx_cells):
            n0 = j * (nx_cells + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx_cells + 1) + 1
            n3 = n0 + (nx_cells + 1)
            conn.append([n0, n1, n2, n3])
            mat_ids.append(0)

    conn = np.array(conn, dtype=np.int32)
    mat_ids = np.array(mat_ids, dtype=np.int32)

    mesh = Mesh.from_data(
        "si_waveguide_sio2_mesh",
        coors,
        None,
        [conn],
        [mat_ids],
        ["2_4"],
    )

    mesh.write(str(mesh_file))
    return xs, ys, coors, conn


D_sio2 = stiffness_from_youngpoisson(2, E_sio2, nu_sio2, plane="strain")
D_si = stiffness_from_youngpoisson(2, E_si, nu_si, plane="strain")
K_si = E_si / (3.0 * (1.0 - 2.0 * nu_si))


def inside_sio2_insert(x, y):
    if not enable_sio2_insert:
        return False

    return (
        sio2_insert_x_min <= x <= sio2_insert_x_max
        and sio2_insert_bottom_y <= y <= sio2_insert_top_y
    )


def inside_si_region(x, y):
    if inside_sio2_insert(x, y):
        return False

    inside_handle = si_handle_bottom_y <= y <= si_handle_top_y
    inside_slab = si_bottom_y <= y <= si_slab_top_y
    inside_rib = (
        np.abs(x) <= si_rib_width / 2.0
        and si_slab_top_y < y <= si_top_y
    )

    return inside_handle or inside_slab or inside_rib


def get_region_name(x, y):
    if si_handle_bottom_y <= y <= si_handle_top_y:
        return "Si handle"

    if box_bottom_y < y < box_top_y:
        return "BOX SiO2"

    if si_bottom_y <= y <= si_slab_top_y:
        return "Si slab"

    if inside_sio2_insert(x, y):
        return "SiO2 insert"

    if (
        np.abs(x) <= si_rib_width / 2.0
        and si_slab_top_y < y <= si_top_y
    ):
        return "Si rib"

    return "SiO2 cladding"


def make_material_function(eps_star_si):
    eps_star_vec = np.array([eps_star_si, eps_star_si, 0.0])
    sigma_star_vec = D_si @ eps_star_vec
    sigma_star_tensor = np.array(
        [
            [sigma_star_vec[0], sigma_star_vec[2]],
            [sigma_star_vec[2], sigma_star_vec[1]],
        ],
        dtype=np.float64,
    )

    def get_material(ts, coors, mode=None, **kwargs):
        if mode != "qp":
            return None

        x = coors[:, 0]
        y = coors[:, 1]
        inside_handle = (y >= si_handle_bottom_y) & (y <= si_handle_top_y)
        inside_slab = (y >= si_bottom_y) & (y <= si_slab_top_y)
        inside_rib = (
            (np.abs(x) <= si_rib_width / 2.0)
            & (y > si_slab_top_y)
            & (y <= si_top_y)
        )
        inside = inside_handle | inside_slab | inside_rib

        if enable_sio2_insert:
            inside_insert = (
                (x >= sio2_insert_x_min)
                & (x <= sio2_insert_x_max)
                & (y >= sio2_insert_bottom_y)
                & (y <= sio2_insert_top_y)
            )
            inside = inside & ~inside_insert

        nqp = coors.shape[0]
        D = np.zeros((nqp, 3, 3), dtype=np.float64)
        prestress = np.zeros((nqp, 2, 2), dtype=np.float64)

        D[~inside, :, :] = D_sio2
        D[inside, :, :] = D_si
        prestress[inside, :, :] = sigma_star_tensor

        return {
            "D": D,
            "prestress": prestress,
        }

    return get_material


def solve_displacement_case(nx_cells=nx, ny_cells=ny, mesh_file=MESH_FILE):
    xs, ys, coors, conn = build_mesh(nx_cells, ny_cells, mesh_file)

    dT = T_use_C - T_ref_C
    eps_cte = (alpha_si - alpha_sio2) * dT
    eps_intrinsic = sigma_intrinsic_si_MPa / (3.0 * K_si)
    eps_star_si = eps_cte + eps_intrinsic
    eps_star_vec = np.array([eps_star_si, eps_star_si, 0.0])

    get_material = make_material_function(eps_star_si)

    conf_dict = {
        "filename_mesh": str(mesh_file),
        "options": {
            "nls": "newton",
            "ls": "ls",
        },
        "functions": {
            "get_material": (get_material,),
        },
        "fields": {
            "displacement": ("real", "vector", "Omega", 1),
        },
        "variables": {
            "u": ("unknown field", "displacement", 0),
            "v": ("test field", "displacement", "u"),
        },
        "regions": {
            "Omega": "all",
            "Bottom": (
                f"vertices in (y < {y_min + 1e-9})",
                "vertex",
            ),
            "BottomCenter": (
                f"vertices in (x > {-1e-9}) & "
                f"(x < {1e-9}) & "
                f"(y < {y_min + 1e-9})",
                "vertex",
            ),
        },
        "materials": {
            "m": "get_material",
        },
        "ebcs": {
            "fix_bottom_y": ("Bottom", {"u.1": 0.0}),
            "fix_bottom_center_x": ("BottomCenter", {"u.0": 0.0}),
        },
        "integrals": {
            "i": 2,
        },
        "equations": {
            "balance": """
                dw_lin_elastic.i.Omega(m.D, v, u)
                =
                dw_lin_prestress.i.Omega(m.prestress, v)
            """,
        },
        "solvers": {
            "ls": (linear_solver_kind, linear_solver_options),
            "newton": (
                "nls.newton",
                {
                    "i_max": 1,
                    "eps_a": 1e-10,
                },
            ),
        },
    }

    conf = ProblemConf.from_dict(conf_dict, sys.modules[__name__])
    problem = Problem.from_conf(conf)

    status = IndexedStruct()
    state = problem.solve(status=status)
    u = state.get_state_parts()["u"].reshape((-1, 2))

    return {
        "xs": xs,
        "ys": ys,
        "coors": coors,
        "conn": conn,
        "u": u,
        "eps_cte": eps_cte,
        "eps_intrinsic": eps_intrinsic,
        "eps_star_si": eps_star_si,
        "eps_star_vec": eps_star_vec,
        "nx_cells": nx_cells,
        "ny_cells": ny_cells,
    }


def solve_case(nx_cells=nx, ny_cells=ny, mesh_file=MESH_FILE):
    solved = solve_displacement_case(nx_cells, ny_cells, mesh_file)

    return collect_results(
        solved["xs"],
        solved["ys"],
        solved["coors"],
        solved["conn"],
        solved["u"],
        solved["eps_cte"],
        solved["eps_intrinsic"],
        solved["eps_star_si"],
        solved["eps_star_vec"],
        solved["nx_cells"],
        solved["ny_cells"],
    )


def solve_case_strain_grid(
    nx_cells=nx,
    ny_cells=ny,
    strain_source="elastic",
    mesh_file=MESH_FILE,
):
    solved = solve_displacement_case(nx_cells, ny_cells, mesh_file)

    return collect_strain_grid(
        solved["xs"],
        solved["ys"],
        solved["coors"],
        solved["conn"],
        solved["u"],
        solved["eps_star_vec"],
        solved["nx_cells"],
        solved["ny_cells"],
        strain_source,
    )


def collect_strain_grid(
    xs,
    ys,
    coors,
    conn,
    u,
    eps_star_vec,
    nx_cells,
    ny_cells,
    strain_source,
):
    if strain_source not in {"elastic", "total"}:
        raise ValueError("strain_source must be 'elastic' or 'total'")

    x_values = 0.5 * (xs[:-1] + xs[1:])
    y_values = 0.5 * (ys[:-1] + ys[1:])
    eps_xx = np.zeros((ny_cells, nx_cells), dtype=float)
    eps_yy = np.zeros((ny_cells, nx_cells), dtype=float)
    eps_xy = np.zeros((ny_cells, nx_cells), dtype=float)

    for j in range(ny_cells):
        for i in range(nx_cells):
            e = j * nx_cells + i
            nodes = conn[e]
            xy = coors[nodes]
            ue = u[nodes]

            dx = xs[i + 1] - xs[i]
            dy = ys[j + 1] - ys[j]

            dNdx = np.array([-0.5 / dx, 0.5 / dx, 0.5 / dx, -0.5 / dx])
            dNdy = np.array([-0.5 / dy, -0.5 / dy, 0.5 / dy, 0.5 / dy])

            dux_dx = np.dot(dNdx, ue[:, 0])
            dux_dy = np.dot(dNdy, ue[:, 0])
            duy_dx = np.dot(dNdx, ue[:, 1])
            duy_dy = np.dot(dNdy, ue[:, 1])

            strain_vec = np.array([dux_dx, duy_dy, dux_dy + duy_dx])
            xc = xy[:, 0].mean()
            yc = xy[:, 1].mean()

            if strain_source == "elastic" and inside_si_region(xc, yc):
                strain_vec = strain_vec - eps_star_vec

            eps_xx[j, i] = strain_vec[0]
            eps_yy[j, i] = strain_vec[1]
            eps_xy[j, i] = 0.5 * strain_vec[2]

    return {
        "x": x_values,
        "y": y_values,
        "eps_xx": eps_xx,
        "eps_yy": eps_yy,
        "eps_xy": eps_xy,
    }


def collect_results(
    xs,
    ys,
    coors,
    conn,
    u,
    eps_cte,
    eps_intrinsic,
    eps_star_si,
    eps_star_vec,
    nx_cells,
    ny_cells,
):
    rows = []

    for j in range(ny_cells):
        for i in range(nx_cells):
            e = j * nx_cells + i
            nodes = conn[e]
            xy = coors[nodes]
            ue = u[nodes]

            dx = xs[i + 1] - xs[i]
            dy = ys[j + 1] - ys[j]

            dNdx = np.array([-0.5 / dx, 0.5 / dx, 0.5 / dx, -0.5 / dx])
            dNdy = np.array([-0.5 / dy, -0.5 / dy, 0.5 / dy, 0.5 / dy])

            dux_dx = np.dot(dNdx, ue[:, 0])
            dux_dy = np.dot(dNdy, ue[:, 0])
            duy_dx = np.dot(dNdx, ue[:, 1])
            duy_dy = np.dot(dNdy, ue[:, 1])

            strain_vec = np.array([dux_dx, duy_dy, dux_dy + duy_dx])
            xc = xy[:, 0].mean()
            yc = xy[:, 1].mean()

            if inside_si_region(xc, yc):
                material_id = 1
                material_name = "Si"
                D = D_si
                eigen = eps_star_vec
            else:
                material_id = 0
                material_name = "SiO2"
                D = D_sio2
                eigen = np.array([0.0, 0.0, 0.0])

            elastic_strain_vec = strain_vec - eigen
            stress_vec = D @ elastic_strain_vec

            rows.append(
                {
                    "case_id": 1,
                    "T_ref_C": T_ref_C,
                    "T_use_C": T_use_C,
                    "si_width_um": si_rib_width,
                    "si_height_um": si_total_thickness,
                    "si_rib_width_um": si_rib_width,
                    "si_total_thickness_um": si_total_thickness,
                    "si_slab_thickness_um": si_slab_thickness,
                    "si_handle_thickness_um": si_handle_thickness,
                    "box_thickness_um": box_thickness,
                    "top_cladding_thickness_um": top_cladding_thickness,
                    "sio2_insert_enabled": int(enable_sio2_insert),
                    "sio2_insert_width_um": sio2_insert_width,
                    "sio2_insert_depth_um": sio2_insert_depth,
                    "sio2_insert_center_x_um": sio2_insert_center_x,
                    "sio2_insert_top_y_um": sio2_insert_top_y,
                    "sio2_insert_bottom_y_um": sio2_insert_bottom_y,
                    "sigma_intrinsic_si_MPa": sigma_intrinsic_si_MPa,
                    "eps_cte": eps_cte,
                    "eps_intrinsic": eps_intrinsic,
                    "eps_star_total": eps_star_si,
                    "mesh_nx": nx_cells,
                    "mesh_ny": ny_cells,
                    "mesh_dx_um": Lx / nx_cells,
                    "mesh_dy_um": Ly / ny_cells,
                    "cell_id": e,
                    "cell_i": i,
                    "cell_j": j,
                    "x_um": xc,
                    "y_um": yc,
                    "material_id": material_id,
                    "material_name": material_name,
                    "region_name": get_region_name(xc, yc),
                    "sigma_xx_MPa": stress_vec[0],
                    "sigma_yy_MPa": stress_vec[1],
                    "sigma_xy_MPa": stress_vec[2],
                    "eps_xx": strain_vec[0],
                    "eps_yy": strain_vec[1],
                    "eps_xy": 0.5 * strain_vec[2],
                    "eps_eigen_xx": eigen[0],
                    "eps_eigen_yy": eigen[1],
                    "eps_eigen_xy": 0.5 * eigen[2],
                    "eps_elastic_xx": elastic_strain_vec[0],
                    "eps_elastic_yy": elastic_strain_vec[1],
                    "eps_elastic_xy": 0.5 * elastic_strain_vec[2],
                    "ux_um": ue[:, 0].mean(),
                    "uy_um": ue[:, 1].mean(),
                }
            )

    return rows


def write_wide_csv(rows):
    with open(DISTRIBUTION_WIDE_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_long_csv(rows):
    with open(DISTRIBUTION_LONG_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = JMP_BASE_COLUMNS + ["TYPE", "VALUE"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            base = {column: row[column] for column in JMP_BASE_COLUMNS}
            for metric in JMP_FIELD_COLUMNS:
                writer.writerow({**base, "TYPE": metric, "VALUE": row[metric]})


def write_stress_strain_2d_csv(rows):
    columns = JMP_BASE_COLUMNS + JMP_STRESS_STRAIN_COLUMNS

    with open(STRESS_STRAIN_2D_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def write_stress_strain_2d_long_csv(rows):
    fieldnames = JMP_BASE_COLUMNS + ["TYPE", "VALUE"]

    with open(STRESS_STRAIN_2D_LONG_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            base = {column: row[column] for column in JMP_BASE_COLUMNS}
            for metric in JMP_STRESS_STRAIN_COLUMNS:
                writer.writerow({**base, "TYPE": metric, "VALUE": row[metric]})


def write_probe_csv(rows):
    probe_row = min(
        rows,
        key=lambda row: (row["x_um"] - probe_x) ** 2 + (row["y_um"] - probe_y) ** 2,
    )

    with open(PROBE_CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value", "unit"])
        writer.writerow(["requested_x", probe_x, "um"])
        writer.writerow(["requested_y", probe_y, "um"])
        writer.writerow(["actual_x", probe_row["x_um"], "um"])
        writer.writerow(["actual_y", probe_row["y_um"], "um"])

        for key, value in probe_row.items():
            if key not in {"x_um", "y_um"}:
                writer.writerow([key, value, ""])


def get_grid_shape(rows):
    nx_cells = max(row["cell_i"] for row in rows) + 1
    ny_cells = max(row["cell_j"] for row in rows) + 1
    return ny_cells, nx_cells


def plot_material(rows):
    grid_shape = get_grid_shape(rows)
    material = np.array([row["material_id"] for row in rows], dtype=int).reshape(grid_shape)
    x_values = np.array([row["x_um"] for row in rows], dtype=float).reshape(grid_shape)
    y_values = np.array([row["y_um"] for row in rows], dtype=float).reshape(grid_shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        material,
        extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap=RGB_MATERIAL_CMAP,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["SiO2", "Si"])
    cbar.set_label("Material")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title("Si Waveguide Material Map")
    fig.tight_layout()
    fig.savefig(MATERIAL_MAP_PNG_FILE, dpi=300)
    plt.close(fig)


def plot_distribution_field(rows, metric, label, filename):
    grid_shape = get_grid_shape(rows)
    values = np.array([row[metric] for row in rows], dtype=float).reshape(grid_shape)
    x_values = np.array([row["x_um"] for row in rows], dtype=float).reshape(grid_shape)
    y_values = np.array([row["y_um"] for row in rows], dtype=float).reshape(grid_shape)

    color_scale = {}
    max_abs = np.nanmax(np.abs(values))

    if metric.startswith("eps_"):
        color_scale["vmin"] = -STRAIN_PNG_ABS_LIMIT
        color_scale["vmax"] = STRAIN_PNG_ABS_LIMIT
    elif max_abs > 0.0:
        color_scale["vmin"] = -max_abs
        color_scale["vmax"] = max_abs

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        values,
        extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()],
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap=RGB_DISTRIBUTION_CMAP,
        **color_scale,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_title(f"2D Distribution: {label}")
    fig.tight_layout()
    fig.savefig(RSLT / filename, dpi=300)
    plt.close(fig)


def plot_2d_distributions(rows):
    for metric, label, filename in PNG_DISTRIBUTION_SPECS:
        plot_distribution_field(rows, metric, label, filename)


def make_mesh_study_cases():
    cases = []

    for target_h_um in mesh_study_target_h_values_um:
        nx_cells = int(np.ceil(Lx / target_h_um))
        ny_cells = int(np.ceil(Ly / target_h_um))

        if nx_cells % 2:
            nx_cells += 1

        case_name = (
            f"h{target_h_um:.3f}_nx{nx_cells}_ny{ny_cells}"
            .replace(".", "p")
        )

        cases.append(
            {
                "mesh_case": case_name,
                "target_h_um": target_h_um,
                "nx_cells": nx_cells,
                "ny_cells": ny_cells,
                "actual_dx_um": Lx / nx_cells,
                "actual_dy_um": Ly / ny_cells,
            }
        )

    return cases


def get_vulnerable_points():
    return [
        dict(point) for point in mesh_study_user_probe_points
    ]


def get_corner_and_interface_points():
    points = [
        {
            "point_name": "handle_box_interface_center",
            "x_um": 0.0,
            "y_um": box_bottom_y,
        },
        {
            "point_name": "box_device_si_interface_center",
            "x_um": 0.0,
            "y_um": box_top_y,
        },
        {
            "point_name": "rib_left_slab_shoulder",
            "x_um": -si_rib_width / 2.0,
            "y_um": si_slab_top_y,
        },
        {
            "point_name": "rib_right_slab_shoulder",
            "x_um": si_rib_width / 2.0,
            "y_um": si_slab_top_y,
        },
        {
            "point_name": "rib_left_top_corner",
            "x_um": -si_rib_width / 2.0,
            "y_um": si_top_y,
        },
        {
            "point_name": "rib_right_top_corner",
            "x_um": si_rib_width / 2.0,
            "y_um": si_top_y,
        },
    ]

    if enable_sio2_insert:
        for x_name, x_um in [
            ("left", sio2_insert_x_min),
            ("right", sio2_insert_x_max),
        ]:
            for y_name, y_um in [
                ("bottom", sio2_insert_bottom_y),
                ("top", sio2_insert_top_y),
            ]:
                points.append(
                    {
                        "point_name": f"sio2_insert_{x_name}_{y_name}_corner",
                        "x_um": x_um,
                        "y_um": y_um,
                    }
                )

    return points


def summarize_vulnerable_points(rows, mesh_case):
    study_rows = []

    x_values = np.array([row["x_um"] for row in rows], dtype=float)
    y_values = np.array([row["y_um"] for row in rows], dtype=float)
    nx_cells = rows[0]["mesh_nx"]
    ny_cells = rows[0]["mesh_ny"]
    dx_um = rows[0]["mesh_dx_um"]
    dy_um = rows[0]["mesh_dy_um"]

    for point in get_vulnerable_points():
        distance = np.sqrt(
            (x_values - point["x_um"]) ** 2
            + (y_values - point["y_um"]) ** 2
        )
        sample_mask = distance <= mesh_study_sample_radius_um
        nearest_index = int(np.argmin(distance))

        if not np.any(sample_mask):
            sample_indices = np.array([nearest_index], dtype=int)
        else:
            sample_indices = np.where(sample_mask)[0]

        nearest_row = rows[nearest_index]

        for metric in mesh_study_metrics:
            metric_values = np.array(
                [rows[index][metric] for index in sample_indices],
                dtype=float,
            )
            abs_values = np.abs(metric_values)
            max_abs_index = int(np.argmax(abs_values))
            max_abs_source_row = rows[int(sample_indices[max_abs_index])]

            study_rows.append(
                {
                    "mesh_case": mesh_case["mesh_case"],
                    "target_h_um": mesh_case["target_h_um"],
                    "mesh_nx": nx_cells,
                    "mesh_ny": ny_cells,
                    "mesh_dx_um": dx_um,
                    "mesh_dy_um": dy_um,
                    "element_count": nx_cells * ny_cells,
                    "sample_radius_um": mesh_study_sample_radius_um,
                    "sampled_cell_count": int(len(sample_indices)),
                    "point_name": point["point_name"],
                    "point_x_um": point["x_um"],
                    "point_y_um": point["y_um"],
                    "nearest_cell_x_um": nearest_row["x_um"],
                    "nearest_cell_y_um": nearest_row["y_um"],
                    "nearest_region_name": nearest_row["region_name"],
                    "metric": metric,
                    "nearest_value": nearest_row[metric],
                    "mean_signed_value": float(np.mean(metric_values)),
                    "mean_abs_value": float(np.mean(abs_values)),
                    "p95_abs_value": float(np.percentile(abs_values, 95.0)),
                    "max_abs_value": float(abs_values[max_abs_index]),
                    "signed_value_at_max_abs": float(metric_values[max_abs_index]),
                    "max_abs_cell_x_um": max_abs_source_row["x_um"],
                    "max_abs_cell_y_um": max_abs_source_row["y_um"],
                    "max_abs_region_name": max_abs_source_row["region_name"],
                }
            )

    return study_rows


def write_csv_rows(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_mesh_convergence_summary(study_rows):
    case_order = []
    values_by_case = {}
    metadata_by_case = {}

    for row in study_rows:
        mesh_case = row["mesh_case"]
        key = (row["point_name"], row["metric"])

        if mesh_case not in values_by_case:
            values_by_case[mesh_case] = {}
            metadata_by_case[mesh_case] = {
                "mesh_case": mesh_case,
                "target_h_um": row["target_h_um"],
                "mesh_nx": row["mesh_nx"],
                "mesh_ny": row["mesh_ny"],
                "mesh_dx_um": row["mesh_dx_um"],
                "mesh_dy_um": row["mesh_dy_um"],
                "element_count": row["element_count"],
            }
            case_order.append(mesh_case)

        values_by_case[mesh_case][key] = row["p95_abs_value"]

    summary_rows = []
    recommended_case = None

    for index, mesh_case in enumerate(case_order):
        summary = dict(metadata_by_case[mesh_case])
        summary["convergence_metric"] = "max_relative_change_in_p95_abs_value"

        if index == 0:
            summary["previous_mesh_case"] = ""
            summary["max_relative_change_from_previous"] = ""
            summary["passes_tolerance"] = ""
        else:
            previous_case = case_order[index - 1]
            rel_changes = []

            for key, current_value in values_by_case[mesh_case].items():
                previous_value = values_by_case[previous_case].get(key)

                if previous_value is None:
                    continue

                denom = max(abs(current_value), abs(previous_value), 1e-30)
                rel_changes.append(abs(current_value - previous_value) / denom)

            max_rel_change = max(rel_changes) if rel_changes else np.nan
            passes_tolerance = bool(max_rel_change <= mesh_study_convergence_tolerance)

            summary["previous_mesh_case"] = previous_case
            summary["max_relative_change_from_previous"] = max_rel_change
            summary["passes_tolerance"] = int(passes_tolerance)

            if recommended_case is None and passes_tolerance:
                recommended_case = mesh_case

        summary["is_recommended_mesh"] = int(mesh_case == recommended_case)
        summary_rows.append(summary)

    if recommended_case is not None:
        for row in summary_rows:
            row["is_recommended_mesh"] = int(row["mesh_case"] == recommended_case)

    return summary_rows


def build_probe_convergence_rows(study_rows):
    rows_by_key = {}

    for row in study_rows:
        key = (row["mesh_case"], row["point_name"])

        if key not in rows_by_key:
            rows_by_key[key] = {
                "mesh_case": row["mesh_case"],
                "target_h_um": row["target_h_um"],
                "mesh_nx": row["mesh_nx"],
                "mesh_ny": row["mesh_ny"],
                "mesh_dx_um": row["mesh_dx_um"],
                "mesh_dy_um": row["mesh_dy_um"],
                "element_count": row["element_count"],
                "sample_radius_um": row["sample_radius_um"],
                "sampled_cell_count": row["sampled_cell_count"],
                "point_name": row["point_name"],
                "point_x_um": row["point_x_um"],
                "point_y_um": row["point_y_um"],
                "nearest_cell_x_um": row["nearest_cell_x_um"],
                "nearest_cell_y_um": row["nearest_cell_y_um"],
                "nearest_region_name": row["nearest_region_name"],
            }

        metric = row["metric"]
        rows_by_key[key][f"{metric}_nearest"] = row["nearest_value"]
        rows_by_key[key][f"{metric}_mean_signed"] = row["mean_signed_value"]
        rows_by_key[key][f"{metric}_p95_abs"] = row["p95_abs_value"]

    return list(rows_by_key.values())


def run_mesh_convergence_study():
    print("Si waveguide probe-point mesh convergence study")
    print("-----------------------------------------------")
    print(f"Simulation width = {Lx:.3f} um")
    print(f"Target h values [um] = {mesh_study_target_h_values_um}")
    print(f"Sample radius = {mesh_study_sample_radius_um:.3f} um")
    print(f"Convergence tolerance = {mesh_study_convergence_tolerance:.3f}")
    print("User probe points:")
    for point in mesh_study_user_probe_points:
        print(
            f"  {point['point_name']}: "
            f"x = {point['x_um']:.3f} um, y = {point['y_um']:.3f} um"
        )

    all_study_rows = []

    for case in make_mesh_study_cases():
        print(
            f"Running {case['mesh_case']}: "
            f"nx = {case['nx_cells']}, ny = {case['ny_cells']}, "
            f"dx = {case['actual_dx_um']:.5f} um, "
            f"dy = {case['actual_dy_um']:.5f} um"
        )

        rows = solve_case(
            nx_cells=case["nx_cells"],
            ny_cells=case["ny_cells"],
            mesh_file=MESH_FILE,
        )
        all_study_rows.extend(summarize_vulnerable_points(rows, case))

    summary_rows = build_mesh_convergence_summary(all_study_rows)
    probe_convergence_rows = build_probe_convergence_rows(all_study_rows)
    write_csv_rows(MESH_STUDY_CSV_FILE, all_study_rows)
    write_csv_rows(MESH_STUDY_SUMMARY_CSV_FILE, summary_rows)
    write_csv_rows(PROBE_CONVERGENCE_CSV_FILE, probe_convergence_rows)

    print(f"Saved mesh convergence study CSV: {MESH_STUDY_CSV_FILE}")
    print(f"Saved mesh convergence summary CSV: {MESH_STUDY_SUMMARY_CSV_FILE}")
    print(f"Saved compact probe convergence CSV: {PROBE_CONVERGENCE_CSV_FILE}")

    recommended = [
        row for row in summary_rows
        if row.get("is_recommended_mesh") == 1
    ]

    if recommended:
        row = recommended[0]
        print(
            "Recommended mesh from tolerance criterion: "
            f"{row['mesh_case']} "
            f"(dx = {row['mesh_dx_um']:.5f} um, "
            f"dy = {row['mesh_dy_um']:.5f} um)"
        )
    else:
        print("No mesh met the convergence tolerance; add finer target h values.")


def main():
    if enable_mesh_convergence_study or "--mesh-study" in sys.argv:
        run_mesh_convergence_study()
        return

    print("Si waveguide plane-strain stress simulation")
    print("------------------------------------------")
    print(f"Simulation width = {Lx:.3f} um")
    print(f"Mesh = {nx} x {ny} cells")
    print(f"Mesh dx = {Lx / nx:.5f} um")
    print(f"Mesh dy = {Ly / ny:.5f} um")
    print(f"T_ref = {T_ref_C:.1f} C")
    print(f"T_use = {T_use_C:.1f} C")
    print(f"Si total thickness = {si_total_thickness:.3f} um")
    print(f"Si slab thickness  = {si_slab_thickness:.3f} um")
    print(f"Si handle thickness = {si_handle_thickness:.3f} um")
    print(f"BOX thickness      = {box_thickness:.3f} um")
    print(f"Top SiO2 cladding  = {top_cladding_thickness:.3f} um above Si")
    print(f"Si rib width       = {si_rib_width:.3f} um")
    print(f"SiO2 rib insert enabled = {enable_sio2_insert}")
    print(
        f"SiO2 rib insert = {sio2_insert_width:.3f} um wide x "
        f"{sio2_insert_depth:.3f} um deep, "
        f"x = {sio2_insert_center_x:.3f} um, "
        f"y = {sio2_insert_bottom_y:.3f} to {sio2_insert_top_y:.3f} um"
    )
    print(f"Intrinsic Si stress = {sigma_intrinsic_si_MPa:.1f} MPa")

    rows = solve_case()
    write_probe_csv(rows)
    write_wide_csv(rows)
    write_long_csv(rows)
    write_stress_strain_2d_csv(rows)
    write_stress_strain_2d_long_csv(rows)
    plot_material(rows)
    plot_2d_distributions(rows)

    print(f"Saved probe CSV: {PROBE_CSV_FILE}")
    print(f"Saved wide CSV: {DISTRIBUTION_WIDE_CSV_FILE}")
    print(f"Saved long CSV: {DISTRIBUTION_LONG_CSV_FILE}")
    print(f"Saved 2D stress/strain CSV for JMP: {STRESS_STRAIN_2D_CSV_FILE}")
    print(f"Saved long 2D stress/strain CSV for JMP: {STRESS_STRAIN_2D_LONG_CSV_FILE}")
    print(f"Saved material map: {MATERIAL_MAP_PNG_FILE}")
    print(f"Saved {len(PNG_DISTRIBUTION_SPECS)} 2D distribution PNGs under: {RSLT}")


if __name__ == "__main__":
    main()
