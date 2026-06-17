import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "results" / "te00_neff_and_dbr_shift_vs_reference_temperature_and_intrinsic_stress.csv"
SWEEP_PNG_ROOT = ROOT / "rslt" / "optical_modes" / "sweep_points"
ASSET_DIR = ROOT / "rslt" / "optical_modes" / "ppt_assets"
MANIFEST_FILE = ROOT / "results" / "sweep_ppt_manifest.json"


def file_token(value):
    text = f"{float(value):.1f}"
    return text.replace("-", "m").replace(".", "p")


def sweep_folder(t_ref_c, intrinsic_stress_mpa):
    return SWEEP_PNG_ROOT / (
        f"Tref_{file_token(t_ref_c)}C_sigma_{file_token(intrinsic_stress_mpa)}MPa"
    )


def read_rows():
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append(
                {
                    **row,
                    "T_ref_C": float(row["T_ref_C"]),
                    "sigma_intrinsic_si_MPa": float(row["sigma_intrinsic_si_MPa"]),
                    "dbr_delta_wavelength_pm": float(row["dbr_delta_wavelength_pm"]),
                }
            )

    rows.sort(key=lambda item: (-item["T_ref_C"], item["sigma_intrinsic_si_MPa"]))
    return rows


def nice_range(values):
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.1, 1.0)
        return vmin - pad, vmax + pad

    pad = 0.08 * (vmax - vmin)
    return vmin - pad, vmax + pad


def get_font(size, bold=False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def color_for_index(index):
    palette = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
        "#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600",
    ]
    return palette[index % len(palette)]


def draw_line_plot(series, x_label, y_label, title, legend_title, output_path):
    width, height = 1100, 900
    margin_left = 130
    margin_right = 260
    margin_top = 90
    margin_bottom = 120
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    all_x = [point[0] for item in series for point in item["points"]]
    all_y = [point[1] for item in series for point in item["points"]]
    x_min, x_max = nice_range(all_x)
    y_min, y_max = nice_range(all_y)

    def map_x(value):
        return plot_left + (value - x_min) / (x_max - x_min) * plot_w

    def map_y(value):
        return plot_bottom - (value - y_min) / (y_max - y_min) * plot_h

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_title = get_font(34, bold=True)
    font_label = get_font(24)
    font_tick = get_font(18)
    font_legend = get_font(18)
    font_legend_title = get_font(20, bold=True)

    draw.text((margin_left, 28), title, fill="#111827", font=font_title)

    for tick in range(6):
        x_value = x_min + tick * (x_max - x_min) / 5.0
        x = map_x(x_value)
        draw.line((x, plot_top, x, plot_bottom), fill="#e5e7eb", width=1)
        draw.text((x - 28, plot_bottom + 14), f"{x_value:g}", fill="#374151", font=font_tick)

        y_value = y_min + tick * (y_max - y_min) / 5.0
        y = map_y(y_value)
        draw.line((plot_left, y, plot_right, y), fill="#e5e7eb", width=1)
        draw.text((20, y - 10), f"{y_value:.0f}", fill="#374151", font=font_tick)

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#111827", width=2)

    for index, item in enumerate(series):
        color = color_for_index(index)
        points = [(map_x(x), map_y(y)) for x, y in item["points"]]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=color)

    draw.text((plot_left + plot_w / 2 - 150, height - 55), x_label, fill="#111827", font=font_label)
    draw.text((plot_left, plot_top - 34), y_label, fill="#111827", font=get_font(18))

    legend_x = plot_right + 42
    legend_y = plot_top
    draw.text((legend_x, legend_y), legend_title, fill="#111827", font=font_legend_title)
    legend_y += 36
    for index, item in enumerate(series):
        color = color_for_index(index)
        y = legend_y + index * 28
        if y > height - 36:
            break
        draw.line((legend_x, y + 10, legend_x + 32, y + 10), fill=color, width=4)
        draw.text((legend_x + 42, y), item["label"], fill="#374151", font=font_legend)

    image.save(output_path)


def plot_pm_vs_temperature(rows, output_path):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sigma_intrinsic_si_MPa"], []).append(row)

    series = []
    for sigma, group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: item["T_ref_C"])
        series.append(
            {
                "label": f"{sigma:g} MPa",
                "points": [
                    (item["T_ref_C"], item["dbr_delta_wavelength_pm"])
                    for item in group
                ],
            }
        )

    draw_line_plot(
        series,
        "Reference temperature [C]",
        "DBR center wavelength shift [pm]",
        "DBR Shift vs. Reference Temperature",
        "Intrinsic stress",
        output_path,
    )


def plot_pm_vs_stress(rows, output_path):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["T_ref_C"], []).append(row)

    series = []
    for t_ref, group in sorted(grouped.items(), reverse=True):
        group = sorted(group, key=lambda item: item["sigma_intrinsic_si_MPa"])
        series.append(
            {
                "label": f"{t_ref:g} C",
                "points": [
                    (item["sigma_intrinsic_si_MPa"], item["dbr_delta_wavelength_pm"])
                    for item in group
                ],
            }
        )

    draw_line_plot(
        series,
        "Intrinsic Si stress [MPa]",
        "DBR center wavelength shift [pm]",
        "DBR Shift vs. Intrinsic Stress",
        "Reference temp.",
        output_path,
    )


def build_manifest(rows, section):
    image_names = {
        "sigma_xx": "stress_sigma_xx_MPa.png",
        "sigma_yy": "stress_sigma_yy_MPa.png",
        "sigma_xy": "stress_sigma_xy_MPa.png",
        "eps_xx": "strain_elastic_eps_xx.png",
        "eps_yy": "strain_elastic_eps_yy.png",
        "eps_xy": "strain_elastic_eps_xy.png",
    }
    slides = []
    missing = []

    for index, row in enumerate(rows, start=1):
        folder = sweep_folder(row["T_ref_C"], row["sigma_intrinsic_si_MPa"]) / section
        images = {key: str((folder / name).resolve()) for key, name in image_names.items()}

        for key, image_path in images.items():
            if not Path(image_path).exists():
                missing.append(
                    {
                        "slide": index,
                        "T_ref_C": row["T_ref_C"],
                        "sigma_intrinsic_si_MPa": row["sigma_intrinsic_si_MPa"],
                        "image_key": key,
                        "path": image_path,
                    }
                )

        slides.append(
            {
                "slide_index": index,
                "section": section,
                "T_ref_C": row["T_ref_C"],
                "sigma_intrinsic_si_MPa": row["sigma_intrinsic_si_MPa"],
                "dbr_delta_wavelength_pm": row["dbr_delta_wavelength_pm"],
                "folder": str(folder.resolve()),
                "images": images,
            }
        )

    return slides, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["perturbed", "unperturbed"], default="perturbed")
    args = parser.parse_args()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    pm_vs_temp = ASSET_DIR / "dbr_shift_pm_vs_reference_temperature_by_stress.png"
    pm_vs_stress = ASSET_DIR / "dbr_shift_pm_vs_intrinsic_stress_by_temperature.png"

    plot_pm_vs_temperature(rows, pm_vs_temp)
    plot_pm_vs_stress(rows, pm_vs_stress)

    slides, missing = build_manifest(rows, args.section)
    manifest = {
        "csv": str(CSV_FILE.resolve()),
        "section": args.section,
        "slide_count": len(slides) + 1,
        "sweep_slide_count": len(slides),
        "summary_slide": {
            "pm_vs_temperature": str(pm_vs_temp.resolve()),
            "pm_vs_stress": str(pm_vs_stress.resolve()),
        },
        "slides": slides,
        "missing": missing,
    }

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared {len(slides)} sweep slides for section: {args.section}")
    print(f"Saved summary plots under: {ASSET_DIR}")
    print(f"Saved manifest: {MANIFEST_FILE}")
    if missing:
        print(f"Missing required PNGs: {len(missing)}")
        for item in missing[:10]:
            print(f"  slide {item['slide']}: {item['image_key']} -> {item['path']}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
