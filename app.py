"""
app.py for the Axelrod (1997) culture model.

Color assignment:
- Each unique culture vector is assigned one color from matplotlib's tab20
- colormap via a dictionary lookup. Same culture → same color always.
- Stable regions visually identifiable as solid monochromatic patches

"""

from __future__ import annotations
import numpy as np

import solara
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mesa.visualization import SolaraViz, Slider
from mesa.visualization.utils import update_counter

from model import CultureModel


# colors are consistent across renders within one run. 
# It is cleared at the start of each CultureGrid render to reset when the model resets.
_color_registry: dict[tuple, tuple] = {}
_tab20 = plt.cm.get_cmap("tab20")


def culture_to_color(culture: np.ndarray) -> tuple:
    """
    Assign a deterministic color to a culture vector.

    The culture tuple is used directly as a dictionary key (tuples are hashable in Python). The first time a culture is seen, it gets the next available color from the tab20 colormap (20 distinct hues).
    Subsequent lookups return the same color instantly.

    Tab20 has 20 colors. If more than 20 distinct cultures exist (early in the run), colors cycle, but by the time the model is stable, there are always far fewer than 20 regions.
    """
    key = tuple(int(x) for x in culture)
    if key not in _color_registry:
        idx = len(_color_registry) % 20
        _color_registry[key] = _tab20(idx / 20)[:3]   # RGB triple in [0,1]
    return _color_registry[key]


# Grid component: imshow heatmap

# Builds a (H, W, 3) NumPy RGB array 

@solara.component
def CultureGrid(model):
    """Render the cultural landscape as a solid colour block grid."""
    update_counter.get()   # re-render on each model step

    # Reset color registry on each render so Reset produces fresh colors
    _color_registry.clear()

    W, H = model.width, model.height

    # Build RGB image: shape (H, W, 3), values in [0, 1]
    # img[y, x] = color of agent at grid coordinate (x, y)
    img = np.zeros((H, W, 3), dtype=float)
    for cell in model.grid.all_cells.cells:
        x, y = cell.coordinate
        img[y, x] = culture_to_color(cell.agents[0].culture)

    # Figure size scales with grid so cells stay square
    fig = Figure(figsize=(W * 0.55, H * 0.55))
    ax = fig.add_subplot(111)

    # interpolation="nearest" -> hard pixel edges, no blurring between cells
    # origin="lower"          -> (0,0) at bottom-left
    ax.imshow(img, origin="lower", interpolation="nearest",
              extent=[-0.5, W - 0.5, -0.5, H - 0.5])

    # Thin white grid lines so individual cells are distinguishable
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(np.arange(0, W, max(1, W // 5)))
    ax.set_yticks(np.arange(0, H, max(1, H // 5)))

    # Show current region/zone count in the title
    df = model.datacollector.get_model_vars_dataframe()
    n_reg = int(df.iloc[-1]["Regions"]) if len(df) > 0 else "?"
    n_zon = int(df.iloc[-1]["Zones"])   if len(df) > 0 else "?"
    stable = "  [STABLE]" if not model.running else ""
    ax.set_title(f"Cultural Map — regions={n_reg}  zones={n_zon}{stable}",
                 fontsize=9)

    fig.tight_layout(pad=0.3)
    solara.FigureMatplotlib(fig)

# Regions / Zones plot: dual y-axis
#
# Regions and Zones are plotted on separate y-axes because their scales differ substantially (e.g. Regions ~100, Zones ~25 early in the run).

@solara.component
def RegionsZonesPlot(model):
    update_counter.get()

    df = model.datacollector.get_model_vars_dataframe()

    fig = Figure(figsize=(5, 2.8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(df.index, df["Regions"], color="#1f77b4",
             label="Regions", linewidth=1.5)
    ax2.plot(df.index, df["Zones"], color="#ff7f0e",
             label="Zones", linewidth=1.5, linestyle="--")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Regions", color="#1f77b4")
    ax2.set_ylabel("Zones",   color="#ff7f0e")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax1.set_title("Cultural Regions and Zones over Time")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=8)

    fig.tight_layout()
    solara.FigureMatplotlib(fig)

# Similarity plot

@solara.component
def SimilarityPlot(model):
    update_counter.get()

    df = model.datacollector.get_model_vars_dataframe()

    fig = Figure(figsize=(5, 2.2))
    ax = fig.add_subplot(111)
    ax.plot(df.index, df["MeanSimilarity"], color="#2ca02c", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Similarity")
    ax.set_title("Average Neighbor Similarity")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)

# Model parameters

model_params = {
    "seed": Slider(
        label="Random seed", value=42, min=0, max=9999, step=1
    ),
    "width": Slider(
        label="Grid width", value=10, min=5, max=20, step=1
    ),
    "height": Slider(
        label="Grid height", value=10, min=5, max=20, step=1
    ),
    "n_features": Slider(
        label="Cultural features (F)", value=5, min=2, max=15, step=1
    ),
    # q=5  -> 1-2 regions, fast (large visible blobs)
    # q=10 -> ~3 regions, Axelrod case
    # q=15 -> ~20 regions, slow fragmented mosaic
    "n_traits": Slider(
        label="Traits per feature (q)", value=5, min=2, max=15, step=1
    ),
    "neighborhood": {
        "type":   "Select",
        "value":  "von_neumann",
        "values": ["von_neumann", "moore"],
        "label":  "Neighborhood",
    },
}


# SolaraViz

initial_model = CultureModel(
    width=10, height=10,
    n_features=5, n_traits=5,
    neighborhood="von_neumann",
    seed=42,
)

page = SolaraViz(
    initial_model,
    components=[CultureGrid, RegionsZonesPlot, SimilarityPlot],
    model_params=model_params,
    name="Axelrod (1997): Dissemination of Culture",
)
page 