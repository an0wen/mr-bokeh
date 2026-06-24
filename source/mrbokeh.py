import os
import numpy as np
import pandas as pd

from bokeh.models import (
    ColumnDataSource,
    ColorBar,
    HoverTool,
    LogColorMapper,
    Slider,
    CheckboxGroup,
    CustomJS,
    CustomJSFilter,
    CDSView,
    Select, MultiSelect, BooleanFilter,
    LabelSet,
    Legend, LegendItem,
    RangeSlider, CustomJSTickFormatter, DateSlider, DateRangeSlider, TextInput,
)
from bokeh.palettes import Cividis

from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, save

from datetime import datetime
from zoneinfo import ZoneInfo

from source.moustache import moustache

# Time of compilation
ts = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")

# ------------------------------------------------
# Load composition curves
# ------------------------------------------------
COMP_DIR = "data/mass_radius_composition/"
lf14 = pd.read_csv(
    COMP_DIR + "master_table_LF14_20201014.csv", comment="#", sep=r"\s+",
)
lf14_base_mask = lf14["metallicity_solar"] == 1.0
lf14_base_mask &= lf14["age_Gyr"] == 10.0
lf14_base_mask &= lf14["F_inc_oplus"] == 10.0

water_rock = np.loadtxt(
    COMP_DIR + "half_water/" + "massradius_50percentH2O_700K_1mbar.txt"
)
earth_like = np.loadtxt(COMP_DIR + "massradiusEarthlikeRocky.txt")
fe         = np.loadtxt(COMP_DIR + "massradiusFe.txt")

# Build lf14 dict: one array per envelope fraction
LF14_ENV_FRACS = [0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
lf14_models = {}
for f_env_pc in LF14_ENV_FRACS:
    mask = lf14_base_mask & (lf14["f_env_pc"] == f_env_pc)
    lf14_models[f_env_pc] = lf14.loc[mask, ["Mass_oplus", "R_oplus"]].to_numpy()

comp_lines = {
    "earth_like": earth_like,
    "pure_iron":  fe,
    "water_rock": water_rock,
    "lf14":       lf14_models,   # dict keyed by envelope fraction
}

# ------------------------------------------------
# Load and prepare the planet data
# ------------------------------------------------
DATA_DIR = os.path.dirname(__file__)
FNAME = os.path.join(DATA_DIR, "../data/nea_cleaned_filled_withTSM_2025-08-20.csv")
data = pd.read_csv(FNAME, comment="#")

data["pl_bmasseerr"] = np.mean(np.abs(data[["pl_bmasseerr1", "pl_bmasseerr2"]]), axis=1)
data["pl_radeerr"]   = np.mean(np.abs(data[["pl_radeerr1",   "pl_radeerr2"]]),   axis=1)

data["pl_bmasse_precision"] = data["pl_bmasse"] / data["pl_bmasseerr"]
data["pl_rade_precision"]   = data["pl_rade"]   / data["pl_radeerr"]

vecflt = np.vectorize(float)
data["mass_uplim_flag"] = vecflt(np.isnan(data["pl_bmasse_precision"]))

data["color"]               = data["pl_eqt"].copy()
data["ttv_flag_str"]        = data["ttv_flag"].apply(lambda f: "yes" if f else "no")
data["mass_uplim_flag_str"] = data["mass_uplim_flag"].apply(lambda f: "yes" if f else "no")
data["marker_radius"]       = 12

# JWST targets with no mass get precision = 0 so they aren't filtered out
to_replace = np.logical_and(data["jwst_obs"] == 1, np.isnan(data["pl_bmasse_precision"]))
data.loc[to_replace, "pl_bmasse_precision"] = 0.0

# Targets with atmospheric features
FNAME2 = os.path.join(DATA_DIR, "../data/targets_with_features.dat")
targets_with_features = np.genfromtxt(FNAME2, dtype="str", delimiter=",")
data["jwst_features"] = data["pl_name"].isin(targets_with_features).astype(float)

data_jwst_mask     = data["jwst_obs"] == 1
data_features_mask = data["jwst_features"] == 1

source_full = ColumnDataSource(data=data)
source      = ColumnDataSource(data=data)

# ------------------------------------------------
# Sliders and selectors
# ------------------------------------------------
slider_teff = RangeSlider(
    start=2500, end=10000, step=100, value=(2500, 10000), title="Teff [K]"
)
slider_period = RangeSlider(
    start=np.log10(0.1), end=np.log10(10000), step=0.25,
    value=(np.log10(0.1), np.log10(10000)),
    format=CustomJSTickFormatter(code="return Math.pow(10, tick).toFixed(2)"),
    title="Orbital period [d]",
)
slider_met      = RangeSlider(start=-0.5, end=0.5,  step=0.05, value=(-0.5, 0.5),  title="[Fe/H]")
slider_eqt      = RangeSlider(start=100,  end=4100,  step=50,   value=(100, 4100),   title="Planet Equilibrium Temperature (K)")
slider_tsm      = Slider(start=0,   end=100, step=5,   value=0,  title="TSM Minimum")
slider_massprec = Slider(start=0,   end=5,   step=1,   value=0,  title="Mass Precision Minimum (sigma)")
slider_jmag     = Slider(
    start=3, end=8, step=0.1, value=3,
    title="Bright Limit (Jmag). NIRISS ~3.5. NIRCam LW ~4.7. NIRSpec G395H ~6.5 (G/K) ~7.8 (M)",
)
slider_ttv = Slider(start=0, end=1, step=1, value=1, title="Show TTV Planets? (0: No, 1: Yes)")

filters_def = [
    ("st_teff",             "range",    slider_teff),
    ("pl_orbper",           "rangelog", slider_period),
    ("st_met",              "range",    slider_met),
    ("pl_eqt",              "range",    slider_eqt),
    ("pl_tsm",              "scalar",   slider_tsm),
    ("pl_bmasse_precision", "scalar",   slider_massprec),
    ("sy_jmag",             "scalar",   slider_jmag),
    ("ttv_flag",            "ttv",      slider_ttv),
]
filters_cols    = [c for c, _, _ in filters_def]
filters_kinds   = [k for _, k, _ in filters_def]
filters_widgets = [w for _, _, w in filters_def]

# ------------------------------------------------
# Combined CustomJSFilter for all sliders
# ------------------------------------------------
filter_combo = CustomJSFilter(
    args=dict(src=source, widgets=filters_widgets, kinds=filters_kinds, cols=filters_cols),
    code="""
    const data = src.data;
    const n = data[cols[0]].length;
    const keep = [];
    const multiSets  = {};
    const scalarVals = {};

    for (let j = 0; j < widgets.length; j++) {
        const kind = kinds[j];
        if (kind === "multi")  multiSets[j]  = new Set(widgets[j].value.map(v => String(v)));
        if (kind === "scalar") scalarVals[j] = widgets[j].value;
        if (kind === "ttv")    scalarVals[j] = widgets[j].value;
    }

    rowLoop:
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < widgets.length; j++) {
            const kind = kinds[j];
            const v    = data[cols[j]][i];

            if (kind === "range") {
                const [L, H] = widgets[j].value;
                if (!(v >= L && v <= H)) continue rowLoop;
            } else if (kind === "rangelog") {
                const [logL, logH] = widgets[j].value;
                if (!(v >= 10**logL && v <= 10**logH)) continue rowLoop;
            } else if (kind === "scalar") {
                if (v < scalarVals[j]) continue rowLoop;
            } else if (kind === "ttv") {
                if (v > scalarVals[j]) continue rowLoop;
            } else if (kind === "multi") {
                const set = multiSets[j];
                if (set.size > 0 && !set.has(String(v))) continue rowLoop;
            }
        }
        keep.push(i);
    }
    return keep;
""")

# ------------------------------------------------
# Checkboxes for JWST targets / features
# ------------------------------------------------
def make_box_filter(src, box):
    return CustomJSFilter(
        args=dict(src=src, box=box),
        code="""
        const enabled = box.active.length > 0;
        const n = src.data[Object.keys(src.data)[0]].length;
        if (!enabled) return [];
        const idx = new Array(n);
        for (let i = 0; i < n; i++) idx[i] = i;
        return idx;
    """)

checkbox_labels        = ["Existing JWST Targets", "Targets With JWST Features"]
checkbox_default_value = [[0], [0]]

checkbox_list = [
    CheckboxGroup(labels=[checkbox_labels[i]], active=checkbox_default_value[i])
    for i in range(len(checkbox_labels))
]

filter_checkbox = [make_box_filter(source, box) for box in checkbox_list]

filter_jwst = BooleanFilter(booleans=list(data_jwst_mask))
filter_feat = BooleanFilter(booleans=list(data_features_mask))

view      = CDSView(filter=filter_combo)
view_jwst = CDSView(filter=filter_combo & filter_jwst & filter_checkbox[0])
view_feat = CDSView(filter=filter_combo & filter_feat & filter_checkbox[1])

# Trigger filter recompute when sliders change
poke = CustomJS(args=dict(f=filter_combo), code="f.change.emit();")

def event_name(w):
    if isinstance(w, (Slider, RangeSlider, DateSlider, DateRangeSlider)):
        return "value"
    if isinstance(w, (TextInput, Select, MultiSelect, CheckboxGroup)):
        return "value"
    return "value"

for w in filters_widgets:
    w.js_on_change(event_name(w), poke)

for box, filt in zip(checkbox_list, filter_checkbox):
    box.js_on_change("active", CustomJS(args=dict(f=filt), code="f.change.emit();"))

# ------------------------------------------------
# Build the figure
# ------------------------------------------------
FIG_TOOLS = (
    "crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,"
    "undo,redo,reset,tap,save,box_select,poly_select,lasso_select"
)
FIG_HEIGHT = 700
FIG_WIDTH  = int(1.95 * FIG_HEIGHT)

p = figure(
    width=FIG_WIDTH, height=FIG_HEIGHT,
    title=f"NEA {FNAME} ; Compiled: {ts}",
    x_axis_label="Mp [Me]", y_axis_label="Rp [Re]",
    x_axis_type="log", y_axis_type="log",
    tools=FIG_TOOLS,
)

# Colorbar
cmap = LogColorMapper(palette="Plasma256", low=300, high=1500)
bar  = ColorBar(color_mapper=cmap, location=(0, 0), title="Planet Equilibrium Temperature [K]")
p.add_layout(bar, "right")

# ------------------------------------------------
# Solar system overlay
# ------------------------------------------------
ss_planet_data = {
    "Mercury": (0.055, 0.383), "Venus":   (0.815,  0.949),
    "Earth":   (1.0,   1.0),   "Mars":    (0.107,  0.532),
    "Jupiter": (317.8, 11.21), "Saturn":  (95.2,   9.45),
    "Uranus":  (14.6,  4.01),  "Neptune": (17.2,   3.88),
}
ss_alchemy_symbols = {
    "Mercury": "☿", "Venus": "♀", "Earth": "⊕", "Mars": "♂",
    "Jupiter": "♃", "Saturn": "♄", "Uranus": "♅", "Neptune": "♆",
}
ss_planets = list(ss_planet_data.keys())
ss_source = ColumnDataSource(dict(
    mass   = [ss_planet_data[p_][0] for p_ in ss_planets],
    radius = [ss_planet_data[p_][1] for p_ in ss_planets],
    symbol = [ss_alchemy_symbols[p_] for p_ in ss_planets],
))
# p.scatter("mass", "radius", source=ss_source, 
#           marker="circle", size=1, alpha=0,
#           legend_label="Solar System"
#           )
p.text("mass", "radius", text="symbol",
       source=ss_source, 
          text_font_size="25pt", text_align="center", text_baseline="middle", text_color="black",
          legend_label="Solar System"
          )

p.scatter([0], [0], marker = 'circle_cross', legend_label="Solar System",
           alpha = 1.0 , line_color="black", fill_color="white"
          )
# p.add_layout(LabelSet(
#     x="mass", y="radius", text="symbol", source=ss_source,
#     text_font_size="20pt", text_align="center", text_baseline="middle", text_color="black",
# ))

# ------------------------------------------------
# Exoplanet scatter plots
# ------------------------------------------------
data_plot = p.scatter(
    "pl_bmasse", "pl_rade", source=source, view=view,
    fill_color={"field": "color", "transform": cmap},
    size="marker_radius", alpha=0.9,
    legend_label="Exoplanets",
)

data_plot_jwst = p.scatter(
    "pl_bmasse", "pl_rade", source=source, view=view_jwst,
    marker="hex", fill_color="gold", line_color="black", line_width=1.5,
    size=6, alpha=1.0,
    legend_label="Existing JWST Targets",
)
data_plot_features = p.scatter(
    "pl_bmasse", "pl_rade", source=source, view=view_feat,
    marker="star", fill_color="lime", line_color="mediumseagreen", line_width=1.5,
    size=25, alpha=0.8,
    legend_label="Targets With JWST Features",
)
#p.add_layout(p.legend[0], "right")
#print(p.legend[0])
#p.legend.click_policy="hide"
data_plot_features.level = "underlay"

# Hover tool for exoplanets
DATA_TOOLTIPS = [
    ("",         "@pl_name"),
    ("V mag",    "@sy_vmag{%.1f}"),
    ("Teff",     "@st_teff{%d} K"),
    ("N pl",     "@sy_pnum"),
    ("P",        "@pl_orbper{%.2f} d"),
    ("Mp",       "@pl_bmasse{%.2f} +/- @pl_bmasseerr{%.2f} Me"),
    ("Rp",       "@pl_rade{%.2f} +/- @pl_radeerr{%.2f} Re"),
    ("Teq",      "@pl_eqt{%.1f}"),
    ("TSM",      "@pl_tsm{%.1f}"),
    ("TTVs?",    "@ttv_flag_str"),
    ("Mass Upper Limit or No Mass?", "@mass_uplim_flag_str"),
    ("Last update", "@rowupdate"),
]
DATA_FORMATTERS = {
    "@sy_vmag":      "printf",
    "@pl_orbper":    "printf",
    "@pl_bmasse":    "printf",
    "@pl_bmasseerr": "printf",
    "@pl_rade":      "printf",
    "@pl_radeerr":   "printf",
    "@pl_eqt":       "printf",
    "@pl_tsm":       "printf",
    "@st_teff":      "printf",
}
p.add_tools(HoverTool(renderers=[data_plot], tooltips=DATA_TOOLTIPS, formatters=DATA_FORMATTERS))

# Move the auto-created exoplanet legend to the right panel
# Must happen before plot_models() adds its own legends


# ------------------------------------------------
# Composition model lines
# Returns two separate CheckboxGroup widgets:
#   - one for the static Zeng curves
#   - one for the LF14 envelope fraction family
# Each group has its own legend added to p.right
# ------------------------------------------------
def plot_models(p, comp_lines):
    lf14_palette   = Cividis[6]

    # --- Static Zeng composition curves ---
    static_models = [
        ("earth_like", "Earth-like Rocky",  "green"),
        ("pure_iron",  "Pure Iron",          "gray"),
        ("water_rock", "50% Water (700K)",   "blue"),
    ]

    static_renderers = []
    static_items     = []

    for key, label, color in static_models:
        arr = comp_lines[key]
        src = ColumnDataSource(data=dict(
            x=[r[0] for r in arr],
            y=[r[1] for r in arr],
        ))
        line = p.line("x", "y", source=src, 
                      line_width=4, color=color, alpha=0.8,
                      legend_label=label)
        p.add_tools(HoverTool(renderers=[line], tooltips=[("", label)]))
        # static_renderers.append(line)
        # static_items.append(LegendItem(label=label, renderers=[line]))

    static_checkbox = CheckboxGroup(
        labels=[m[1] for m in static_models],
        active=list(range(len(static_models))),
    )
    static_checkbox.js_on_change("active", CustomJS(
        args=dict(renderers=static_renderers),
        code="""
            for (let i = 0; i < renderers.length; i++) {
                renderers[i].visible = cb_obj.active.includes(i);
            }
        """,
    ))

    # --- LF14 envelope fraction family ---
    lf14_env_fracs = LF14_ENV_FRACS
    lf14_renderers = []
    lf14_items     = []

    for i, f_env_pc in enumerate(lf14_env_fracs):
        label = f"{f_env_pc}% H/He"
        arr   = comp_lines["lf14"][f_env_pc]
        src   = ColumnDataSource(data=dict(
            x=[r[0] for r in arr],
            y=[r[1] for r in arr],
        ))
        line = p.line(
            "x", "y", source=src,
            line_width=4, color=lf14_palette[i], alpha=0.8,
            line_dash="solid",legend_label=label
        )
        p.add_tools(HoverTool(renderers=[line], tooltips=[("", f"LF14: {label}")]))
        # lf14_renderers.append(line)
        # lf14_items.append(LegendItem(label=label, renderers=[line]))

    #total_items=
    p.add_layout(p.legend[0],"right")
    # legend_column = column(p.legend[0],static_items[0],lf14_items[0])
    # p.add_layout(legend_column,"right")
    # print(p.legend[0])
    # print(static_items[0])
    # print(lf14_items[0])

    # print(type(p.legend[0]))
    # print(type(static_items[0]))
    # print(type(lf14_items[0]))
    #print(p.legend)

    p.legend.click_policy="hide"
    #p.add_layout(Legend(items=, title="Composition Models",location=(-200,300)), "right")
    #
    #p.legend.click_policy="hide"

    # lf14_checkbox = CheckboxGroup(
    #     labels=[f"{f}% H/He" for f in lf14_env_fracs],
    #     active=list(range(len(lf14_env_fracs))),
    # )
    # lf14_checkbox.js_on_change("active", CustomJS(
    #     args=dict(renderers=lf14_renderers),
    #     code="""
    #         for (let i = 0; i < renderers.length; i++) {
    #             renderers[i].visible = cb_obj.active.includes(i);
    #         }
    #     """,
    # ))

    return 0,0 #static_checkbox, lf14_checkbox

static_checkbox, lf14_checkbox = plot_models(p, comp_lines)

# ------------------------------------------------
# Final layout
# ------------------------------------------------
col1 = column(slider_teff, slider_period, slider_met, slider_eqt, margin=(0, 24, 0, 30))
col2 = column(slider_tsm, slider_massprec, slider_jmag, slider_ttv)
# col3 = column(
#     *checkbox_list,         # JWST / features toggles
#     static_checkbox,        # Zeng composition curve toggles
#     lf14_checkbox,          # LF14 envelope fraction toggles
# )
all_widgets = row(col1, col2)

page_layout = column(p, all_widgets)

FNAME_OUTPUT = os.path.join(DATA_DIR, "../output/mrbokeh.html")
output_file(FNAME_OUTPUT, mode="inline")
save(page_layout)
print("HTML successfully generated and output to ",FNAME_OUTPUT)
print(moustache)