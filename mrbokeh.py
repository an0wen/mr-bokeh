import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors

from bokeh.models import (
    ColumnDataSource,
    LinearColorMapper,
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
    Toggle,
    Select, MultiSelect, BooleanFilter
)
from bokeh.palettes import Cividis
from bokeh.layouts import column, row
from bokeh.plotting import figure, output_file, save

from bokeh.layouts import column, layout, row
from bokeh.models import ColumnDataSource, RangeSlider, CustomJS, CustomJSTickFormatter, DateSlider, DateRangeSlider, TextInput
from bokeh.plotting import figure, output_file, show
from bokeh.io import curdoc, save
from bokeh.resources import INLINE

from datetime import datetime
from zoneinfo import ZoneInfo

# Time of compilation
ts = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S %Z")

# Load some composition curves
COMP_DIR = "data/mass_radius_composition/"
lf14 = pd.read_csv(
    COMP_DIR + "master_table_LF14_20201014.csv", comment="#", sep=r"\s+",
)
lf14_base_mask  = lf14["metallicity_solar"] == 1.0
lf14_base_mask &= lf14["age_Gyr"] == 10.0
lf14_base_mask &= lf14["F_inc_oplus"] == 10.0

water_rock = np.loadtxt(COMP_DIR + "half_water/" + "massradius_50percentH2O_700K_1mbar.txt")
# Zeng models
water_rock = np.loadtxt(
    COMP_DIR + "half_water/" + f"massradius_50percentH2O_700K_1mbar.txt"
)
earth_like = np.loadtxt(COMP_DIR + "massradiusEarthlikeRocky.txt")
fe = np.loadtxt(COMP_DIR + "massradiusFe.txt")

LF14_ENV_FRACS = [0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
lf14_models = {}
for f_env_pc in LF14_ENV_FRACS:
    mask = lf14_base_mask & (lf14["f_env_pc"] == f_env_pc)
    lf14_models[f_env_pc] = lf14.loc[mask, ["Mass_oplus", "R_oplus"]].to_numpy()

comp_lines = {
    "earth_like": earth_like,
    "pure_iron":  fe,
    "water_rock": water_rock,
    "lf14":       lf14_models,
}

def get_marker_radius(mp_precision):
    """
    Hacky way to make marker size proportional to TSM
    """
    #base_size = 5
    #if mp_precision < 40:
    #    return base_size
    #elif mp_precision >= 40 and mp_precision < 60:
    #    return base_size + 2
    #elif mp_precision >= 60 and mp_precision < 80:
    #    return base_size + 4
    #elif mp_precision >= 80 and mp_precision < 100:
    #    return base_size + 6
    #elif mp_precision >= 100:
    #    return base_size + 8
    return 12

# Function to plot composition curves
def plot_composition_curves(p, comp_lines):
    lf14 = comp_lines["lf14"]

    # Lopez and Fortney
    lf14_palette = Cividis[6]
    for i, f_env_pc in enumerate([0.01, 0.1, 1.0, 5.0, 10.0, 20.0]):
        mask = lf14_base_mask & (lf14["f_env_pc"] == f_env_pc)
        label = f"{f_env_pc}% H/He"
        comp_line = p.line(
            lf14.loc[mask, "Mass_oplus"],
            lf14.loc[mask, "R_oplus"],
            legend_label=label,
            line_width=4,
            color=lf14_palette[i],
            alpha=0.8,
        )
        p.add_tools(HoverTool(renderers=[comp_line], tooltips=[("", label)]))

    # Zeng curves
    label = "100% Fe"
    fe = comp_lines["fe"]
    comp_line = p.line(
        fe[:, 0], fe[:, 1], legend_label=label, line_width=4, color="red", alpha=0.8
    )
    p.add_tools(HoverTool(renderers=[comp_line], tooltips=[("", label)]))

    label = "Earth-like"
    earth_like = comp_lines["earth_like"]
    comp_line = p.line(
        earth_like[:, 0],
        earth_like[:, 1],
        legend_label=label,
        line_width=4,
        color="green",
        alpha=0.8,
    )
    p.add_tools(HoverTool(renderers=[comp_line], tooltips=[("", label)]))

    label = "50% Water + 50% Rock"
    water_rock = comp_lines["water_rock"]
    comp_line = p.line(
        water_rock[:, 0],
        water_rock[:, 1],
        legend_label=label,
        line_width=4,
        color="blue",
        alpha=0.8,
    )
    p.add_tools(HoverTool(renderers=[comp_line], tooltips=[("", label)]))

# ------------------------------------------------
# ------------------------------------------------
# Load and prepare the data
# ------------------------------------------------
# ------------------------------------------------
DATA_DIR = "./"  #### just put it here
FNAME = "nea_cleaned_filled_withTSM_2025-08-20.csv"
data = pd.read_csv(DATA_DIR + FNAME, comment="#")

# Some initial filtering
# mask = data.default_flag == 1
# mask &= ~np.isnan(data.pl_bmasse)
# data = data[mask].reset_index(drop=True)

# Add column for a symmetric Gaussian error bar that's the average of the lower and upper bars
data["pl_bmasseerr"] = np.mean(np.abs(data[["pl_bmasseerr1", "pl_bmasseerr2"]]), axis=1)
data["pl_radeerr"] = np.mean(np.abs(data[["pl_radeerr1", "pl_radeerr2"]]), axis=1)

data["pl_bmasse_precision"] = data["pl_bmasse"] / data["pl_bmasseerr"]

data["pl_rade_precision"] = data["pl_rade"] / data["pl_radeerr"]

vecint = np.vectorize(int)
vecflt = np.vectorize(float)
data["mass_uplim_flag"] = vecflt(np.isnan(data["pl_bmasse_precision"]))

data["ttv_flag_str"]        = data["ttv_flag"].apply(lambda f: "yes" if f else "no")
data["mass_uplim_flag_str"] = data["mass_uplim_flag"].apply(lambda f: "yes" if f else "no")
data["marker_radius"]       = 12
# Add some additional columns
data["color"] = data["pl_eqt"].copy()  # data["st_met"].copy()
data["ttv_flag_str"] = data["ttv_flag"].apply(lambda flag: "yes" if flag else "no")
data["mass_uplim_flag_str"] = data["mass_uplim_flag"].apply(lambda flag: "yes" if flag else "no")
data["marker_radius"] = data["pl_tsm"].apply(get_marker_radius)

#now that we have flagged planets with bad masses, get the ones with JWST obs
#to be 0.0 so they don't get filtered out
to_replace_bmasse_precision = np.logical_and(data['jwst_obs']==1, np.isnan(data['pl_bmasse_precision']))
for i in range(len(to_replace_bmasse_precision)):
    if to_replace_bmasse_precision[i]:
        data.at[i, "pl_bmasse_precision"] = 0.0

#add targets with features
targets_with_features = np.genfromtxt("targets_with_features.dat", dtype='str',
        delimiter=',')
has_jwst_features = np.zeros(len(data))
for i in range(len(data)):
    if data['pl_name'][i] in targets_with_features:
        has_jwst_features[i] = 1
data["jwst_features"] = has_jwst_features

data_jwst_mask = data["jwst_obs"]==1
# df_jwst = data[data["jwst_obs"]==1]

data_features_mask = data["jwst_features"]==1
# df_features = data[data["jwst_features"]==1]

# Create the CDS
source_full = ColumnDataSource(data=data)     # full dataset
source      = ColumnDataSource(data=data)      # filtered dataset (will be updated)
# source_jwst = ColumnDataSource(data=df_jwst)  # filtered jwst dataset
# source_feat = ColumnDataSource(data=df_features)  # filtered dataset with jwst features

# ------------------------------------------------
# ------------------------------------------------
# Make all the sliders and other selectors
# ------------------------------------------------
# ------------------------------------------------

# Create Teff slider
start = 2500
end = 10000
step = 100
slider_teff = RangeSlider(
    start=start, end=end, step=step, value=(start, end), title="Teff [K]"
)

# Create (log_10) orbital period widget
start = np.log10(0.1)
end = np.log10(10000)
step = 0.25
slider_period = RangeSlider(
    start=start,
    end=end,
    step=step,
    value=(start, end),
    format=CustomJSTickFormatter(code="return Math.pow(10, tick).toFixed(2)"),
    title="Orbital period [d]",
)

# Create stellar metallicity widget
start = -0.5
end = 0.5
step = 0.05
slider_met = RangeSlider(
    start=start,
    end=end,
    step=step,
    value=(start, end),
    title="[Fe/H]",
)

# Create teq widget
start = 100
end = 4100
step = 50
slider_eqt = RangeSlider(
    start=start,
    end=end,
    step=step,
    value=(start, end),
    title="Planet Equilibrium Temperature (K)",
)

# Create TSM slider
start = 0
end = 100
step = 5
slider_tsm = Slider(
    start=start,
    end=end,
    step=step,
    value=start,
    title="TSM Minimum",
)

# Create mass precision slider
start = 0
end = 5
step = 1
slider_massprec = Slider(
    start=start,
    end=end,
    step=step,
    value=start,
    title="Mass Precision Minimum (sigma)",
)

#create a jmag slider
start = 3
end = 8
step = 0.1
slider_jmag = Slider(
    start=start,
    end=end,
    step=step,
    value=start,
    title="Bright Limit (Jmag). new NIRISS ~3.5. NIRCam LW ~4.7. NIRSpec G395H ~6.5 (G/K) ~7.8 (M)  ",
)


# Create TTV flag toggle widget
slider_ttv = Slider(
    start=0, end=1, step=1, value=1, title="Show TTV Planets? (0: No, 1: Yes)"
)

# categorical examples (adjust options to your data)
# select_spectype = MultiSelect(title="Spectral type", options=sorted(set(map(str, data["st_spectype"]))), value=[])
# select_discmethod = Select(title="Discovery", options=["any"] + sorted(set(map(str, data["discoverymethod"]))), value="any")

# ----- Describe each filter once (column, kind, widget) -----
# kinds supported below: "range" (RangeSlider), "scalar" (Select/Slider), "multi" (MultiSelect)
filters_def = [
    ("st_teff",   "range",  slider_teff),
    ("pl_orbper",  "rangelog",  slider_period),
    ("st_met",    "range",  slider_met),
    ("pl_eqt",    "range",  slider_eqt),
    ("pl_tsm",    "scalar",  slider_tsm),
    ("pl_bmasse_precision",    "scalar",  slider_massprec),
    ("sy_jmag",    "scalar",  slider_jmag),
    ("ttv_flag", "ttv", slider_ttv),
    # ("st_spectype", "multi",  select_spectype),
    # ("discoverymethod", "scalar", select_discmethod) 
]

filters_cols   = [c for c, _, _ in filters_def]
filters_kinds  = [k for _, k, _ in filters_def]
filters_widgets = [w for _, _, w in filters_def]

# ------------------------------------------------
# ------------------------------------------------
# Make sliders interactive with JS (Hell)
# ------------------------------------------------
# ------------------------------------------------

# ----- One combined CustomJSFilter for all widgets -----
filter_combo = CustomJSFilter(args=dict(src=source, widgets=filters_widgets, kinds=filters_kinds, cols=filters_cols), code="""
    const data = src.data; // This is our data
    const n = data[cols[0]].length;  // assume all columns same length
    const keep = []; // This is our output which is passed to "source"

    // Precompute helper objects for multi/scalar
    const multiSets = {};
    const scalarVals = {};

    // The values of a "range" slider is a 2-tuple 
    // The values of a "scalar" slider is a number
    // This loop is just here to store scalar values 
    //    (and other non-traditional slider values)
    //    into the "scalarVals" array
    for (let j = 0; j < widgets.length; j++) {
        const kind = kinds[j];
        if (kind === "multi") {
            // Set of allowed values (strings); empty set => no restriction
            multiSets[j] = new Set(widgets[j].value.map(v => String(v)));
        } else if (kind === "scalar") {
            scalarVals[j] = widgets[j].value;
        } else if (kind === "ttv") {
            scalarVals[j] = widgets[j].value;
        }
    }

    // This structure scans over all rows (planets) and applies all filters on each row
    // First loop on all rows (planets)
    // Then loop over all filters
    // If any of the filters activate, "continue rowLoop" means 
    //    "do nothing and go to next row
    // If all filters are passed, then the end of the loop saves
    //    the planet in the array "keep" (output array)
    rowLoop:
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < widgets.length; j++) {
            const kind = kinds[j]; // Type of the widget, defined in the dictionary
            const col  = cols[j]; // Column name as found in the NEA
            const arr  = data[col]; // The property to verify
            const v    = arr[i]; // The value of the property to verify

            // Treat null/NaN as failing the filter; tweak if you prefer to keep them
            if (v == null || (typeof v === "number" && !isFinite(v))) {
                // continue rowLoop;
            }

            if (kind === "range") {
                // Range slider compares "v" to bounds "[L, H]"
                const [L, H] = widgets[j].value;
                if (!(v >= L && v <= H)) continue rowLoop;
            } else if (kind === "rangelog") {
                // Range-log slider converts bounds to 10 ** x
                const [logL, logH] = widgets[j].value;
                const L = 10 ** logL;           // or: Math.pow(10, logL)
                const H = 10 ** logH;           // or: Math.pow(10, logH)
                if (!(v >= L && v <= H)) continue rowLoop;
            } else if (kind === "scalar") {
                // Scalar slider programmed to keep only cases where value > slider limlit
                const wanted = scalarVals[j];
                if (v < wanted) continue rowLoop;
            } else if (kind === "ttv") {
                // Needed to make TTVs special because they work in the opposite way
                const wanted = scalarVals[j];
                if (v > wanted) continue rowLoop;
            } else if (kind === "multi") {
                // MultiSelect (string compare)
                // Just random stuff ChatGPT gave me
                const set = multiSets[j];
                if (set.size > 0 && !set.has(String(v))) continue rowLoop;

            } else {
                // Unknown kind -> ignore or treat as pass
            }
        }
        keep.push(i); // This is the line that appends row "i" to "keep"
    }
    return keep;   // indices that pass ALL filters (AND logic)
""")

# ------------------------------------------------
# ------------------------------------------------
# Make checkboxes (easier than sliders because less things to specify manually)
# Very simple on/off filters: if the checkbox is pressed, return all indices, if not return empty array
# ------------------------------------------------
# ------------------------------------------------

# ----- Manual part -----
# Specify the text (labels) to appear next to each checkbox
checkbox_labels = ["Existing JWST Targets", "Targets With JWST Features"]
# Default value: for each group, it must be a list containing the index of the box to be active by default. Since we have N groups of size 1, each element should be [0] (active) or [] (inactive)
checkbox_default_value = [[0], [0]]

# ----- Automatic part -----
# Number of boxes
num_boxes = len(checkbox_labels)

# Put all boxes in a list
checkbox_list = []
for idx_box in range(num_boxes):
    checkbox_list.append(CheckboxGroup(labels=[checkbox_labels[idx_box]], active=checkbox_default_value[idx_box]))

# Define a function that takes a box and returns the corresponding filter, all filters are placed in a list
def make_box_filter(src,box):
    one_filter = CustomJSFilter(
        args=dict(src=src, box=box),
        code="""
        const enabled = box.active.length > 0;            // [0] when checked, [] when not
        const data = src.data;
        const n = data[Object.keys(data)[0]].length;      // length of first column
        if (!enabled) return [];                          // return empty array if checkbox disabled
        const idx = new Array(n);
        for (let i = 0; i < n; i++) idx[i] = i;           // keep all rows
        return idx;
    """)
    return one_filter

# Make all filters at once
filter_checkbox = []
for box in checkbox_list:
    filter_checkbox.append(make_box_filter(source,box))

# ----- Boolean filter for JWST observed targets -----
filter_jwst = BooleanFilter(data_jwst_mask)

# ----- Boolean filter for targets with JWST features -----
filter_feat = BooleanFilter(data_features_mask)

# ----- Some level of manual work again -----
# Specify which filter applies to which view
# New 3.x API: single 'filter=' and no 'source=' on CDSView
view = CDSView(filter=filter_combo)
view_jwst = CDSView(filter=filter_combo & filter_jwst & filter_checkbox[0])
view_feat = CDSView(filter=filter_combo & filter_feat & filter_checkbox[1])


# IMPORTANT: explicitly trigger recompute when sliders change
poke = CustomJS(args=dict(f=filter_combo), code="f.change.emit();")
# poke_box = CustomJS(args=dict(f=filter_jwst_checkbox), code="f.change.emit();")

def event_name(w):
    if isinstance(w, (Slider, RangeSlider, DateSlider, DateRangeSlider)):
        return "value"   # fires when the user releases the handle
    if isinstance(w, TextInput):
        return "value_input"       # fires on each keystroke; use "value" to wait for blur
    if isinstance(w, (Select, MultiSelect)):
        return "value"
    if isinstance(w, (CheckboxGroup, RadioButtonGroup, Toggle)):
        return "active"
    return "value"                 # sensible default

for w in filters_widgets:
    w.js_on_change(event_name(w), poke)

# Also trigger recompute when checkboxes change
for box,filt in zip(checkbox_list,filter_checkbox):
    box.js_on_change("active", CustomJS(args=dict(f=filt), code="f.change.emit();"))

# ------------------------------------------------
# ------------------------------------------------
# Make the figure
# ------------------------------------------------
# ------------------------------------------------

# Tools for the whole figure
FIG_TOOLS = "crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset"
FIG_TOOLS += ",tap,save,box_select,poly_select,lasso_select,"
FIG_HEIGHT = 700
FIG_WIDTH = int(1.95 * FIG_HEIGHT)
p = figure(
    width=FIG_WIDTH,
    height=FIG_HEIGHT,
    title=f"NEA {FNAME} ; Compiled: {ts}",
    x_axis_label="Mp [Me]",
    y_axis_label="Rp [Re]",
    x_axis_type="log",
    y_axis_type="log",
    tools=FIG_TOOLS,
)

# Colorbar — created after p so it attaches correctly
cmap = LogColorMapper(palette="Plasma256", low=300, high=1500)
bar  = ColorBar(color_mapper=cmap, location=(0, 0), title="Planet Eq. Temperature [K]")
# Add the colorbar
cmap_low = 300  # -0.5
cmap_high = 1500  # 0.5
cmap = LogColorMapper(palette="Plasma256", low=cmap_low, high=cmap_high)
bar = ColorBar(
    color_mapper=cmap, location=(0, 0), title="Planet Equilibrium Temperature [K]"
)
p.add_layout(bar, "right")

# ------------------------------------------------
# Color-variable selector and gray-out toggle
# ------------------------------------------------
color_select = Select(
    title="Color points by:",
    value="pl_eqt",
    options=[
        ("pl_eqt",  "Equilibrium Temperature"),
        ("pl_tsm",  "TSM"),
        ("st_teff", "Stellar Teff"),
    ],
    width=220,
)

gray_toggle = Toggle(label="Gray out all points", active=False, width=180)

_palette_map = {k: v["bokeh_palette"] for k, v in COLOR_VARS.items()}
_low_map     = {k: v["vmin"]          for k, v in COLOR_VARS.items()}
_high_map    = {k: v["vmax"]          for k, v in COLOR_VARS.items()}
_title_map   = {k: v["label"]         for k, v in COLOR_VARS.items()}

color_callback = CustomJS(
    args=dict(
        source=source,
        mapper=cmap,
        bar=bar,
        select=color_select,
        toggle=gray_toggle,
        palette_map=_palette_map,
        low_map=_low_map,
        high_map=_high_map,
        title_map=_title_map,
    ),
    code="""
        const col    = select.value;
        const grayed = toggle.active;
        const data   = source.data;

        if (grayed) {
            data['color'] = data['colors_gray'].slice();
            bar.visible   = false;
        } else {
            data['color']  = data['colors_' + col].slice();
            bar.visible    = true;
            bar.title      = title_map[col];
            mapper.low     = low_map[col];
            mapper.high    = high_map[col];
            mapper.palette = palette_map[col];
            mapper.change.emit();
        }
        source.change.emit();
    """,
)

color_select.js_on_change("value", color_callback)
gray_toggle.js_on_change("active", color_callback)

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
ss_plot=p.text("mass", "radius", text="symbol",
       source=ss_source,
       text_font_size="25pt", text_align="center", text_baseline="middle", text_color="black",
       legend_label="Solar System"
       )
p.scatter([0], [0], marker="circle_cross", legend_label="Solar System",
          alpha=1.0, line_color="black", fill_color="white",
          )
ss_plot.level='overlay'
# ------------------------------------------------
# Exoplanet scatter plots
# ------------------------------------------------
# Plot the data
data_plot = p.scatter(
    "pl_bmasse", "pl_rade", source=source, view=view,
    fill_color="color",
    size="marker_radius", alpha=0.9,
    "pl_bmasse",
    "pl_rade",
    source=source,
    view=view,
    fill_color={"field": "color", "transform": cmap},
    size="marker_radius",
    alpha=0.9,
    legend_label="Exoplanets",
)
data_plot_jwst = p.scatter(
    "pl_bmasse",
    "pl_rade",
    source=source,
    view=view_jwst,
    marker='hex', fill_color='gold', 
    line_color='black', 
    line_width=1.5, 
    size=6,
    alpha=1.,
    legend_label="Existing JWST Targets",
)

data_plot_features = p.scatter(
    "pl_bmasse",
    "pl_rade",
    source=source,
    view=view_feat,
    marker='star', fill_color='lime', 
    line_color='mediumseagreen', 
    line_width=1.5, 
    size=25,
    alpha=0.8,
    legend_label="Targets With JWST Features",
)
data_plot_features.level = "underlay"
data_plot_features.level = 'underlay'

# Hover properties for planet data
DATA_TOOLTIPS = [
    ("", "@pl_name"),
    ("V mag", "@sy_vmag{%.1f}"),
    ("Teff", "@st_teff{%d} K"),
    ("N pl", "@sy_pnum"),
    ("P", "@pl_orbper{%.2f} d"),
    ("Mp", "@pl_bmasse{%.2f} +/- @pl_bmasseerr{%.2f} Me"),
    ("Rp", "@pl_rade{%.2f} +/- @pl_radeerr{%.2f} Re"),
    ("Teq", "@pl_eqt{%.1f}"),
    ("TSM", "@pl_tsm{%.1f}"),
    ("TTVs?", "@ttv_flag_str"),
    ("Mass Upper Limit or No Mass?", "@mass_uplim_flag_str"),
    ("Last update", "@rowupdate"),
]
DATA_FORMATTERS = {
    "@sy_vmag": "printf",
    "@pl_orbper": "printf",
    "@pl_bmasse": "printf",
    "@pl_bmasseerr": "printf",
    "@pl_rade": "printf",
    "@pl_radeerr": "printf",
    "@pl_eqt": "printf", 
    "@pl_tsm": "printf",
    "@st_teff": "printf",
}
p.add_tools(HoverTool(renderers=[data_plot], tooltips=DATA_TOOLTIPS, formatters=DATA_FORMATTERS))

# ------------------------------------------------
# Composition model lines
# ------------------------------------------------
def plot_models(p, comp_lines):
    lf14_palette = Cividis[6]

    static_models = [
        ("earth_like", "Earth-like Rocky", "green"),
        ("pure_iron",  "Pure Iron",         "gray"),
        ("water_rock", "50% Water (700K)",  "blue"),
    ]

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

    for i, f_env_pc in enumerate(LF14_ENV_FRACS):
        label = f"{f_env_pc}% H/He"
        arr   = comp_lines["lf14"][f_env_pc]
        src   = ColumnDataSource(data=dict(
            x=[r[0] for r in arr],
            y=[r[1] for r in arr],
        ))
        line = p.line(
            "x", "y", source=src,
            line_width=4, color=lf14_palette[i], alpha=0.8,
            line_dash="solid", legend_label=label,
        )
        p.add_tools(HoverTool(renderers=[line], tooltips=[("", f"LF14: {label}")]))

    p.add_layout(p.legend[0], "right")
    p.legend.click_policy = "hide"

plot_models(p, comp_lines)

# ------------------------------------------------
# Final layout
# ------------------------------------------------
col1 = column(slider_teff, slider_period, slider_met, slider_eqt, margin=(0, 24, 0, 30))
col2 = column(slider_tsm, slider_massprec, slider_jmag, slider_ttv)
col3 = column(color_select, gray_toggle)
all_widgets = row(col1, col2, col3)
p.add_tools(
    HoverTool(
        renderers=[data_plot], tooltips=DATA_TOOLTIPS, formatters=DATA_FORMATTERS
    )
)

# Add the composition curves
plot_composition_curves(
    p, {"lf14": lf14, "fe": fe, "earth_like": earth_like, "water_rock": water_rock}
)


# Move legend outside the plot
p.add_layout(p.legend[0], "right")

# Use your real columns instead of 'x' and 'y'
# p.scatter(
#         "pl_bmasse",
#         "pl_rade",
#         source=source,
#         view=view,
#         fill_color={"field": "color", "transform": cmap},
#         size="marker_radius",
#         alpha=0.9,
#     )

# Preparing the layout
# Margins are top, right, bottom, left
col1 = column(slider_teff, slider_period, slider_met, slider_eqt,margin=(0, 24, 0, 0))
col2 = column(slider_tsm, slider_massprec, slider_jmag,slider_ttv)
col3 = column(checkbox_list)
all_widgets = row(col1,col2,col3)


layout = layout( column(p, all_widgets) )

# show(column(range_slider, p))
# show(layout) 

output_file("mrbokeh.html", mode="inline")  # embed BokehJS so it works offline
save(layout)
