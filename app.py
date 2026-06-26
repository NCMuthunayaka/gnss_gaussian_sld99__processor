"""
Tkinter Application and Plotting Module

Implements the user interface, scatter/gaussian distribution plots,
and coordinates summary display using matplotlib and tkinter.
"""

import os
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Ellipse

from math_helpers import norm_fit, norm_pdf
from transformer import SLD99Transformer
from processor import parse_out_file, gaussian_iterative_rejection, weighted_mean

# UI Color Palette System
DARK   = "#0f172a"
PANEL  = "#1e293b"
BORDER = "#334155"
TEXT   = "#e2e8f0"
MUTED  = "#94a3b8"
BLUE   = "#38bdf8"
GREEN  = "#4ade80"
RED    = "#f87171"
AMBER  = "#fbbf24"
VIOLET = "#a78bfa"
BTN_BG = "#0ea5e9"
TEAL   = "#2dd4bf"

def _dms(deg):
    """Convert decimal degrees to Degrees Minutes Seconds (DMS) string representation."""
    d = int(abs(deg))
    m = int((abs(deg) - d) * 60)
    s = ((abs(deg) - d) * 60 - m) * 60
    sign = "-" if deg < 0 else ""
    return f"{sign}{d}\u00b0{m:02d}'{s:06.3f}\""

def enable_interactive_zoom_pan(ax, scroll_canvas):
    """Enables scroll wheel zoom on ax, and scrolls scroll_canvas otherwise."""
    fig = ax.get_figure()
    canvas = fig.canvas

    state = {
        'press': None,
        'xlim': None,
        'ylim': None
    }

    def on_scroll(event):
        if event.inaxes == ax:
            xdata = event.xdata
            ydata = event.ydata
            if xdata is None or ydata is None:
                return

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Zoom factor: zoom in with up scroll, out with down scroll
            scale_factor = 0.82 if event.button == 'up' else 1.22

            new_xlim = [xdata - (xdata - xlim[0]) * scale_factor,
                        xdata + (xlim[1] - xdata) * scale_factor]
            new_ylim = [ydata - (ydata - ylim[0]) * scale_factor,
                        ydata + (ylim[1] - ydata) * scale_factor]

            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            canvas.draw_idle()
        else:
            # Scroll parent canvas vertically
            if event.step:
                scroll_canvas.yview_scroll(int(-3 * event.step), "units")
            elif event.button == 'up':
                scroll_canvas.yview_scroll(-3, "units")
            elif event.button == 'down':
                scroll_canvas.yview_scroll(3, "units")

    def on_press(event):
        if event.inaxes != ax or event.dblclick:
            return
        
        # Avoid panning when a standard toolbar zoom/pan tool is active
        tb = fig.canvas.manager.toolbar if hasattr(fig.canvas, 'manager') and fig.canvas.manager else None
        if tb and tb.mode != '':
            return

        if event.button in [1, 3]: # Left-click or Right-click
            state['press'] = event.x, event.y
            state['xlim'] = ax.get_xlim()
            state['ylim'] = ax.get_ylim()

    def on_release(event):
        state['press'] = None
        canvas.draw_idle()

    def on_motion(event):
        if state['press'] is None or event.inaxes != ax:
            return
        tb = fig.canvas.manager.toolbar if hasattr(fig.canvas, 'manager') and fig.canvas.manager else None
        if tb and tb.mode != '':
            return

        dx = event.x - state['press'][0]
        dy = event.y - state['press'][1]

        # Convert to data coordinates
        inv = ax.transData.inverted()
        p0 = inv.transform((state['press'][0], state['press'][1]))
        p1 = inv.transform((event.x, event.y))

        d_data_x = p1[0] - p0[0]
        d_data_y = p1[1] - p0[1]

        ax.set_xlim(state['xlim'][0] - d_data_x, state['xlim'][1] - d_data_x)
        ax.set_ylim(state['ylim'][0] - d_data_y, state['ylim'][1] - d_data_y)
        canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.dblclick:
            if hasattr(ax, "cluster_limits") and hasattr(ax, "default_limits"):
                lims = ax.cluster_limits
                cur_xlim = ax.get_xlim()
                cur_w = cur_xlim[1] - cur_xlim[0]
                cln_w = lims[1] - lims[0]
                
                # Switch between cluster limits and default limits on double click
                if abs(cur_w - cln_w) / cln_w < 0.15:
                    lims = ax.default_limits
                else:
                    lims = ax.cluster_limits
                    
                ax.set_xlim(lims[0], lims[1])
                ax.set_ylim(lims[2], lims[3])
                canvas.draw_idle()

    canvas.mpl_connect('scroll_event', on_scroll)
    canvas.mpl_connect('button_press_event', on_press)
    canvas.mpl_connect('button_release_event', on_release)
    canvas.mpl_connect('motion_notify_event', on_motion)
    canvas.mpl_connect('button_press_event', on_click)

def bind_scroll_wheel(canvas):
    """Binds mouse wheel and touchpad scroll events to a Tkinter Canvas when hovered."""
    def _on_mousewheel(event):
        if event.delta:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    def _on_linux_scroll_up(event):
        canvas.yview_scroll(-1, "units")
        
    def _on_linux_scroll_down(event):
        canvas.yview_scroll(1, "units")

    def _on_enter(event):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_linux_scroll_up)
        canvas.bind_all("<Button-5>", _on_linux_scroll_down)

    def _on_leave(event):
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas.bind("<Enter>", _on_enter)
    canvas.bind("<Leave>", _on_leave)

class CustomNavigationToolbar(NavigationToolbar2Tk):
    """Custom navigation toolbar that filters out the subplot configuration tool."""
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if t[0] != 'Subplots']

def make_plot(raw, clean, result, last_iter, sld99=None):
    """Generate high-quality geodetic scatter plot and Gaussian error distribution graphs."""
    fig = plt.figure(figsize=(10, 22))
    fig.patch.set_facecolor("#0f172a")

    gs = gridspec.GridSpec(
        4, 1,
        hspace=0.45,
        left=0.08, right=0.78,
        top=0.93,  bottom=0.02)

    ax_sc  = fig.add_subplot(gs[0])
    ax_hx  = fig.add_subplot(gs[1])
    ax_hy  = fig.add_subplot(gs[2])
    ax_bar = fig.add_subplot(gs[3])

    for ax in (ax_sc, ax_hx, ax_hy, ax_bar):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COLOR := "#334155")

    cx = clean["X"].mean()
    cy = clean["Y"].mean()
    
    def to_as(vals, centre):
        return (vals - centre) * 3600.0

    raw_dx  = to_as(raw["X"], cx)
    raw_dy  = to_as(raw["Y"], cy)
    cln_dx  = to_as(clean["X"], cx)
    cln_dy  = to_as(clean["Y"], cy)
    res_dx  = to_as(result["X"], cx)
    res_dy  = to_as(result["Y"], cy)

    h_raw = ax_sc.scatter(raw_dx, raw_dy, s=8, alpha=0.18, color=BLUE, zorder=2)
    h_cln = ax_sc.scatter(cln_dx, cln_dy, s=14, alpha=0.70, color=GREEN, zorder=3)
    h_star = ax_sc.scatter(res_dx, res_dy, s=90, color=RED, marker="*", zorder=5)
    h_dot = ax_sc.scatter(res_dx, res_dy, s=8, color="white", marker="o", zorder=6)
    
    ellipse = Ellipse(
        xy=(res_dx, res_dy),
        width=4 * result["X_std"] * 3600, height=4 * result["Y_std"] * 3600,
        edgecolor=AMBER, facecolor="none",
        linewidth=1.6, linestyle="--", zorder=4)
    h_ellipse = ax_sc.add_patch(ellipse)

    ax_sc.set_xlabel("\u0394\u03bb  (arc-sec from cluster centre)", color=MUTED, fontsize=10)
    ax_sc.set_ylabel("\u0394\u03c6  (arc-sec from cluster centre)", color=MUTED, fontsize=10)
    ax_sc.set_title("Position Scatter — Gaussian Filtered",
                    color=TEXT, fontsize=12, fontweight="bold", pad=10)
    ax_sc.grid(True, color=BORDER, linewidth=0.6, zorder=1)
    # Calculate limits for cluster zoom, default zoom, and all data zoom
    min_cln_x, max_cln_x = cln_dx.min(), cln_dx.max()
    min_cln_y, max_cln_y = cln_dy.min(), cln_dy.max()
    w_cln = max_cln_x - min_cln_x
    h_cln = max_cln_y - min_cln_y
    w_cln = max(w_cln, 1e-4)
    h_cln = max(h_cln, 1e-4)
    ax_sc.cluster_limits = (min_cln_x - 0.1 * w_cln, max_cln_x + 0.1 * w_cln,
                            min_cln_y - 0.1 * h_cln, max_cln_y + 0.1 * h_cln)

    std_x = max(result["X_std"] * 3600, 1e-4)
    std_y = max(result["Y_std"] * 3600, 1e-4)
    ax_sc.default_limits = (res_dx - 4 * std_x, res_dx + 4 * std_x,
                            res_dy - 4 * std_y, res_dy + 4 * std_y)

    min_raw_x, max_raw_x = raw_dx.min(), raw_dx.max()
    min_raw_y, max_raw_y = raw_dy.min(), raw_dy.max()
    w_raw = max_raw_x - min_raw_x
    h_raw = max_raw_y - min_raw_y
    w_raw = max(w_raw, 1e-4)
    h_raw = max(h_raw, 1e-4)
    ax_sc.all_limits = (min_raw_x - 0.05 * w_raw, max_raw_x + 0.05 * w_raw,
                        min_raw_y - 0.05 * h_raw, max_raw_y + 0.05 * h_raw)

    ax_sc.set_xlim(ax_sc.default_limits[0], ax_sc.default_limits[1])
    ax_sc.set_ylim(ax_sc.default_limits[2], ax_sc.default_limits[3])
    import matplotlib.legend_handler as handler
    leg = ax_sc.legend(
        handles=[h_raw, h_cln, (h_star, h_dot), h_ellipse],
        labels=[f"All epochs ({len(raw)})", f"Clean epochs ({len(clean)})", "Weighted mean", "2\u03c3 ellipse"],
        handler_map={tuple: handler.HandlerTuple(ndivide=None)},
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=8, facecolor=DARK, edgecolor=BORDER, labelcolor=TEXT, framealpha=0.95)

    # Override legend markers to make them solid and larger for readability
    leg_handles = leg.legend_handles if hasattr(leg, "legend_handles") else leg.legendHandles
    for lh in leg_handles:
        if hasattr(lh, "set_alpha"):
            lh.set_alpha(1.0)
        if hasattr(lh, "set_sizes"):
            lh.set_sizes([40.0])


    for ax_h, col, colour, label in [
        (ax_hx, "X", BLUE, "Longitude (\u03bb) deviation (arc-sec)"),
        (ax_hy, "Y", GREEN, "Latitude  (\u03c6) deviation (arc-sec)"),
    ]:
        centre = clean[col].mean()
        data = (clean[col].values - centre) * 3600.0
        mu_f, sig_f = norm_fit(data)
        xr = np.linspace(data.min(), data.max(), 400)
        ax_h.hist(data, bins=40, density=True,
                  color=colour, alpha=0.35,
                  edgecolor=PANEL, linewidth=0.3, zorder=2)
        ax_h.plot(xr, norm_pdf(xr, mu_f, sig_f),
                  color=colour, linewidth=2.0, zorder=3,
                  label=f"\u03bc={mu_f:.4f}\"\n\u03c3={sig_f:.4f}\"")
        for mult, lc, ls in [(0.6745, MUTED, ":"),
                             (1.9600, AMBER, "--"),
                             (2.9680, RED, "-")]:
            for sign in (+1, -1):
                ax_h.axvline(mu_f + sign * mult * sig_f,
                             color=lc, linewidth=0.9, linestyle=ls, zorder=4)
        p = last_iter[f"sw_p_{col.lower()}"]
        pc = GREEN if p > 0.05 else RED
        ax_h.text(0.97, 0.95, f"SW p={p:.3f}",
                  transform=ax_h.transAxes, fontsize=7,
                  ha="right", va="top", color=pc, family="monospace",
                  bbox=dict(boxstyle="round,pad=0.3",
                            facecolor=DARK, edgecolor=BORDER, alpha=0.9))
        axis_name = "Longitude (\u03bb)" if col == "X" else "Latitude (\u03c6)"
        ax_h.set_title(f"Gaussian Fit \u2014 {axis_name}",
                       color=TEXT, fontsize=9, fontweight="bold", pad=5)
        ax_h.set_xlabel(label, color=MUTED, fontsize=8)
        ax_h.set_ylabel("Density", color=MUTED, fontsize=8)
        ax_h.grid(True, color=BORDER, linewidth=0.5, zorder=1)
        ax_h.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    fontsize=7, facecolor=DARK, edgecolor=BORDER,
                    labelcolor=TEXT, framealpha=0.9)

    n_clean = len(clean)
    zones = ["<1\u03c3\n68.3%", "1\u20132\u03c3\n27.2%",
             "2\u20133\u03c3\n4.2%", ">3\u03c3\n0.3%"]
    expected_p = [0.6827, 0.2718, 0.0430, 0.0027]
    counts_x = [last_iter[k] for k in ("z1_x", "z2_x", "z3_x", "z4_x")]
    counts_y = [last_iter[k] for k in ("z1_y", "z2_y", "z3_y", "z4_y")]
    expected = [p * n_clean for p in expected_p]
    x_pos = np.arange(4)
    w = 0.25

    bx = ax_bar.bar(x_pos - w, counts_x, w, color=BLUE, alpha=0.8,
                    label="\u03bb (Lon) observed", zorder=3)
    by = ax_bar.bar(x_pos, counts_y, w, color=GREEN, alpha=0.8,
                    label="\u03c6 (Lat) observed", zorder=3)
    be = ax_bar.bar(x_pos + w, expected, w, color=AMBER, alpha=0.6,
                    label="Expected", zorder=3)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(zones, fontsize=7.5, color=MUTED)
    ax_bar.set_ylabel("Epoch count", color=MUTED, fontsize=8)
    ax_bar.set_title("Sigma Zone Distribution",
                     color=TEXT, fontsize=9, fontweight="bold", pad=5)
    ax_bar.grid(True, axis="y", color=BORDER, linewidth=0.5, zorder=1)
    ax_bar.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=7, facecolor=DARK, edgecolor=BORDER,
                  labelcolor=TEXT, framealpha=0.9)
    for bars in (bx, by, be):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                            f"{int(round(h))}", ha="center", va="bottom",
                            fontsize=6, color=TEXT)


    fig.suptitle("GNSS Gaussian E-Threshold Analysis  \u00b7  SLD99 Transform  [EPSG:5235]",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.97)
    return fig

class App(tk.Tk):
    """Main application window using Tkinter and custom layout."""
    def __init__(self):
        super().__init__()
        self.title("GNSS Gaussian Processor + SLD99  [EPSG:5235]")
        self.configure(bg=DARK)
        self.geometry("1380x800")
        self.minsize(1100, 660)
        self.filepath = ""
        self.raw_df = None
        self.clean_df = None
        self.iters = None
        self.result = None
        self.sld99_res = None
        self._build()

    def _build(self):
        sb = tk.Frame(self, bg=PANEL, padx=18, pady=18)
        sb.pack(side="left", fill="y", padx=(10, 4), pady=10)

        tk.Label(sb, text="GNSS",
                 bg=PANEL, fg=BLUE,
                 font=("Courier New", 22, "bold")).pack(anchor="w")
        tk.Label(sb, text="Gaussian + SLD99",
                 bg=PANEL, fg=MUTED,
                 font=("Courier New", 10)).pack(anchor="w", pady=(0, 2))
        tk.Label(sb, text="Ghilani \u00a73.6  Iterative\nBlunder Rejection",
                 bg=PANEL, fg=VIOLET,
                 font=("Courier New", 9, "italic")).pack(anchor="w")

        self._sep(sb)
        self._section(sb, "INPUT FILE")
        self.lbl_file = tk.Label(sb, text="No file selected",
                                 bg="#0f2744", fg=MUTED,
                                 font=("Courier New", 9),
                                 wraplength=230, justify="left",
                                 padx=6, pady=5)
        self.lbl_file.pack(fill="x", pady=(4, 6))
        self._btn(sb, "  Browse .out file", self._browse, BTN_BG)

        self._sep(sb)
        self._btn(sb, "  Run Full Analysis", self._run, "#16a34a",
                  font=("Courier New", 12, "bold"), pady=10)

        self._sep(sb)
        self._section(sb, "RESULTS SUMMARY")
        self.res_box = tk.Text(sb, bg="#0f172a", fg=TEXT,
                               font=("Courier New", 9),
                               relief="flat", height=30,
                               state="disabled", bd=0,
                               width=28, padx=10, pady=8)
        self.res_box.pack(fill="x", pady=(4, 0))
        for tag, fg, bold in [
            ("lbl",  MUTED,  False), ("val",  BLUE,   True),
            ("ok",   GREEN,  True),  ("warn", AMBER,  False),
            ("err",  RED,    True),  ("sld",  TEAL,   True),
            ("slbl", TEAL,   False),
        ]:
            self.res_box.tag_configure(
                tag, foreground=fg,
                font=("Courier New", 9, "bold") if bold else ("Courier New", 9))

        right = tk.Frame(self, bg=DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook", background=DARK, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background=BORDER, foreground=MUTED,
                        font=("Courier New", 9), padding=(14, 6))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", BTN_BG)],
                  foreground=[("selected", DARK)])

        self.nb = ttk.Notebook(right, style="Dark.TNotebook")
        self.nb.pack(fill="both", expand=True)

        lf = tk.Frame(self.nb, bg="#0b1120")
        self.nb.add(lf, text="  Log  ")
        self.log_txt = tk.Text(lf, bg="#0b1120", fg=TEXT,
                               font=("Courier New", 9), relief="flat",
                               state="disabled", wrap="word", bd=0,
                               padx=10, pady=10)
        sbl = tk.Scrollbar(lf, command=self.log_txt.yview,
                           relief="flat", bg=BORDER)
        self.log_txt.configure(yscrollcommand=sbl.set)
        sbl.pack(side="right", fill="y")
        self.log_txt.pack(fill="both", expand=True)
        for tag, fg, bold in [
            ("ok",   GREEN, True), ("err",  RED,   True),
            ("hd",   BLUE,  True), ("info", MUTED, False),
            ("warn", AMBER, False),("sld",  TEAL,  True),
        ]:
            self.log_txt.tag_configure(
                tag, foreground=fg,
                font=("Courier New", 9, "bold") if bold else ("Courier New", 9))

        self.tbl_frame = tk.Frame(self.nb, bg=DARK)
        self.nb.add(self.tbl_frame, text="  Statistics Table  ")

        self.sld_frame = tk.Frame(self.nb, bg=DARK)
        self.nb.add(self.sld_frame, text="  SLD99 Transform  ")
        self._build_sld99_panel()

        self.plot_frame = tk.Frame(self.nb, bg=DARK)
        self.nb.add(self.plot_frame, text="  Scatter + Gaussian Plot  ")

        # Controls panel at the top (stays fixed, does not scroll)
        self.plot_ctrl = tk.Frame(self.plot_frame, bg=PANEL, padx=10, pady=6)
        self.plot_ctrl.pack(side="top", fill="x")
        self.lbl_ctrl_placeholder = tk.Label(self.plot_ctrl, text="Plot controls will appear here after running analysis",
                                             bg=PANEL, fg=MUTED, font=("Courier New", 9, "italic"))
        self.lbl_ctrl_placeholder.pack(side="left", pady=2)

        # Scrollable container for plots
        self.plot_canvas = tk.Canvas(self.plot_frame, bg=DARK, highlightthickness=0)
        self.plot_vsb = tk.Scrollbar(self.plot_frame, orient="vertical",
                                     command=self.plot_canvas.yview, bg=BORDER, relief="flat")
        self.plot_canvas.configure(yscrollcommand=self.plot_vsb.set)
        self.plot_vsb.pack(side="right", fill="y")
        self.plot_canvas.pack(side="left", fill="both", expand=True)
        bind_scroll_wheel(self.plot_canvas)

        self.plot_inner = tk.Frame(self.plot_canvas, bg=DARK)
        self.plot_canvas_id = self.plot_canvas.create_window((0, 0), window=self.plot_inner,
                                                             anchor="nw")

        def _on_plot_resize(e):
            self.plot_canvas.itemconfig(self.plot_canvas_id, width=e.width)
        self.plot_canvas.bind("<Configure>", _on_plot_resize)

        def _on_plot_inner(e):
            self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))
        self.plot_inner.bind("<Configure>", _on_plot_inner)

    def _build_sld99_panel(self):
        tk.Label(self.sld_frame,
                 text="SLD99 Coordinate Output  [EPSG:5235]",
                 bg=DARK, fg=TEAL,
                 font=("Courier New", 14, "bold")).pack(anchor="w", padx=20, pady=(18, 2))
        tk.Label(self.sld_frame,
                 text="Sri Lanka Datum 1999  \u00b7  Transverse Mercator  \u00b7  FE=FN=500 000 m",
                 bg=DARK, fg=MUTED,
                 font=("Courier New", 9)).pack(anchor="w", padx=20, pady=(0, 14))

        canvas = tk.Canvas(self.sld_frame, bg=DARK, highlightthickness=0)
        vsb = tk.Scrollbar(self.sld_frame, orient="vertical",
                           command=canvas.yview, bg=BORDER, relief="flat")
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 12))
        bind_scroll_wheel(canvas)

        self._sld_canvas = canvas
        self._sld_inner = tk.Frame(canvas, bg=DARK)
        self._sld_canvas_id = canvas.create_window((0, 0), window=self._sld_inner,
                                                   anchor="nw")

        def _on_resize(e):
            canvas.itemconfig(self._sld_canvas_id, width=e.width)
        canvas.bind("<Configure>", _on_resize)

        def _on_inner(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self._sld_inner.bind("<Configure>", _on_inner)

        self._sld_placeholder()

    def _sep(self, p):
        tk.Frame(p, bg=BORDER, height=1).pack(fill="x", pady=10)

    def _section(self, p, text):
        tk.Label(p, text=text, bg=PANEL, fg=AMBER,
                 font=("Courier New", 9, "bold")).pack(anchor="w")

    def _btn(self, p, text, cmd, bg, font=None, pady=7):
        tk.Button(p, text=text, command=cmd, bg=bg, fg=DARK,
                  font=font or ("Courier New", 10, "bold"),
                  relief="flat", pady=pady, cursor="hand2",
                  activebackground=BLUE,
                  activeforeground=DARK).pack(fill="x", pady=(4, 0))

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select gLAB .out file",
            filetypes=[("OUT files", "*.out"), ("All files", "*.*")])
        if path:
            self.filepath = path
            self.lbl_file.config(text=os.path.basename(path), fg=TEXT)
            self._log(f"File: {path}\n", "info")

    def _run(self):
        if not self.filepath:
            messagebox.showwarning("No File", "Select a .out file first.")
            return
        try:
            self._log("\u2501\u2501 Step 1: Parsing OUTPUT lines \u2501\u2501\n", "hd")
            self._log("   Reading: col[10]=PDOP  col[20]=Lat  col[21]=Lon  col[22]=H\n", "info")
            self.raw_df = parse_out_file(self.filepath)
            self._log(f"   {len(self.raw_df)} epochs parsed\n", "ok")
            sample = self.raw_df.iloc[0]
            self._log(f"   Sample epoch: Lat={sample['Lat']:.8f}\u00b0  "
                      f"Lon={sample['Lon']:.8f}\u00b0  "
                      f"H={sample['Height']:.3f}m  "
                      f"PDOP={sample['PDOP']:.4f}\n", "info")

            self._log("\u2501\u2501 Step 2: Gaussian iterative rejection \u2501\u2501\n", "hd")
            self.clean_df, self.iters = gaussian_iterative_rejection(
                self.raw_df, log_cb=self._log)
            removed = len(self.raw_df) - len(self.clean_df)
            self._log(f"   Removed : {removed} epochs "
                      f"({removed/len(self.raw_df)*100:.1f}%)\n", "warn")
            self._log(f"   Clean   : {len(self.clean_df)} epochs\n", "ok")

            self._log("\u2501\u2501 Step 3: PDOP-weighted mean \u2501\u2501\n", "hd")
            self.result = weighted_mean(self.clean_df)
            r = self.result
            self._log(f"   Latitude  (\u03c6) = {r['Lat']:.8f}\u00b0  [{_dms(r['Lat'])} N]\n", "ok")
            self._log(f"   Longitude (\u03bb) = {r['Lon']:.8f}\u00b0  [{_dms(r['Lon'])} E]\n", "ok")
            self._log(f"   Height    (h) = {r['Height']:.3f} m\n", "ok")
            self._log(f"   S\u0304\u03c6 = {r['S_mean_lat']*3600*1000:.4f} mas  "
                      f"S\u0304\u03bb = {r['S_mean_lon']*3600*1000:.4f} mas\n", "info")

            self._log("\u2501\u2501 Step 4: WGS84 \u2192 SLD99 Transform [EPSG:5235] \u2501\u2501\n", "sld")
            self.sld99_res = SLD99Transformer.transform(r["Lat"], r["Lon"], r["Height"])
            s = self.sld99_res
            self._log(f"   SLD99 Easting  = {s['SLD99_E']:.3f} m\n", "sld")
            self._log(f"   SLD99 Northing = {s['SLD99_N']:.3f} m\n", "sld")
            self._log(f"   Everest Lat    = {_dms(s['eve_lat_deg'])} N\n", "info")
            self._log(f"   Everest Lon    = {_dms(s['eve_lon_deg'])} E\n", "info")
            self._log("   Done \u2713\n", "ok")

            self._show_results()
            self._display_sld99(r["Lat"], r["Lon"], r["Height"], self.sld99_res)
            self._build_table()
            self._show_plot()

        except Exception as e:
            import traceback
            self._log(f"   ERROR: {e}\n{traceback.format_exc()}\n", "err")
            messagebox.showerror("Error", str(e))

    def _log(self, msg, tag="info"):
        self.log_txt.configure(state="normal")
        self.log_txt.insert("end", msg, tag)
        self.log_txt.see("end")
        self.log_txt.configure(state="disabled")

    def _show_results(self):
        r = self.result
        last = self.iters[-1]
        s = self.sld99_res
        self.res_box.configure(state="normal")
        self.res_box.delete("1.0", "end")
        rows = [
            ("lbl", "\u2500\u2500 WGS84 Geographic \u2500\u2500\u2500\u2500\n"),
            ("lbl", "Lat (\u03c6): "), ("val", f"{r['Lat']:.8f}\u00b0\n"),
            ("lbl", "        "), ("lbl", f"{_dms(r['Lat'])} N\n"),
            ("lbl", "Lon (\u03bb): "), ("val", f"{r['Lon']:.8f}\u00b0\n"),
            ("lbl", "        "), ("lbl", f"{_dms(r['Lon'])} E\n"),
            ("lbl", "Height : "), ("val", f"{r['Height']:.3f} m\n"),
            ("lbl", "\n\u2500\u2500 Precision \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"),
            ("lbl", "\u03c3\u03c6      : "), ("ok",  f"{r['Lat_std']*3600:.5f}\"\n"),
            ("lbl", "\u03c3\u03bb      : "), ("ok",  f"{r['Lon_std']*3600:.5f}\"\n"),
            ("lbl", "S\u0304\u03c6     : "), ("ok",  f"{r['S_mean_lat']*3600*1000:.4f} mas\n"),
            ("lbl", "S\u0304\u03bb     : "), ("ok",  f"{r['S_mean_lon']*3600*1000:.4f} mas\n"),
            ("lbl", "\n\u2500\u2500 SLD99 [EPSG:5235] \u2500\u2500\u2500\n"),
        ]
        if s:
            rows += [
                ("slbl", "Easting : "), ("sld", f"{s['SLD99_E']:.3f} m\n"),
                ("slbl", "Northing: "), ("sld", f"{s['SLD99_N']:.3f} m\n"),
                ("slbl", "Eve.Lat : "), ("lbl", f"{s['eve_lat_deg']:.6f}\u00b0\n"),
                ("slbl", "Eve.Lon : "), ("lbl", f"{s['eve_lon_deg']:.6f}\u00b0\n"),
            ]
        else:
            rows += [("lbl", "  Not yet computed\n")]
        rows += [
            ("lbl", "\n\u2500\u2500 Session \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"),
            ("lbl", "Clean n  : "), ("val", f"{r['n_epochs']}\n"),
            ("lbl", "Raw n    : "), ("lbl", f"{len(self.raw_df)}\n"),
            ("lbl", "Removed  : "),("warn", f"{len(self.raw_df)-r['n_epochs']}\n"),
            ("lbl", "Iters    : "),("val",  f"{len(self.iters)}\n"),
            ("lbl", "Mean PDOP: "),("val",  f"{r['PDOP_mean']:.3f}\n"),
            ("lbl", "\n\u2500\u2500 E-Thresholds (final)\n"),
            ("lbl", "E99.7 \u03bb : "),("warn", f"{last['E997_lon']*3600:.5f}\"\n"),
            ("lbl", "E99.7 \u03c6 : "),("warn", f"{last['E997_lat']*3600:.5f}\"\n"),
        ]
        for tag, text in rows:
            self.res_box.insert("end", text, tag)
        self.res_box.configure(state="disabled")

    def _sld_placeholder(self):
        for w in self._sld_inner.winfo_children():
            w.destroy()
        tk.Label(self._sld_inner,
                 text="\n\n   Run Full Analysis to see SLD99 results here.",
                 bg=DARK, fg=MUTED,
                 font=("Courier New", 10, "italic")).pack(anchor="w", padx=20)

    def _display_sld99(self, lat, lon, h, s):
        f = self._sld_inner
        for w in f.winfo_children():
            w.destroy()

        def _card(parent, title, title_fg=TEAL):
            outer = tk.Frame(parent, bg=PANEL, padx=0, pady=0)
            outer.pack(fill="x", padx=20, pady=(0, 14))
            tk.Frame(outer, bg=title_fg, height=3).pack(fill="x")
            inner = tk.Frame(outer, bg=PANEL, padx=16, pady=12)
            inner.pack(fill="x")
            tk.Label(inner, text=title, bg=PANEL, fg=title_fg,
                     font=("Courier New", 10, "bold")).pack(anchor="w", pady=(0, 8))
            return inner

        def _row(parent, label, value, val_fg=TEXT, val_size=10):
            row = tk.Frame(parent, bg=PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=PANEL, fg=MUTED,
                     font=("Courier New", 9), width=22, anchor="w").pack(side="left")
            tk.Label(row, text=value, bg=PANEL, fg=val_fg,
                     font=("Courier New", val_size, "bold")).pack(side="left")

        def _divider(parent):
            tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=6)

        hero = tk.Frame(f, bg="#0a1f3d", padx=0, pady=0)
        hero.pack(fill="x", padx=20, pady=(10, 18))
        tk.Frame(hero, bg=TEAL, height=4).pack(fill="x")
        inner_hero = tk.Frame(hero, bg="#0a1f3d", padx=24, pady=20)
        inner_hero.pack(fill="x")
        tk.Label(inner_hero, text="SLD99 FINAL COORDINATES  [EPSG:5235]",
                 bg="#0a1f3d", fg=TEAL,
                 font=("Courier New", 11, "bold")).pack(anchor="w")
        tk.Frame(inner_hero, bg=BORDER, height=1).pack(fill="x", pady=(6, 14))
        coords_frame = tk.Frame(inner_hero, bg="#0a1f3d")
        coords_frame.pack(fill="x")

        e_block = tk.Frame(coords_frame, bg="#0d2a4a", padx=20, pady=16)
        e_block.pack(side="left", expand=True, fill="both", padx=(0, 8))
        tk.Label(e_block, text="EASTING  (E)", bg="#0d2a4a", fg=MUTED,
                 font=("Courier New", 8, "bold")).pack(anchor="w")
        tk.Label(e_block, text=f"{s['SLD99_E']:,.3f}", bg="#0d2a4a", fg=BLUE,
                 font=("Courier New", 18, "bold")).pack(anchor="w", pady=(4, 0))
        tk.Label(e_block, text="metres", bg="#0d2a4a", fg=MUTED,
                 font=("Courier New", 8)).pack(anchor="w")

        n_block = tk.Frame(coords_frame, bg="#0d3a1a", padx=20, pady=16)
        n_block.pack(side="left", expand=True, fill="both", padx=(8, 0))
        tk.Label(n_block, text="NORTHING  (N)", bg="#0d3a1a", fg=MUTED,
                 font=("Courier New", 8, "bold")).pack(anchor="w")
        tk.Label(n_block, text=f"{s['SLD99_N']:,.3f}", bg="#0d3a1a", fg=GREEN,
                 font=("Courier New", 18, "bold")).pack(anchor="w", pady=(4, 0))
        tk.Label(n_block, text="metres", bg="#0d3a1a", fg=MUTED,
                 font=("Courier New", 8)).pack(anchor="w")

        c1 = _card(f, "INPUT \u2014 WGS84 Geographic (weighted mean from file)", AMBER)
        _row(c1, "Latitude  (\u03c6)",  f"{lat:.8f}\u00b0")
        _row(c1, "            ",   f"{_dms(lat)} N", MUTED, 9)
        _divider(c1)
        _row(c1, "Longitude (\u03bb)",  f"{lon:.8f}\u00b0")
        _row(c1, "            ",   f"{_dms(lon)} E", MUTED, 9)
        _divider(c1)
        _row(c1, "Height (h)",     f"{h:.3f} m")

        c2 = _card(f, "EVEREST 1830 (1937 ADJ) \u2014 Geodetic", VIOLET)
        _row(c2, "Latitude  (\u03c6)",  f"{s['eve_lat_deg']:.8f}\u00b0")
        _row(c2, "            ",   f"{_dms(s['eve_lat_deg'])} N", MUTED, 9)
        _divider(c2)
        _row(c2, "Longitude (\u03bb)",  f"{s['eve_lon_deg']:.8f}\u00b0")
        _row(c2, "            ",   f"{_dms(s['eve_lon_deg'])} E", MUTED, 9)

    def _build_table(self):
        for w in self.tbl_frame.winfo_children():
            w.destroy()

        style = ttk.Style()
        style.configure("Stats.Treeview",
                        background=PANEL, foreground=TEXT,
                        fieldbackground=PANEL, rowheight=24,
                        font=("Courier New", 9), borderwidth=0)
        style.configure("Stats.Treeview.Heading",
                        background=BORDER, foreground=AMBER,
                        font=("Courier New", 9, "bold"), relief="flat")
        style.map("Stats.Treeview",
                  background=[("selected", BTN_BG)],
                  foreground=[("selected", DARK)])

        def heading(text, fg=BLUE):
            tk.Label(self.tbl_frame, text=text, bg=DARK, fg=fg,
                     font=("Courier New", 11, "bold")).pack(
                anchor="w", padx=10, pady=(10, 3))

        def make_tree(cols, widths, height):
            frame = tk.Frame(self.tbl_frame, bg=DARK)
            frame.pack(fill="x", padx=10, pady=(0, 4))
            sbx = tk.Scrollbar(frame, orient="horizontal",
                               bg=BORDER, relief="flat")
            tree = ttk.Treeview(frame, columns=cols, show="headings",
                                height=height, style="Stats.Treeview",
                                xscrollcommand=sbx.set)
            sbx.config(command=tree.xview)
            sbx.pack(side="bottom", fill="x")
            tree.pack(fill="x")
            for col, w in zip(cols, widths):
                tree.heading(col, text=col, anchor="center")
                tree.column(col, width=w, anchor="center", stretch=False)
            tree.tag_configure("even", background="#1a2942", foreground=TEXT)
            tree.tag_configure("odd",  background=PANEL,    foreground=TEXT)
            tree.tag_configure("last", background="#0f2744", foreground=GREEN)
            return tree

        heading("\u2460 Per-Iteration Statistics")
        c1 = ("Iter","N start","Flagged",
              "Mean Lon (\u00b0)","Mean Lat (\u00b0)",
              "S\u03bb (\u00b0)","S\u03c6 (\u00b0)",
              "E99.7 \u03bb (\u00b0)","E99.7 \u03c6 (\u00b0)",
              "SW-p \u03bb","SW-p \u03c6")
        w1 = [46,72,62,140,140,110,110,110,110,76,76]
        t1 = make_tree(c1, w1, min(len(self.iters)+1, 7))
        for i, it in enumerate(self.iters):
            tag = "last" if i == len(self.iters)-1 else \
                  ("even" if i%2==0 else "odd")
            t1.insert("","end",tags=(tag,), values=(
                it["iteration"], it["n_start"], it["n_flagged"],
                f"{it['mean_lon']:.8f}", f"{it['mean_lat']:.8f}",
                f"{it['S_lon']:.8f}",    f"{it['S_lat']:.8f}",
                f"{it['E997_lon']:.8f}", f"{it['E997_lat']:.8f}",
                f"{it['sw_p_x']:.4f}", f"{it['sw_p_y']:.4f}",
            ))

        heading("\u2461 Sigma Zone Distribution  (final clean set)")
        last = self.iters[-1]
        n_clean = len(self.clean_df)
        ep_p = [0.6827, 0.2718, 0.0430, 0.0027]
        znames = ["< 1\u03c3  (68.3%)", "1\u20132\u03c3  (27.2%)",
                  "2\u20133\u03c3  (4.2%)",  "> 3\u03c3  (0.3%)"]
        xobs = [last[k] for k in ("z1_x","z2_x","z3_x","z4_x")]
        yobs = [last[k] for k in ("z1_y","z2_y","z3_y","z4_y")]
        c2 = ("Zone","Exp %","Exp n",
              "\u03bb obs","\u03bb %","\u03bb \u0394",
              "\u03c6 obs","\u03c6 %","\u03c6 \u0394")
        w2 = [132,68,68, 68,62,62, 68,62,62]
        t2 = make_tree(c2, w2, 4)
        ztags = {"z1":(GREEN,"#14532d"), "z2":(BLUE,"#1e3a5f"),
                 "z3":(AMBER,"#4a3000"),"z4":(RED, "#4a0000")}
        for ztag,(fg,bg) in ztags.items():
            t2.tag_configure(ztag, foreground=fg, background=bg)
        for idx,(zn,ep,xt,yt) in enumerate(zip(znames,ep_p,xobs,yobs)):
            exp_n = ep * n_clean
            t2.insert("","end",tags=(list(ztags)[idx],), values=(
                zn, f"{ep*100:.1f}%", f"{exp_n:.0f}",
                xt, f"{xt/n_clean*100:.1f}%", f"{xt-exp_n:+.0f}",
                yt, f"{yt/n_clean*100:.1f}%", f"{yt-exp_n:+.0f}",
            ))

        heading("\u2462 Final Coordinate Summary  (WGS84 Geographic)  \u2014  see SLD99 tab for projected output")
        r = self.result
        c3 = ("Axis","WGS84 Mean (\u00b0)","Std Dev (\u00b0)",
              "Std of Mean (mas)","E\u2085\u2080 (\u00b0)","E\u2089\u2085 (\u00b0)","E\u2089\u2089.\u2087 (\u00b0)")
        w3 = [80,170,120,130,120,120,120]
        t3 = make_tree(c3, w3, 2)
        t3.tag_configure("X", background="#0f2744", foreground=BLUE)
        t3.tag_configure("Y", background="#14532d", foreground=GREEN)
        for axis,mv,sv,smv,e50,e95,e997,tag in [
            ("Lon (\u03bb)",r["Lon"],r["Lon_std"],r["S_mean_lon"],
             last["E50_lon"],last["E95_lon"],last["E997_lon"],"X"),
            ("Lat (\u03c6)",r["Lat"],r["Lat_std"],r["S_mean_lat"],
             last["E50_lat"],last["E95_lat"],last["E997_lat"],"Y"),
        ]:
            t3.insert("","end",tags=(tag,), values=(
                axis,
                f"{mv:.8f}", f"{sv:.8f}", f"{smv*3600*1000:.4f}",
                f"{e50:.8f}", f"{e95:.8f}", f"{e997:.8f}",
            ))

    def _show_plot(self):
        for w in self.plot_inner.winfo_children():
            w.destroy()
        fig = make_plot(self.raw_df, self.clean_df,
                        self.result, self.iters[-1], self.sld99_res)
        
        ax_sc = fig.axes[0]
        enable_interactive_zoom_pan(ax_sc, self.plot_canvas)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_inner)
        canvas.draw()
        
        # Clear control panel placeholder and populate control elements
        for w in self.plot_ctrl.winfo_children():
            w.destroy()

        # Add interactive zoom/pan toolbar at the top (left-side of control panel)
        toolbar = CustomNavigationToolbar(canvas, self.plot_ctrl, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="left", fill="y", padx=5)

        # Separator inside control panel
        sep = tk.Frame(self.plot_ctrl, bg=BORDER, width=2)
        sep.pack(side="left", fill="y", padx=15)

        # Quick Zoom Frame inside control panel
        zoom_frame = tk.Frame(self.plot_ctrl, bg=PANEL)
        zoom_frame.pack(side="left", fill="y")

        tk.Label(zoom_frame, text="Quick Zoom:", bg=PANEL, fg=MUTED,
                 font=("Courier New", 9, "bold")).pack(side="left", padx=(0, 6), pady=5)

        def zoom_to_cluster():
            if hasattr(ax_sc, "cluster_limits"):
                lims = ax_sc.cluster_limits
                ax_sc.set_xlim(lims[0], lims[1])
                ax_sc.set_ylim(lims[2], lims[3])
                canvas.draw_idle()

        def zoom_to_default():
            if hasattr(ax_sc, "default_limits"):
                lims = ax_sc.default_limits
                ax_sc.set_xlim(lims[0], lims[1])
                ax_sc.set_ylim(lims[2], lims[3])
                canvas.draw_idle()

        def zoom_to_all():
            if hasattr(ax_sc, "all_limits"):
                lims = ax_sc.all_limits
                ax_sc.set_xlim(lims[0], lims[1])
                ax_sc.set_ylim(lims[2], lims[3])
                canvas.draw_idle()

        def make_zoom_btn(text, cmd, color):
            btn = tk.Button(zoom_frame, text=text, command=cmd, bg=color, fg=DARK,
                            font=("Courier New", 9, "bold"), relief="flat", padx=10, pady=2,
                            cursor="hand2", activebackground=BLUE, activeforeground=DARK)
            btn.pack(side="left", padx=4, pady=2)
            return btn

        make_zoom_btn("Cluster (Clean)", zoom_to_cluster, GREEN)
        make_zoom_btn("Default (4\u03c3)", zoom_to_default, BLUE)
        make_zoom_btn("All Epochs (Raw)", zoom_to_all, AMBER)

        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.nb.select(3)
