"""Shared matplotlib style for SCOPE-Rx main figures."""
import matplotlib as mpl
import matplotlib.pyplot as plt

def use_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.dpi': 110,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })

# Palette
LIHC_COLOR = '#d4665a'  # warm red
LUAD_COLOR = '#4a8db8'  # blue
ATTENTION_COLOR = '#2ca02c'
NNLS_COLOR = '#d62728'
SCADEN_COLOR = '#9467bd'
GENEFORMER_COLOR = '#1f77b4'
SCVI_COLOR = '#ff7f0e'
PCA_COLOR = '#2ca02c'
RANDOM_COLOR = '#7f7f7f'

def panel_label(ax, label, x=-0.18, y=1.12, **kw):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left', **kw)

def save_both(fig, path):
    fig.savefig(f'{path}.png')
    fig.savefig(f'{path}.pdf')
