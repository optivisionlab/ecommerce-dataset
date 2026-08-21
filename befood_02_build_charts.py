import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

df = pd.read_pickle('/home/claude/viz/df.pkl')

# ---- palette (matches the HTML map theme) ----
INK     = '#241608'
PAPER   = '#fbf2e3'
PAPER2  = '#f3e4c9'
CHILI   = '#d5451b'
CHILID  = '#a8330f'
TURM    = '#f2a007'
LEAF    = '#3e7a4d'
LINE    = '#e2c99b'

CAT_COLORS = {
    "Cà Phê/Trà":"#6f4e2c",
    "Cơm/Cơm tấm":"#d5451b",
    "Bún/Phở/Mỳ/Cháo":"#a8330f",
    "Sinh tố/Nước ép/Sữa":"#3e7a4d",
    "Thức ăn nhanh":"#f2a007",
    "Ăn vặt":"#c2185b",
    "Ẩm thực quốc tế":"#3f51b5",
    "Trà sữa":"#8e5fb0",
    "Món truyền thống/Đặc sản":"#b8860b",
    "Bánh Mì/Xôi":"#e07a1f",
    "Tráng miệng":"#e91e63",
    "Lẩu & Nướng":"#c0392b",
    "Pizza":"#d84315",
    "Ốc/Cá/Hải sản":"#0288a8",
    "Món chay":"#43913e",
    "Mart":"#607d8b",
    "Món Dinh dưỡng/Bổ dưỡng":"#009688"
}

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = LINE
plt.rcParams['axes.labelcolor'] = INK
plt.rcParams['xtick.color'] = INK
plt.rcParams['ytick.color'] = INK
plt.rcParams['text.color'] = INK
plt.rcParams['figure.facecolor'] = PAPER
plt.rcParams['axes.facecolor'] = PAPER

def style_ax(ax, grid_axis='y'):
    ax.set_facecolor(PAPER)
    for s in ['top','right']:
        ax.spines[s].set_visible(False)
    for s in ['left','bottom']:
        ax.spines[s].set_color(LINE)
    ax.grid(axis=grid_axis, color=LINE, linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

# =====================================================================
# FIGURE 1: Geographic scatter — Hà Nội & TP.HCM side by side (colored by category)
# =====================================================================
def geo_fig(figsize=(14,7.5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    cities = [('Hà Nội', 190), ('TP.HCM', 189)]
    for ax, (name, code) in zip(axes, cities):
        sub = df[df['city'] == code]
        for cat, color in CAT_COLORS.items():
            s = sub[sub['merchant_category'] == cat]
            if len(s) == 0: continue
            ax.scatter(s['longitude'], s['latitude'], s=7, color=color, alpha=0.65,
                       linewidths=0, label=cat)
        ax.set_title(f"{name}  ({len(sub):,} quán)", fontsize=15, fontweight='bold',
                     color=INK, family='DejaVu Sans', pad=10)
        ax.set_xlabel('Kinh độ', fontsize=9)
        ax.set_ylabel('Vĩ độ', fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')
        style_ax(ax, grid_axis='both')
        ax.tick_params(labelsize=8)

    handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=7, label=cat)
               for cat, c in CAT_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Phân bố vị trí quán ăn beFood theo ngành hàng', fontsize=18, fontweight='bold',
                 color=CHILID, y=1.02, family='DejaVu Sans')
    fig.tight_layout()
    return fig

fig1 = geo_fig()
fig1.savefig('/home/claude/viz/01_geo_scatter.svg', format='svg', bbox_inches='tight', facecolor=PAPER)
fig1.savefig('/home/claude/viz/01_geo_scatter.png', format='png', dpi=170, bbox_inches='tight', facecolor=PAPER)
plt.close(fig1)
print('fig1 done')

# =====================================================================
# FIGURE 2: Density heatmap (hexbin) per city
# =====================================================================
def density_fig(figsize=(14,7)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    cities = [('Hà Nội', 190), ('TP.HCM', 189)]
    for ax, (name, code) in zip(axes, cities):
        sub = df[df['city'] == code]
        hb = ax.hexbin(sub['longitude'], sub['latitude'], gridsize=45, cmap='YlOrRd', mincnt=1)
        ax.set_title(f"Mật độ quán ăn — {name}", fontsize=14, fontweight='bold', color=INK)
        ax.set_xlabel('Kinh độ', fontsize=9)
        ax.set_ylabel('Vĩ độ', fontsize=9)
        ax.set_aspect('equal', adjustable='datalim')
        style_ax(ax, grid_axis='both')
        cb = fig.colorbar(hb, ax=ax, shrink=0.75)
        cb.set_label('Số quán / ô', fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle('Bản đồ mật độ (heatmap) quán ăn', fontsize=18, fontweight='bold', color=CHILID, y=1.03)
    fig.tight_layout()
    return fig

fig2 = density_fig()
fig2.savefig('/home/claude/viz/02_density_heatmap.svg', format='svg', bbox_inches='tight', facecolor=PAPER)
fig2.savefig('/home/claude/viz/02_density_heatmap.png', format='png', dpi=170, bbox_inches='tight', facecolor=PAPER)
plt.close(fig2)
print('fig2 done')

# =====================================================================
# FIGURE 3: Top categories bar chart (grouped by city)
# =====================================================================
def category_bar_fig(figsize=(12,7)):
    ct = df.groupby(['merchant_category','city_name']).size().unstack(fill_value=0)
    ct['total'] = ct.sum(axis=1)
    ct = ct.sort_values('total', ascending=True)
    ct = ct.drop(columns='total')

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    y = np.arange(len(ct))
    bar_h = 0.38
    ax.barh(y+bar_h/2, ct['Hà Nội'], height=bar_h, color=TURM, label='Hà Nội')
    ax.barh(y-bar_h/2, ct['TP.HCM'], height=bar_h, color=CHILI, label='TP.HCM')
    ax.set_yticks(y)
    ax.set_yticklabels(ct.index, fontsize=10)
    ax.set_xlabel('Số lượng quán', fontsize=10)
    ax.set_title('Số lượng quán theo ngành hàng & thành phố', fontsize=16, fontweight='bold', color=CHILID, pad=14)
    style_ax(ax, grid_axis='x')
    ax.legend(frameon=False, fontsize=10, loc='lower right')
    fig.tight_layout()
    return fig

fig3 = category_bar_fig()
fig3.savefig('/home/claude/viz/03_category_bar.svg', format='svg', bbox_inches='tight', facecolor=PAPER)
fig3.savefig('/home/claude/viz/03_category_bar.png', format='png', dpi=170, bbox_inches='tight', facecolor=PAPER)
plt.close(fig3)
print('fig3 done')

# =====================================================================
# FIGURE 4: Rating distribution (histogram) per city
# =====================================================================
def rating_fig(figsize=(12,6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    bins = np.arange(2.0, 5.05, 0.1)
    for name, color in [('Hà Nội', TURM), ('TP.HCM', CHILI)]:
        sub = df[(df['city_name']==name) & df['rating'].notna()]
        ax.hist(sub['rating'], bins=bins, alpha=0.6, color=color, label=f"{name} (TB {sub['rating'].mean():.2f})",
                edgecolor='white', linewidth=0.4)
    ax.set_xlabel('Rating', fontsize=10)
    ax.set_ylabel('Số lượng quán', fontsize=10)
    ax.set_title('Phân phối rating quán ăn theo thành phố', fontsize=16, fontweight='bold', color=CHILID, pad=14)
    style_ax(ax, grid_axis='y')
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    return fig

fig4 = rating_fig()
fig4.savefig('/home/claude/viz/04_rating_dist.svg', format='svg', bbox_inches='tight', facecolor=PAPER)
fig4.savefig('/home/claude/viz/04_rating_dist.png', format='png', dpi=170, bbox_inches='tight', facecolor=PAPER)
plt.close(fig4)
print('fig4 done')

# =====================================================================
# FIGURE 5: Top districts (khu vực) by restaurant count
# =====================================================================
def district_fig(figsize=(12,10)):
    dc = df.groupby(['district','city_name']).size().reset_index(name='count')
    dc = dc.sort_values('count', ascending=True).tail(25)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    colors = [TURM if c=='Hà Nội' else CHILI for c in dc['city_name']]
    ax.barh(dc['district'], dc['count'], color=colors)
    ax.set_xlabel('Số lượng quán', fontsize=10)
    ax.set_title('Top 25 khu vực có nhiều quán ăn nhất', fontsize=16, fontweight='bold', color=CHILID, pad=14)
    style_ax(ax, grid_axis='x')
    handles = [plt.Rectangle((0,0),1,1, color=TURM, label='Hà Nội'),
               plt.Rectangle((0,0),1,1, color=CHILI, label='TP.HCM')]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc='lower right')
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig

fig5 = district_fig()
fig5.savefig('/home/claude/viz/05_district_bar.svg', format='svg', bbox_inches='tight', facecolor=PAPER)
fig5.savefig('/home/claude/viz/05_district_bar.png', format='png', dpi=170, bbox_inches='tight', facecolor=PAPER)
plt.close(fig5)
print('fig5 done')

# =====================================================================
# FIGURE 6: Rating vs review_count scatter
# =====================================================================
def scatter_fig(figsize=(12,7)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    for name, color in [('Hà Nội', TURM), ('TP.HCM', CHILI)]:
        sub = df[(df['city_name']==name) & df['rating'].notna() & df['review_count'].notna()]
        ax.scatter(sub['review_count'], sub['rating'], s=10, alpha=0.35, color=color, label=name, linewidths=0)
    ax.set_xscale('symlog')
    ax.set_xlabel('Số lượt đánh giá (thang log)', fontsize=10)
    ax.set_ylabel('Rating', fontsize=10)
    ax.set_title('Tương quan Rating và Số lượt đánh giá', fontsize=16, fontweight='bold', color=CHILID, pad=14)
    style_ax(ax, grid_axis='both')
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    return fig

fig6 = scatter_fig()
fig6.savefig('/home/claude/viz/06_rating_review_scatter.svg', format='svg', bbox_inches='tight', facecolor=PAPER)
fig6.savefig('/home/claude/viz/06_rating_review_scatter.png', format='png', dpi=170, bbox_inches='tight', facecolor=PAPER)
plt.close(fig6)
print('fig6 done')

# =====================================================================
# COVER PAGE
# =====================================================================
def cover_fig(figsize=(14,9)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(INK)
    ax.axis('off')
    ax.text(0.5, 0.62, 'beFood', fontsize=64, fontweight='bold', color=TURM,
            ha='center', va='center', family='DejaVu Sans')
    ax.text(0.5, 0.50, 'Báo cáo trực quan hoá dữ liệu quán ăn', fontsize=22, color=PAPER,
            ha='center', va='center', family='DejaVu Sans')
    ax.text(0.5, 0.42, 'Hà Nội  ·  TP. Hồ Chí Minh', fontsize=15, color='#d8c6a8',
            ha='center', va='center')
    n_total = len(df)
    n_hn = (df['city']==190).sum()
    n_hcm = (df['city']==189).sum()
    stats = f"{n_total:,} quán ăn   ·   {n_hn:,} tại Hà Nội   ·   {n_hcm:,} tại TP.HCM   ·   17 ngành hàng   ·   43 khu vực"
    ax.text(0.5, 0.30, stats, fontsize=11.5, color=TURM, ha='center', va='center')
    ax.plot([0.15,0.85],[0.25,0.25], color=CHILI, linewidth=2)
    ax.text(0.5, 0.06, 'Nguồn dữ liệu: befood_restaurant_data.json', fontsize=9, color='#8a6b3b', ha='center')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    return fig

figc = cover_fig()
figc.savefig('/home/claude/viz/00_cover.png', dpi=170, bbox_inches='tight', facecolor=INK)
plt.close(figc)
print('cover done')

# =====================================================================
# Combine all into one multi-page PDF report
# =====================================================================
with PdfPages('/home/claude/viz/befood_report.pdf') as pdf:
    for fn, kwargs in [
        (cover_fig, dict(facecolor=INK)),
        (geo_fig, dict(facecolor=PAPER)),
        (density_fig, dict(facecolor=PAPER)),
        (category_bar_fig, dict(facecolor=PAPER)),
        (district_fig, dict(facecolor=PAPER)),
        (rating_fig, dict(facecolor=PAPER)),
        (scatter_fig, dict(facecolor=PAPER)),
    ]:
        f = fn()
        pdf.savefig(f, **kwargs)
        plt.close(f)

print('PDF report done')
