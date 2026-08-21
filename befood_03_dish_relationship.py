import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

df = pd.read_pickle('/home/claude/viz/df.pkl')

# ---- palette (matches the map / previous charts) ----
INK, PAPER, PAPER2 = '#241608', '#fbf2e3', '#f3e4c9'
CHILI, CHILID, TURM, LEAF, LINE = '#d5451b', '#a8330f', '#f2a007', '#3e7a4d', '#e2c99b'

CAT_COLORS = {
    "Cà Phê/Trà":"#6f4e2c", "Cơm/Cơm tấm":"#d5451b", "Bún/Phở/Mỳ/Cháo":"#a8330f",
    "Sinh tố/Nước ép/Sữa":"#3e7a4d", "Thức ăn nhanh":"#f2a007", "Ăn vặt":"#c2185b",
    "Ẩm thực quốc tế":"#3f51b5", "Trà sữa":"#8e5fb0", "Món truyền thống/Đặc sản":"#b8860b",
    "Bánh Mì/Xôi":"#e07a1f", "Tráng miệng":"#e91e63", "Lẩu & Nướng":"#c0392b",
    "Pizza":"#d84315", "Ốc/Cá/Hải sản":"#0288a8", "Món chay":"#43913e",
    "Mart":"#607d8b", "Món Dinh dưỡng/Bổ dưỡng":"#009688"
}

plt.rcParams['font.family'] = 'DejaVu Sans'
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
# 1. Extract dish keywords from restaurant names
# =====================================================================
# (regex pattern -> display label). Order matters: more specific phrases first.
DISH_PATTERNS = [
    (r'cơm\s*tấm', 'Cơm Tấm'),
    (r'cơm\s*gà', 'Cơm Gà'),
    (r'cơm\s*chiên', 'Cơm Chiên'),
    (r'cơm\s*chay', 'Cơm Chay'),
    (r'cơm\s*văn\s*phòng', 'Cơm Văn Phòng'),
    (r'cơm\s*sườn', 'Cơm Sườn'),
    (r'cơm\s*niêu', 'Cơm Niêu'),
    (r'cơm\s*bụi|cơm\s*nhà|cơm\s*phần', 'Cơm Nhà/Bụi'),
    (r'\bcơm\b', 'Cơm (khác)'),
    (r'bún\s*bò', 'Bún Bò'),
    (r'bún\s*riêu', 'Bún Riêu'),
    (r'bún\s*chả', 'Bún Chả'),
    (r'bún\s*thịt\s*nướng', 'Bún Thịt Nướng'),
    (r'bún\s*đậu', 'Bún Đậu'),
    (r'bún\s*mắm', 'Bún Mắm'),
    (r'bún\s*cá', 'Bún Cá'),
    (r'\bbún\b', 'Bún (khác)'),
    (r'phở', 'Phở'),
    (r'hủ\s*tiếu', 'Hủ Tiếu'),
    (r'mì\s*quảng', 'Mì Quảng'),
    (r'mì\s*cay', 'Mì Cay'),
    (r'mì\s*ý|pasta', 'Mì Ý/Pasta'),
    (r'mì\s*trộn', 'Mì Trộn'),
    (r'\bmì\b|\bmỳ\b', 'Mì (khác)'),
    (r'cháo', 'Cháo'),
    (r'sủi\s*cảo', 'Sủi Cảo'),
    (r'bánh\s*mì', 'Bánh Mì'),
    (r'\bxôi\b', 'Xôi'),
    (r'bánh\s*xèo', 'Bánh Xèo'),
    (r'bánh\s*canh', 'Bánh Canh'),
    (r'bánh\s*cuốn', 'Bánh Cuốn'),
    (r'bánh\s*tráng', 'Bánh Tráng Trộn'),
    (r'bánh\s*ướt', 'Bánh Ướt'),
    (r'bánh\s*su\s*kem|bánh\s*ngọt|bánh\s*kem', 'Bánh Ngọt/Kem'),
    (r'trà\s*sữa', 'Trà Sữa'),
    (r'cà\s*phê|cafe|coffee', 'Cà Phê'),
    (r'sinh\s*tố', 'Sinh Tố'),
    (r'nước\s*ép', 'Nước Ép'),
    (r'trà\s*chanh', 'Trà Chanh'),
    (r'trà\s*đào', 'Trà Đào'),
    (r'nước\s*mía', 'Nước Mía'),
    (r'nước\s*sâm|nước\s*mát', 'Nước Sâm/Mát'),
    (r'\btrà\b', 'Trà (khác)'),
    (r'gà\s*rán|fried\s*chicken', 'Gà Rán'),
    (r'pizza', 'Pizza'),
    (r'hamburger|burger', 'Hamburger'),
    (r'sushi', 'Sushi'),
    (r'ramen', 'Ramen'),
    (r'taco', 'Tacos'),
    (r'kimbap', 'Kimbap'),
    (r'tokbokki|tteokbokki|tokpokki', 'Tokbokki'),
    (r'lẩu', 'Lẩu'),
    (r'nướng', 'Nướng'),
    (r'ốc\b', 'Ốc'),
    (r'hải\s*sản', 'Hải Sản'),
    (r'\bcá\b', 'Cá'),
    (r'chè\b', 'Chè'),
    (r'\bkem\b', 'Kem'),
    (r'trái\s*cây', 'Trái Cây'),
    (r'salad', 'Salad'),
    (r'gỏi\s*cuốn|nem', 'Gỏi Cuốn/Nem'),
    (r'chả\s*giò', 'Chả Giò'),
    (r'vịt', 'Vịt'),
    (r'heo|thịt\s*nướng', 'Thịt Heo/Nướng'),
    (r'bò\b', 'Bò'),
]

def extract_dishes(name):
    if not isinstance(name, str):
        return []
    n = name.lower()
    found = []
    matched_spans = []
    for pattern, label in DISH_PATTERNS:
        for m in re.finditer(pattern, n):
            span = m.span()
            # avoid double counting overlapping spans for same underlying word
            if any(not (span[1] <= s[0] or span[0] >= s[1]) for s in matched_spans):
                continue
            found.append(label)
            matched_spans.append(span)
    return list(dict.fromkeys(found))  # unique, keep order

df['dishes'] = df['name'].apply(extract_dishes)
df['n_dishes'] = df['dishes'].apply(len)

# Explode into long form: one row per (restaurant, dish)
rows = []
for _, r in df.iterrows():
    for d in r['dishes']:
        rows.append({'restaurant_id': r['restaurant_id'], 'name': r['name'], 'dish': d,
                     'merchant_category': r['merchant_category'], 'city_name': r['city_name'],
                     'rating': r['rating']})
dish_df = pd.DataFrame(rows)
print('Total (restaurant, dish) pairs:', len(dish_df))
print('Restaurants with >=1 dish match:', (df['n_dishes']>0).sum(), '/', len(df))
print(dish_df['dish'].value_counts().head(30))

dish_df.to_pickle('/home/claude/viz/dish_df.pkl')
df.to_pickle('/home/claude/viz/df.pkl')

# =====================================================================
# FIGURE 7: Top dish keywords overall (bar, colored by dominant category)
# =====================================================================
def top_dishes_fig(figsize=(12,9), topn=25):
    counts = dish_df['dish'].value_counts().head(topn)
    # dominant category per dish (most frequent category for that dish)
    dom_cat = dish_df[dish_df['dish'].isin(counts.index)].groupby('dish')['merchant_category'] \
                .agg(lambda s: s.value_counts().idxmax())
    colors = [CAT_COLORS.get(dom_cat[d], '#999') for d in counts.index[::-1]]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors)
    ax.set_xlabel('Số lượng quán nhắc đến món này trong tên', fontsize=10)
    ax.set_title('Top 25 món ăn xuất hiện nhiều nhất trong tên quán', fontsize=16,
                  fontweight='bold', color=CHILID, pad=14)
    style_ax(ax, grid_axis='x')
    for i, v in enumerate(counts.values[::-1]):
        ax.text(v + max(counts.values)*0.01, i, str(v), va='center', fontsize=8.5, color=INK)
    fig.tight_layout()
    return fig

fig7 = top_dishes_fig()
fig7.savefig('/home/claude/viz/07_top_dishes.svg', format='svg', bbox_inches='tight', facecolor=PAPER)
fig7.savefig('/home/claude/viz/07_top_dishes.png', format='png', dpi=170, bbox_inches='tight', facecolor=PAPER)
plt.close(fig7)
print('fig7 done')

# =====================================================================
# FIGURE 8: Heatmap — Ngành hàng x Món ăn (top dishes)
# =====================================================================
def heatmap_fig(figsize=(14,9), topn=22):
    top_dishes = dish_df['dish'].value_counts().head(topn).index.tolist()
    mat = pd.crosstab(dish_df['merchant_category'], dish_df['dish'])
    mat = mat.reindex(columns=top_dishes).fillna(0)
    # order rows by total
    mat = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
    mat = mat[mat.sum(axis=1) > 0]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=55, ha='right', fontsize=9)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=9.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat.values[i,j])
            if v > 0:
                txt_color = 'white' if v > mat.values.max()*0.55 else INK
                ax.text(j, i, str(v), ha='center', va='center', fontsize=6.8, color=txt_color)
    ax.set_title('Ma trận quan hệ: Ngành hàng (loại quán) × Món ăn phổ biến', fontsize=16,
                  fontweight='bold', color=CHILID, pad=14)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('Số lượng quán', fontsize=9)
    fig.tight_layout()
    return fig

fig8 = heatmap_fig()
fig8.savefig('/home/claude/viz/08_category_dish_heatmap.svg', format='svg', bbox_inches='tight', facecolor=PAPER)
fig8.savefig('/home/claude/viz/08_category_dish_heatmap.png', format='png', dpi=170, bbox_inches='tight', facecolor=PAPER)
plt.close(fig8)
print('fig8 done')

# =====================================================================
# FIGURE 9: Bipartite network — Ngành hàng <-> Món ăn (top N edges)
# =====================================================================
def network_fig(figsize=(15,15), top_dishes_n=28, min_edge=15):
    top_dishes = dish_df['dish'].value_counts().head(top_dishes_n).index.tolist()
    sub = dish_df[dish_df['dish'].isin(top_dishes)]
    edge_counts = sub.groupby(['merchant_category','dish']).size().reset_index(name='count')
    edge_counts = edge_counts[edge_counts['count'] >= min_edge]

    cats = [c for c in CAT_COLORS if c in edge_counts['merchant_category'].unique()]
    dishes = [d for d in top_dishes if d in edge_counts['dish'].unique()]

    n_cat, n_dish = len(cats), len(dishes)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(PAPER)

    # positions: categories on left column, dishes on right column
    cat_y = np.linspace(0.95, 0.05, n_cat)
    dish_y = np.linspace(0.98, 0.02, n_dish)
    cat_pos = {c: (0.22, y) for c, y in zip(cats, cat_y)}
    dish_pos = {d: (0.78, y) for d, y in zip(dishes, dish_y)}

    max_count = edge_counts['count'].max()
    for _, row in edge_counts.iterrows():
        c, d, cnt = row['merchant_category'], row['dish'], row['count']
        if c not in cat_pos or d not in dish_pos: continue
        x1,y1 = cat_pos[c]; x2,y2 = dish_pos[d]
        lw = 0.4 + 4.2 * (cnt / max_count)
        ax.plot([x1,x2],[y1,y2], color=CAT_COLORS.get(c,'#999'), alpha=0.45, linewidth=lw, zorder=1,
                solid_capstyle='round')

    # draw category nodes
    for c,(x,y) in cat_pos.items():
        deg = edge_counts[edge_counts['merchant_category']==c]['count'].sum()
        size = 400 + deg*1.1
        ax.scatter([x],[y], s=size, color=CAT_COLORS.get(c,'#999'), edgecolor='white',
                   linewidth=1.6, zorder=3)
        ax.text(x-0.025, y, c, ha='right', va='center', fontsize=10.5, fontweight='bold', color=INK, zorder=4)

    # draw dish nodes
    for d,(x,y) in dish_pos.items():
        deg = edge_counts[edge_counts['dish']==d]['count'].sum()
        size = 120 + deg*1.6
        ax.scatter([x],[y], s=size, color=PAPER, edgecolor=CHILID, linewidth=1.4, zorder=3)
        ax.text(x+0.025, y, d, ha='left', va='center', fontsize=9.5, color=INK, zorder=4)

    ax.set_xlim(-0.05, 1.25)
    ax.set_ylim(-0.02, 1.02)
    ax.axis('off')
    ax.set_title('Mạng lưới quan hệ: Ngành hàng quán ăn ↔ Món ăn phổ biến', fontsize=18,
                  fontweight='bold', color=CHILID, pad=6)
    ax.text(0.5, -0.015, 'Độ dày đường nối = số lượng quán · Kích thước nút = tổng liên kết',
            ha='center', fontsize=9.5, color='#8a6b3b', transform=ax.transAxes)
    fig.tight_layout()
    return fig

fig9 = network_fig()
fig9.savefig('/home/claude/viz/09_category_dish_network.svg', format='svg', bbox_inches='tight', facecolor=PAPER)
fig9.savefig('/home/claude/viz/09_category_dish_network.png', format='png', dpi=170, bbox_inches='tight', facecolor=PAPER)
plt.close(fig9)
print('fig9 done')

# =====================================================================
# FIGURE 10: Rating trung bình theo món ăn (top dishes by count)
# =====================================================================
def rating_by_dish_fig(figsize=(12,9), topn=20):
    top_dishes = dish_df['dish'].value_counts().head(topn).index.tolist()
    sub = dish_df[dish_df['dish'].isin(top_dishes) & dish_df['rating'].notna()]
    stats = sub.groupby('dish')['rating'].agg(['mean','count']).reindex(top_dishes)
    stats = stats.sort_values('mean')

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn((stats['mean'] - stats['mean'].min()) / (stats['mean'].max() - stats['mean'].min() + 1e-9))
    bars = ax.barh(stats.index, stats['mean'], color=colors)
    ax.set_xlim(stats['mean'].min()-0.15, 5.0)
    ax.set_xlabel('Rating trung bình', fontsize=10)
    ax.set_title(f'Rating trung bình theo {topn} món ăn phổ biến nhất', fontsize=16,
                 fontweight='bold', color=CHILID, pad=14)
    style_ax(ax, grid_axis='x')
    for i,(m,c) in enumerate(zip(stats['mean'], stats['count'])):
        ax.text(m+0.01, i, f"{m:.2f}  (n={int(c)})", va='center', fontsize=8.3, color=INK)
    fig.tight_layout()
    return fig

fig10 = rating_by_dish_fig()
fig10.savefig('/home/claude/viz/10_rating_by_dish.svg', format='svg', bbox_inches='tight', facecolor=PAPER)
fig10.savefig('/home/claude/viz/10_rating_by_dish.png', format='png', dpi=170, bbox_inches='tight', facecolor=PAPER)
plt.close(fig10)
print('fig10 done')

# =====================================================================
# Append these 4 new figures into the PDF report
# =====================================================================
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('/home/claude/viz/befood_dish_report.pdf') as pdf:
    for fn in [top_dishes_fig, heatmap_fig, network_fig, rating_by_dish_fig]:
        f = fn()
        pdf.savefig(f, facecolor=PAPER)
        plt.close(f)
print('dish PDF done')
