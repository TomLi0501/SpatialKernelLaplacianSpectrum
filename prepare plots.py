import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'


def panel_grid_periodic(ax):



    L = 1.0
    # grid size (side length of one cell)
    g = 0.25
    n = int(L / g)

    # ---- draw the periodic tiling (ghost boxes) faintly ----
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            if ix == 0 and iy == 0:
                continue
            ax.add_patch(Rectangle((ix * L, iy * L), L, L,
                                   fill=False, edgecolor='0.85', lw=0.8, ls=':'))

    # ---- main box ----
    ax.add_patch(Rectangle((0, 0), L, L, fill=False, edgecolor='k', lw=2))

    # ---- draw the grid inside the main box ----
    for i in range(n + 1):
        ax.plot([i * g, i * g], [0, L], color='0.6', lw=1)
        ax.plot([0, L], [i * g, i * g], color='0.6', lw=1)

    # ---- reference particle P in the corner cell (0,0) ----
    P = np.array([0.125, 0.125])
    cell = (0, 0)

    # ---- highlight the 3x3 block of neighbouring cells WITH periodic wrap ----
    # The 3x3 block around the corner cell wraps around to the far edges,
    # so the "neighbouring" cells include those on the opposite side of the box.
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ci = (cell[0] + di) % n
            cj = (cell[1] + dj) % n
            xi = ci * g
            yj = cj * g
            ax.add_patch(Rectangle((xi, yj), g, g, fill=True,
                                   color='tab:blue', alpha=0.15, lw=0))
    # redraw grid lines on top of the highlight
    for i in range(n + 1):
        ax.plot([i * g, i * g], [0, L], color='0.6', lw=1)
        ax.plot([0, L], [i * g, i * g], color='0.6', lw=1)

    # ---- particles ----
    rng = np.random.default_rng(7)
    pts = []
    for _ in range(26):
        x = rng.uniform(0, L)
        y = rng.uniform(0, L)
        pts.append(np.array([x, y]))
    pts = np.array(pts)

    # reference particle P
    ax.scatter(*P, s=110, color='k', zorder=6, marker='*')
    ax.annotate('P', P + np.array([0.01, 0.01]), fontsize=13, fontweight='bold')

    # classify each particle: inside the wrapped 3x3 block (connected) or not
    for q in pts:
        cx = int(q[0] // g)
        cy = int(q[1] // g)
        dcell = min(abs(cx - cell[0]), n - abs(cx - cell[0]))
        dcell2 = min(abs(cy - cell[1]), n - abs(cy - cell[1]))
        inside = (dcell <= 1) and (dcell2 <= 1)
        if inside:
            ax.scatter(*q, s=45, color='tab:blue', zorder=5)
        else:
            ax.scatter(*q, s=45, color='tab:red', zorder=5)

    # ---- legend / explanation ----

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue',
               markersize=8, label='neighbouring cell (weighted)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red',
               markersize=8, label='outside block (weight = 0)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='k',
               markersize=12, label='reference particle P'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)


    ax.set_xlim(-1.05 * L, 2.05 * L)
    ax.set_ylim(-0.35 * L, 1.35 * L)
    ax.set_aspect('equal')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['0', 'L'])
    ax.set_yticklabels(['0', 'L'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def main():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    panel_grid_periodic(ax)
    fig.tight_layout()
    fig.savefig('figures/grid_periodic_illustration.png', dpi=200,
                bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
