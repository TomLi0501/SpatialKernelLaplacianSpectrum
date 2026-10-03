import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata

from unifromly_distributed_particles import Space


def _draw_nodal_domain(ax, positions, adj, eigenvector, eigenvalue, eigenvalue_index,
                       L, N, sigma, draw_edges=True, edge_threshold=1e-2,
                       node_size=40, background=True, grid_res=400):
    """
    Draw a single nodal-domain panel on ``ax``.

    Nodes are coloured only by the sign of the eigenvector component
    (red = positive domain, blue = negative domain).  The zero contour of the
    interpolated eigenvector field is drawn as a bold black nodal line.
    """
    # --- interpolate the eigenvector onto a fine grid ---
    # The eigenvector is only defined on the particles, so we interpolate it onto
    # a regular grid.  The zero contour of this field is the nodal line that
    # separates the nodal domains (positive vs negative regions).
    gx = np.linspace(0, L, grid_res)
    gy = np.linspace(0, L, grid_res)
    GX, GY = np.meshgrid(gx, gy)

    # Periodic interpolation: replicate the particles in the 8 neighbouring
    # images so the interpolated field is continuous across the boundary.
    pts = positions
    vals = eigenvector
    all_pts = []
    all_vals = []
    for sx in (-L, 0.0, L):
        for sy in (-L, 0.0, L):
            all_pts.append(pts + np.array([sx, sy]))
            all_vals.append(vals)
    all_pts = np.vstack(all_pts)
    all_vals = np.concatenate(all_vals)

    field = griddata(all_pts, all_vals, (GX, GY), method='linear')
    # Fill any NaNs (outside the convex hull) with nearest-neighbour values
    if np.isnan(field).any():
        field_nn = griddata(all_pts, all_vals, (GX, GY), method='nearest')
        field = np.where(np.isnan(field), field_nn, field)

    # --- optional background heatmap of the interpolated field ---
    if background:
        vlim = np.percentile(np.abs(field), 98)
        if vlim <= 0:
            vlim = np.max(np.abs(field)) or 1.0
        field = np.clip(field, -vlim, vlim)
        ax.pcolormesh(GX, GY, field, cmap='coolwarm', vmin=-vlim, vmax=vlim,
                      shading='auto', zorder=0)

    # --- nodal line (zero contour) drawn as a bold black line ---
    ax.contour(GX, GY, field, levels=[0.0], colors='black',
               linewidths=2.0, zorder=3)

    # --- edges (drawn first so nodes sit on top) ---
    if draw_edges:
        segments = []
        weights = []
        for i in range(N):
            for j in range(i + 1, N):
                w = adj[i, j]
                if abs(w) > edge_threshold:
                    p1 = positions[i]
                    p2 = positions[j]
                    # Minimum-image displacement (same convention as Space.distance)
                    d = p2 - p1
                    d = d - L * np.round(d / L)
                    # Draw the wrapped edge, plus copies shifted by +/- L so the
                    # segment is visible on both sides of the boundary.
                    for shift in ([0.0, 0.0], [L, 0.0], [-L, 0.0],
                                  [0.0, L], [0.0, -L]):
                        segments.append([p1, p1 + d + np.array(shift)])
                    weights.append(abs(w))

        if segments:
            weights = np.array(weights)
            # Normalise edge alpha/width by weight for a cleaner look
            if weights.max() > weights.min():
                norm_w = (weights - weights.min()) / (weights.max() - weights.min())
            else:
                norm_w = np.ones_like(weights)

            # Repeat the per-edge weights for the 5 shifted copies of each edge
            norm_w = np.repeat(norm_w, 5)

            lc = LineCollection(
                segments,
                colors='grey',
                linewidths=0.2 + 1.3 * norm_w,
                alpha=0.15 + 0.5 * norm_w,
                zorder=1,
            )
            ax.add_collection(lc)

    # --- nodes coloured by the sign of the eigenvector ---
    # Only the sign is marked: red = positive nodal domain, blue = negative
    # nodal domain.  No colour bar is drawn.
    sign_cmap = ListedColormap(['#1f4fd8', '#d81f1f'])
    node_colors = (eigenvector > 0).astype(int)
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=node_colors,
        cmap=sign_cmap,
        vmin=-0.5,
        vmax=1.5,
        s=node_size,
        edgecolors='black',
        linewidths=0.3,
        zorder=4,
    )

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(
        f'Eigenvalue #{eigenvalue_index} '
        f'($\\lambda_{{{eigenvalue_index}}} = {eigenvalue:.3f}$)'
    )


def generate_network(N=500, L=1.0, sigma=0.05, eigenvalue_indices=(5, 50, 300),
                     combination=False, diagonal=True, seed=None,
                     draw_edges=True, edge_threshold=1e-2,
                     node_size=40, save_path=None,
                     background=True, grid_res=400):
    """
    Generate a random geometric network and draw the nodal domains of one or
    more eigenvalues side by side.

    Each node is coloured only by the sign of the corresponding eigenvector
    component (red = positive domain, blue = negative domain), and the zero
    contour of the interpolated eigenvector field is drawn as a bold black
    nodal line.

    Parameters
    ----------
    N : int
        Number of particles (nodes) in the network.
    L : float
        Side length of the square box the particles live in.
    sigma : float
        Kernel width of the Gaussian interaction.
    eigenvalue_indices : int or sequence of int
        1-based index (or indices) of the eigenvalue(s) whose eigenvector(s)
        are visualised.  Defaults to ``(5, 50, 300)``, drawn side by side.
    combination : bool
        If True, compute all O(N^2) pairs; otherwise use the box optimisation.
    diagonal : bool
        If True, set the diagonal to minus the row sums (graph Laplacian).
    seed : int or None
        Optional random seed for reproducibility.
    draw_edges : bool
        Whether to draw the edges of the network.
    edge_threshold : float
        Only draw edges whose (absolute) weight is above this threshold.
    node_size : float
        Marker size for the nodes.
    save_path : str or None
        If given, the figure is saved to this path.
    background : bool
        Whether to draw the interpolated eigenvector field as a background heatmap.
    grid_res : int
        Resolution of the interpolation grid.

    Returns
    -------
    fig, axes : matplotlib figure and axes array
    space : the Space object (contains particles and adjacency matrix)
    eigvals, eigvecs : eigenvalues and eigenvectors
    """
    if seed is not None:
        np.random.seed(seed)

    # Allow a single integer to be passed
    if np.isscalar(eigenvalue_indices):
        eigenvalue_indices = (int(eigenvalue_indices),)
    eigenvalue_indices = list(eigenvalue_indices)

    # ------------------------------------------------------------------
    # 1. Build the network (particles + adjacency matrix)
    # ------------------------------------------------------------------
    space = Space(N, L, sigma)
    space.place_particles()
    space.adjacency(diagonal=diagonal, combination=combination)
    adj = space.adj_matrix

    # ------------------------------------------------------------------
    # 2. Diagonalise
    # ------------------------------------------------------------------
    eigvals, eigvecs = np.linalg.eigh(adj)

    positions = np.array(space._particles)

    # ------------------------------------------------------------------
    # 3. Draw one panel per requested eigenvalue, side by side
    # ------------------------------------------------------------------
    n_panels = len(eigenvalue_indices)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    for ax, eigenvalue_index in zip(axes, eigenvalue_indices):
        idx = eigenvalue_index - 1
        if idx < 0 or idx >= N:
            raise ValueError(
                f"eigenvalue_index must be between 1 and {N}, got {eigenvalue_index}"
            )
        eigenvalue = eigvals[idx]
        eigenvector = eigvecs[:, idx]

        _draw_nodal_domain(
            ax, positions, adj, eigenvector, eigenvalue, eigenvalue_index,
            L, N, sigma, draw_edges=draw_edges,
            edge_threshold=edge_threshold, node_size=node_size,
            background=background, grid_res=grid_res,
        )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')

    plt.show()

    return fig, axes, space, eigvals, eigvecs


if __name__ == "__main__":
    generate_network(
        N=500,
        L=1.0,
        sigma=0.05,
        background=True,
        draw_edges=True,
        edge_threshold=1e-3,
        eigenvalue_indices=(5, 50, 300),
        combination=False,
        diagonal=True,
        seed=42,
    )
