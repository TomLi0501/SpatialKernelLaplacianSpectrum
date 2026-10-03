from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
# 锦瑟年华谁与度？月桥花苑，琐窗朱户，只有春知处。


class Space:
    def __init__(self,N, L, sigma):
        self.N = N
        self.L = L
        self.sigma = sigma
        self.box_length = 0
        self._particles = []
        self._particles_index = []
        self.adj_matrix = np.zeros((self.N, self.N))
        self.p_b_mapping = []
        self.box_point_mapping = {}
        self.all_pairs = set()
        self.box_centers = []

    def place_particles(self): # Uniform selected
        self._particles = []
        self._particles_index = []
        for p_index in range(self.N):
            x = np.random.random() * self.L
            y = np.random.random() * self.L
            self._particles.append([x,y])
            self._particles_index.append(p_index)
        plt.show()

    def distance(self, pos1: list, pos2:list):
        dx = np.abs(pos2[0] - pos1[0])
        dy = np.abs(pos2[1] - pos1[1])
        if dx > self.L/2: # All distance smaller than L/2
            dx = self.L - dx
        if dy > self.L/2:
            dy = self.L - dy
        return np.sqrt(dx**2 + dy**2)

    def kernel(self, pos1: list, pos2: list):
        dr = self.distance(pos1, pos2)
        # linear Kernel
        #return dr
        # Gaussian kernel
        return 1 * np.exp(-dr**2/(2*self.sigma**2))

    def adjacency(self,diagonal=False, weight_on_error = 1.0, combination = False):
        self.adj_matrix = np.zeros((self.N, self.N)) # clean it first
        if combination:
            selected_pairs = combinations(self._particles_index, 2) # O(N^2)
        else:
            selected_pairs = self.split_box_get_all_pairs(weight_on_error=weight_on_error)
        # if i != j
        for particle_pairs in tqdm(selected_pairs, 'computing adj_matrix', disable=False):
            id1 = particle_pairs[0]
            id2 = particle_pairs[1]
            ele = - self.kernel(self._particles[id1], self._particles[id2])
            # symmetric matrix
            self.adj_matrix[id1][id2] = ele
            self.adj_matrix[id2][id1] = ele
        # diagonal
        if diagonal:
            row_sums = np.sum(self.adj_matrix, axis=1)
            np.fill_diagonal(self.adj_matrix, -row_sums)
        else:
            pass
        return self.adj_matrix

    def eigenvalues_spectrum(self, weight_on_error = 1.0,  combination = False, diagonal=True):
        self.place_particles()
        self.adjacency(diagonal=diagonal, weight_on_error = weight_on_error, combination = combination)
        spectrum = np.linalg.eigh(self.adj_matrix)[0] # the first is eigen value
        return spectrum

    def split_box(self, weight_on_error = 1.0):
        N = self.N
        L = self.L
        sigma = self.sigma
        uncer = UniformUncertainty(N, L, sigma)
        #self.box_length, __, ___ = uncer.optimize_sub_box(num_test = 100, graph=True, weight_on_error = weight_on_error)
        # Number of boxes per dimension (ensures full coverage of [0, L))
        self.box_length = 4 * self.sigma
        box_nums = int(np.ceil(L / self.box_length))
        d = self.box_length / 2


        # Build grid of box centers
        self.box_centers = []
        for i in range(box_nums):
            for q in range(box_nums):
                self.box_centers.append([i * self.box_length + d, q * self.box_length + d])

        # Build particle-to-box mapping: for each particle, store which box(es) it belongs to
        # p_b_mapping[p_id] = list of box centers that contain particle p_id
        self.p_b_mapping = [[] for _ in range(self.N)]
        self.box_point_mapping = {}
        for p_id, p in enumerate(self._particles):
            for box in self.box_centers:
                if np.abs(p[0] - box[0]) <= self.box_length/2 and np.abs(p[1] - box[1]) <= self.box_length/2:
                    self.p_b_mapping[p_id].append(box)
                    box = tuple(box)
                    if box not in self.box_point_mapping:
                        self.box_point_mapping[box] = []
                    self.box_point_mapping[box].append(p_id)

    def box_box_mapping(self, box_center):
        """
        Given a box center, find all 9 neighboring box centers (including itself)
        in the 3x3 block around it, with periodic boundary conditions.
        """
        box_nums = int(np.ceil(self.L / self.box_length))

        # Find the grid indices (i, q) of the given box_center
        # Box centers are at positions: [d, d+box_length, d+2*box_length, ...]
        # where d = box_length/2, so index = (pos - d) / box_length
        d = self.box_length / 2
        i = round((box_center[0] - d) / self.box_length)
        q = round((box_center[1] - d) / self.box_length)

        neighbouring_box_centers = []
        for di in [-1, 0, 1]:
            for dq in [-1, 0, 1]:
                ni = (i + di) % box_nums
                nq = (q + dq) % box_nums
                neighbouring_box_centers.append([ni * self.box_length + d, nq * self.box_length + d])

        return neighbouring_box_centers

    def get_neighbouring_points(self, point_index):
        """
        Find all particles that are in the same or neighbouring boxes as particle point_index.
        Uses box_box_mapping to find adjacent boxes, then checks which particles are in those boxes.
        """
        neighbouring_points = set()
        # Get all boxes that contain this particle
        current_boxes = self.p_b_mapping[point_index]
        # For each box this particle is in, find neighbouring boxes
        neighbouring_boxes = set()
        for box in current_boxes:
            for nb in self.box_box_mapping(box):
                neighbouring_boxes.add(tuple(nb))

        # Find all particles in those neighbouring boxes
        for box in neighbouring_boxes:
            box = tuple(box)
            neighbouring_points.update(self.box_point_mapping.get(box, []))

        # Add pairs, avoiding duplicates and self-pairs
        for end in neighbouring_points:
            if end == point_index:
                continue
            if (point_index, end) not in self.all_pairs and (end, point_index) not in self.all_pairs:
                self.all_pairs.add((point_index, end))

    def split_box_get_all_pairs(self, weight_on_error = 1.0):
        self.all_pairs = set()
        self.split_box(weight_on_error = weight_on_error)
        for i in tqdm(self._particles_index,'preparing pairs', disable=True):
            self.get_neighbouring_points(i)
        return list(self.all_pairs)

class UniformUncertainty:
    def __init__(self, N, L, sigma):
        self.N = N
        self.L = L
        self.sigma = sigma

    def density(self, r): # Point density: number/area
        return self.N/self.L**2

    def point_cdf(self, r):
        # Ensure input is a NumPy array
        r = np.asarray(r)
        L = self.L
        N = self.N

        # Initialise output array
        result = np.empty_like(r, dtype=float)

        # Negative
        mask_neg = r < 0
        result[mask_neg] = 0.0

        # r <= L/2
        mask_le = r <= L / 2
        r_le = r[mask_le]
        if r_le.size > 0:
            result[mask_le] = self.density(r_le) * np.pi * r_le ** 2

        # r > L/2
        mask_gt = r > L / 2
        r_gt = r[mask_gt]
        if r_gt.size > 0:
            # The terms are computed only on the valid subset (r_gt > L/2)
            term_pi = np.pi * r_gt ** 2
            overlap = 4 * (
                    np.arccos(L / (2 * r_gt)) * r_gt ** 2
                    - (L / 2) * np.sqrt(r_gt ** 2 - (L / 2) ** 2)
            )
            result[mask_gt] = self.density(r_gt) * (term_pi - overlap)

        mask3 = r > L / np.sqrt(2)
        r3 = r[mask3]
        if r3.size > 0:
            result[mask3] = N

        return result/self.N

    def r_a_relation(self, a):
        return np.sqrt(- 2 * self.sigma**2 * np.log(a * np.sqrt(2 * np.pi) * self.sigma**2)) # 2 s^2 ln(a sqr(2pi) s^2)

    def ele_cdf(self, a):
        return self.point_cdf(self.r_a_relation(a)) # substitute to get pdf of elements of adj matrix

    def ele_pdf(self, a):
        return - np.gradient(self.ele_cdf(a), a)

    def absolute_error(self, c, num_column = 100):

        x = np.linspace(0, c, num_column)
        return np.trapezoid(self.ele_pdf(x) * x * (self.N-1), x)

    def a_r_relation(self, r):
        return 1 / (np.sqrt(2 * np.pi) * self.sigma ** 2) * np.exp(-r ** 2 / (2 * self.sigma ** 2))

    def optimize_sub_box(self, num_test = 100, graph = True, weight_on_error = 1.0):
        d = np.linspace(0, self.L, num_test)
        errd = []
        for cutoff_r in d:
            errd.append(self.absolute_error(self.a_r_relation(3*cutoff_r/np.sqrt(np.pi))))
        expected_computations = (self.N**2 / self.L ** 2 * 9 * d ** 2)/2
        expected_computations = expected_computations#/np.max(expected_computations)
        errd = np.array(errd)
        distance = np.abs(errd/np.max(errd) * weight_on_error)  + np.abs(expected_computations/np.max(expected_computations))
        min_index = np.argmin(distance)
        related_subbox_length = d[min_index]
        if graph:
            print('optimised box length: ' + str(related_subbox_length))
            print('Estimated computations: ' + str(expected_computations[min_index]))
            print('Estimated uncertainty(absolute):  ' + str(errd[min_index]))
            plt.scatter(errd, expected_computations)
            plt.xlabel('error')
            plt.ylabel('computations')
            plt.show()
        return related_subbox_length, expected_computations[min_index] * self.N , errd[min_index]


def get_ele_pdf():
    N = 500
    L = 10
    sigma = 2
    space = Space(N, L, sigma)

    uncer = UniformUncertainty(N, L, sigma)
    space.place_particles()
    adj = space.adjacency(combination = True)

    ranga = np.linspace(0, np.abs(np.min(adj)), 500)
    k = 2 * np.pi * sigma ** 2 / (L ** 2)
    plt.hist(-adj.flatten(), density=True, bins=200)
    plt.plot(ranga, k / ranga)
    #plt.plot(ranga, uncer.point_cdf(ranga))
    plt.plot(ranga, uncer.ele_pdf(ranga))
    plt.show()

def get_error_spectrum():
    N = 500
    L = 1
    sigma = 0.1
    space = Space(N, L, sigma)

    uncer = UniformUncertainty(N, L, sigma)
    space.place_particles()
    adj = space.adjacency()

    c = np.linspace(0, np.abs(np.min(adj.flatten())), 500)
    eps = []
    for cutoff in c:
        eps.append(uncer.absolute_error(cutoff))

    rs = uncer.r_a_relation(c)
    plt.plot(rs, eps)
    plt.ylabel('Total error')
    plt.xlabel('Cutoff radius')
    plt.show()
    expected_eigenvalue = 0 - np.sum(adj.flatten()) / N
    plt.plot(rs, eps / expected_eigenvalue)
    plt.ylabel('Total percentage error')
    plt.xlabel('Cutoff radius')
    plt.show()

def get_optimize():
    N = 500
    L = 1
    sigma = 0.1
    space = Space(N, L, sigma)

    uncer = UniformUncertainty(N, L, sigma)
    space.place_particles()
    adj = space.adjacency()
    expected_eigenvalue = 0 - np.sum(adj.flatten()) / N
    # now let d be the side length of a small box
    _,__,err = uncer.optimize_sub_box(num_test = 100, graph = True, weight_on_error = 1)
    print('estimated percentage error: ' + str(err / expected_eigenvalue))

def iterateN():
    avg_list = []
    samples = range(1000,6000,1000)
    for t in tqdm(samples):
        N = t
        L = 1
        sigma = 0.01
        space = Space(N, L, sigma)

        uncertainty = UniformUncertainty(N, L, sigma)
        plt.hist(space.eigenvalues_spectrum(weight_on_error = 1, combination=False), bins = 100, density = True)
        #space.eigenvalues_spectrum(weight_on_error=1, combination=True)
        adj = space.adj_matrix
        avg = np.trace(adj) / N
        avg_list.append(avg)
        print('Average eignevalue:' + str(avg))
        r_c = 6  # cutoff radius
        C = r_c**2 * np.pi * N / L**2
        # Expected value of off-diagonal element a_ij:
        # E[a_ij] = sqrt(2*pi) / L^2 * (1 - exp(-r_c^2/(2*sigma^2)))
        # For sigma=0.01, r_c=6: exp(-36/0.0002) ≈ 0, so E[a_ij] ≈ sqrt(2*pi)/L^2
        print((N-1) * np.sqrt(2 * np.pi) / L**2 * (1 - np.exp(-r_c**2 / (2 * sigma**2))))
        plt.show()
    plt.plot(samples,avg_list)
    plt.show()

def distribution_of_expection():
    avg_list = []
    samples = range(500)
    for t in tqdm(samples):
        N = 300
        L = 1
        sigma = 0.008
        space = Space(N, L, sigma)

        uncertainty = UniformUncertainty(N, L, sigma)
        #plt.hist(space.eigenvalues_spectrum(weight_on_error = 1, combination=False), bins = 100, density = True)
        space.eigenvalues_spectrum(weight_on_error=1, combination=False)
        adj = space.adj_matrix
        avg = np.trace(adj) / N
        avg_list.append(avg)
        #print('Average eignevalue:' + str(avg))
        #r_c = 6  # cutoff radius
        # Expected value of off-diagonal element a_ij:
        # E[a_ij] = sqrt(2*pi) / L^2 * (1 - exp(-r_c^2/(2*sigma^2)))
        # For sigma=0.01, r_c=6: exp(-36/0.0002) ≈ 0, so E[a_ij] ≈ sqrt(2*pi)/L^2
        #print(avg - (N-1) * np.sqrt(2 * np.pi) / L**2 * (1 - np.exp(-r_c**2 / (2 * sigma**2))))
        #plt.show()
    plt.hist(avg_list, bins = 50, density = True)
    def normal(x, N, L, sigma):
        x = np.array(x)
        var = (1/(2*L**2 * sigma**2) - 2*np.pi / L**4)
        mu = np.sqrt(2 * np.pi) / L**2 * (N-1)
        return 1/(np.sqrt(2*np.pi) * np.sqrt(var)) * np.exp(-(x-mu)**2/(2*var))
    mean = np.linspace(np.min(avg_list),np.max(avg_list),500)
    plt.plot(mean,normal(mean,N, L, sigma))
    plt.show()

def degree_distribution(N):
    N = N
    L = 1
    sigma = 0.01
    space = Space(N, L, sigma)
    space.place_particles()
    diag = space.adjacency(weight_on_error = 1, combination=False, diagonal = True)
    print(np.mean(diag)/np.var(diag))
    ele = []
    for i in range(N):
        ele.append(diag[i][i])
    plt.hist(ele, bins = 50, density = True)
    def normal(x, N, L, sigma):
        x = np.array(x)
        var = (1/(2*L**2 * sigma**2) - 2*np.pi / L**4) * N
        mu = np.sqrt(2 * np.pi) / L**2 * (N-1)
        return 1/(np.sqrt(2*np.pi) * np.sqrt(var)) * np.exp(-(x-mu)**2/(2*var))
    x = np.linspace(np.min(ele),np.max(ele),500)
    plt.plot(x, normal(x, N, L, sigma))
    plt.show()


def piecewise_fit(x, C, mu, sigma_sq, cutoff):
    """Vectorised piecewise function: constant C for x <= cutoff, normal PDF for x > cutoff."""
    result = np.empty_like(x, dtype=float)
    result[x <= cutoff] = C
    mask = x > cutoff
    result[mask] = 1 / np.sqrt(2 * np.pi * sigma_sq) * np.exp(-(x[mask] - mu)**2 / (2 * sigma_sq))
    return result


def piecewise_semicircle(x, C, center, radius, cutoff):
    """Vectorised piecewise function: constant C for x <= cutoff, Wigner semicircle for x > cutoff."""
    result = np.empty_like(x, dtype=float)
    result[x <= cutoff] = C
    mask = x > cutoff
    # Wigner's semicircle law: (2 / (pi * R^2)) * sqrt(R^2 - (x - center)^2) for |x - center| <= R
    r = np.sqrt(np.maximum(0, radius**2 - (x[mask] - center)**2))
    result[mask] = 2 / (np.pi * radius**2) * r
    return result

def plot3():
    L = 1
    sigma = 0.01
    Ns = [1000,5000,8000]

    # -------------------------
    # Calculate everything first
    # -------------------------
    spectra = {}

    for N in Ns:
        print(f"Calculating N={N}...", flush=True)

        space = Space(N, L, sigma)

        eigens = np.asarray(
            space.eigenvalues_spectrum(
                weight_on_error=1,
                combination=False
            )
        )

        eigens = eigens[np.isfinite(eigens)]
        spectra[N] = eigens

        print(f"Finished N={N}: {len(eigens)} eigenvalues", flush=True)

    # -------------------------
    # ONLY NOW create the figure
    # -------------------------
    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5)
    )

    for ax, N in zip(axes, Ns):
        eigens = spectra[N]

        ax.hist(
            eigens,
            bins=200,
            density=True
        )

        ax.set_title(f"$N={N}$")
        ax.set_xlabel("Eigenvalue")
        ax.set_ylabel("Density")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot3()
    #N = 5000
    #L = 1
    #sigma = 0.01
    #space = Space(N, L, sigma)
    #eigens = space.eigenvalues_spectrum(weight_on_error=1, combination=False)
    #counts, bins, _ = plt.hist(eigens, bins=200, density=True)
    #adj = space.adj_matrix
    #avg = np.trace(adj) / N
    #print('Average eigenvalue:' + str(np.mean(eigens)))
    ##print('mu:' + str(np.sqrt(2 * np.pi) / L**2 * (N-1)))
    #print('Variance in eigenvalue:' + str(np.var(eigens)))
    #print('ratio variance/mean' + str(np.var(eigens)/np.mean(eigens)))
    ##print('Standard deviation in eigenvalue:' + str(np.sqrt((1/(2*L**2 * sigma**2) - 2*np.pi / L**4) * N)))
    #r_c = 6  # cutoff radius
    #print((N - 1) * np.sqrt(2 * np.pi) / L ** 2 * (1 - np.exp(-r_c ** 2 / (2 * sigma ** 2))))
#
    # Fit the piecewise function (constant then normal) to the histogram
    #bin_centres = (bins[:-1] + bins[1:]) / 2
    #p0 = [np.max(counts), np.mean(eigens), np.var(eigens), np.percentile(eigens, 10)]
    #popt, _ = curve_fit(piecewise_fit, bin_centres, counts, p0=p0)
    #x = np.linspace(np.min(eigens), np.max(eigens), 500)
    #plt.plot(x, piecewise_fit(x, *popt), 'r-',
    #         label=f'Normal fit: C={popt[0]:.3f}, mu={popt[1]:.3f}, var={popt[2]:.3f}, cutoff={popt[3]:.3f}')
#
    ## Fit the piecewise function (constant then semicircle) to the histogram
    #p0_sc = [np.max(counts), np.mean(eigens), 2 * np.std(eigens), np.percentile(eigens, 10)]
    #popt_sc, _ = curve_fit(piecewise_semicircle, bin_centres, counts, p0=p0_sc)
    #plt.plot(x, piecewise_semicircle(x, *popt_sc), 'g--',
    #         label=f'Semicircle fit: C={popt_sc[0]:.3f}, center={popt_sc[1]:.3f}, R={popt_sc[2]:.3f}, cutoff={popt_sc[3]:.3f}')
    #plt.title(f'{N} particles')
    ##plt.legend()
    #plt.show()






