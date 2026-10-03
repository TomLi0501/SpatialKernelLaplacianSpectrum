from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image
import io
class Space:
    def __init__(self,N, L, sigma, dt = 1e-5, Dx = 1):
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
        self.dt = dt
        self.Dx = Dx
        self.t = 0
        self.place_particles()

    def place_particles(self): # Gaussian blob centered in the box
        self._particles = []
        self._particles_index = []
        for p_index in range(self.N):
            x = np.random.random() * self.L
            y = np.random.random() * self.L
            self._particles.append(np.array([x,y]))
            self._particles_index.append(p_index)

    def distance(self, pos1: list, pos2:list):
        dx = np.abs(pos2[0] - pos1[0])
        dy = np.abs(pos2[1] - pos1[1])
        if dx > self.L/2: # All distance smaller than L/2
            dx = self.L - dx
        if dy > self.L/2:
            dy = self.L - dy
        return np.sqrt(dx**2 + dy**2)

    def update(self):
        # one, sampling
        zeta = []
        for _ in self._particles_index: # Do I need to sample different seeds to make it 'more' random? it is probably overshoot
            rx = np.random.normal()
            ry = np.random.normal()
            zeta.append(np.array([rx, ry]))
        zeta = np.array(zeta)
        # two, compute change in particles
        factor = np.sqrt(2 * self.Dx * self.dt)
        dr = factor * zeta
        # three, move particles according to the change(and update time for recording)
        self._particles = np.array(self._particles) + dr
        for pos in self._particles:
            if pos[0] > self.L:
                pos[0] -= self.L
            if pos[0] < 0:
                pos[0] += self.L
            if pos[1] > self.L:
                pos[1] -= self.L
            if pos[1] < 0:
                    pos[1] += self.L
        self.t += self.dt

    def kernel(self, pos1: list, pos2: list):
        dr = self.distance(pos1, pos2)
        # linear Kernel
        # return dr
        # Gaussian kernel
        return 1 * np.exp(-dr**2/(2*self.sigma**2))

    def adjacency(self,diagonal=False, combination = False):
        self.adj_matrix = np.zeros((self.N, self.N)) # clean it first
        if combination:
            selected_pairs = combinations(self._particles_index, 2) # O(N^2)
        else:
            selected_pairs = self.split_box_get_all_pairs()
        # if i != j

        for particle_pairs in tqdm(selected_pairs, 'computing adj_matrix', disable=True):
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

    def eigenvalues_spectrum(self,  combination = False, diagonal=True):
        self.adjacency(diagonal=diagonal, combination = combination)
        spectrum = np.linalg.eigh(self.adj_matrix)[0] # the first is eigen value
        return spectrum

    def split_box(self):
        # Number of boxes per dimension (ensures full coverage of [0, L))
        self.box_length = 4 * self.sigma
        box_nums = int(np.ceil(self.L / self.box_length))
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

    def split_box_get_all_pairs(self):
        self.all_pairs = set()
        self.split_box()
        for i in tqdm(self._particles_index,'preparing pairs', disable=True):
            self.get_neighbouring_points(i)
        return list(self.all_pairs)


def cor_matrix(e, test = ''):
    cormatrix = np.cov(e, rowvar=True)
    plt.imshow(cormatrix, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Correlation coefficient')
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue index')
    plt.title(f'Correlation Matrix heatmap')
    plt.savefig(f'correlation_matrix_Test{test}.png')
    plt.show()


def conduct():
    N = 1000
    L = 1
    dt = 0.01
    Dx = 0.01
    sigma = 10
    runs = 200
    space = Space(N, L, sigma, dt=dt, Dx=Dx)
    time_detector = []
    eigenvalues_detector = []
    c = 0
    for _ in tqdm(range(runs), 'running', leave=False):
        c += 1
        eigenvalues = space.eigenvalues_spectrum()
        # rescale eigenvalues by rho sigma ^2
        rho = N / L ** 2
        eigenvalues = eigenvalues / (rho * sigma ** 2) # Eigenvalue Normalised
        space.update()
        time_detector.append(space.t)
        eigenvalues_detector.append(eigenvalues)
        # Save adjmatrix
        if c == 10:
            f = open('adj10.txt', 'w')
            f.write(str(space.adj_matrix))
            f.close()
        if c==100:
            f = open('adj100.txt', 'w')
            f.write(str(space.adj_matrix))
            f.close()
        if c == 150:
            f = open('adj150.txt', 'w')
            f.write(str(space.adj_matrix))
            f.close()


    return eigenvalues_detector, time_detector

# Four Factors May results in Negative Correlation

'''
1. Diffusion Process
2. Laplacian Matrix
3. Spacial Structure
4. Gaussian Kernel
'''

# Test On 1
class Test1(Space):
    pass
    def update(self):
       self.place_particles()

# Test On 2
class Test2(Space):
    pass
    def eigenvalues_spectrum(self, diagonal=False, combination=False):
        return super().eigenvalues_spectrum(diagonal=diagonal, combination=combination)

# Test On 3
class Test3(Space):
    pass
    def distance(self, pos1: list, pos2:list):
        return np.random.rand()
    # R is always random
    # This test automatically ignore Diffusion as well. which should be irrelevent

class Test4(Space):
    pass
    def kernel(self, pos1: list, pos2: list):
        return self.distance(pos1, pos2)

def nearest_distance(adj):
    nearest = []
    for i in adj:
        nearest.append(np.abs(min(i)))
    return np.sum(nearest)




def correlation_nearrestD():
    N = 1000
    L = 1
    dt = 0.01
    Dx = 0.01
    sigma = 1
    runs = 200
    space = Space(N, L, sigma, dt=dt, Dx=Dx)
    time_detector = []
    eigenvalues_detector = []
    spreadness_detector = []
    for _ in tqdm(range(runs), 'running', leave=False):
        eigenvalues = space.eigenvalues_spectrum()
        # rescale eigenvalues by rho sigma ^2
        rho = N / L ** 2
        eigenvalues = eigenvalues / (rho * sigma ** 2)
        space.update()
        time_detector.append(space.t)
        eigenvalues_detector.append(eigenvalues)
        spreadness_detector.append(nearest_distance(space.adj_matrix))


    return eigenvalues_detector, spreadness_detector

if __name__ == '__main__':
    e,t = conduct()
    e = np.array(e)
    e = e.T
    #e = np.ones((1000, 1000))
    #e = - e # negative
    #for i in range(1000):
    #    for j in range(1000):
    #        if i == j:
    #             e[i][j] = 999

    cor_matrix(e, test = 'connectedness')







