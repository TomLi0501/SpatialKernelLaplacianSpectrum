import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

def get_gif():
    # get all data
    file_info = []
    parentfolder = 'data/NormallyDistributed'
    data_dir = os.listdir(parentfolder)

    for i in tqdm(data_dir):
        data = os.listdir(f'{parentfolder}/' + i)
        dxt = i.split(' ')
        dx = float(dxt[0].replace('dx=', ''))
        t = float(dxt[1].replace('s', ''))

        for q in data:
            path = f'{parentfolder}/' + i + '/' + q
            sigma = float(q.replace('sigma_', '').replace('.txt', ''))
            eigens = np.loadtxt(path)
            x = np.linspace(0, 1, len(eigens))
            file_info.append((dx, t, sigma, x, eigens))

    # sort by dx
    file_info.sort(key=lambda x: x[0])
    file_selected = []
    for i in file_info:
        if i[2] == 7.75e-2:
            file_selected.append(i)
    file_info = file_selected

    fig, ax = plt.subplots(figsize=(10, 6))


    def update(frame):
        dx, t, sigma, x, eigens = file_info[frame]
        ax.clear()
        ax.plot(x, eigens)
        ax.set_title(f'dx={dx}, t={t}, σ={sigma:.3f}', fontsize=12)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Eigenvalue', fontsize=10)
        return ax,


    anim = animation.FuncAnimation(fig, update, frames=len(file_info), interval=5)

    # save as GIF
    anim.save('eigenvalues_animation.gif', writer='pillow', fps=2, dpi=100)

    plt.close()
    print("GIF Saved: eigenvalues_animation.gif")


def graph(dx:str, t:str, sigma:str):
    dx = dx
    t = t
    sigma = sigma
    parentfolder = 'data/UniformlyDistributed'
    path = f'{parentfolder}/' + f'dx={dx} {t}s/' + f'sigma_{sigma}.txt'
    eigens = np.loadtxt(path)
    c = 0
    for i in eigens[2]:
        print(i)
        if i <= 1e-10:
            c += 1
    print(c)
    x = np.linspace(0, float(t), len(eigens))
    plt.plot(x, eigens)
    plt.xlabel('Time')
    plt.show()

def same_time_correlation(dx: str, t: str, sigma: str):

    parentfolder = 'data/UniformlyDistributed'
    path = f'{parentfolder}/dx={dx} {t}s/sigma_{sigma}.txt'

    eigens = np.loadtxt(path)
    cor_matrix = np.corrcoef(eigens.T)

    fig = plt.figure()
    writer = PillowWriter(fps=30)   # 根据需要调整帧率

    with writer.saving(fig, "same_time_correlation.gif", dpi=100):
        for line in range(1000):
            plt.clf()

            x = range(len(cor_matrix[line]))
            y = cor_matrix[line]

            plt.ylim(-1, 1)
            plt.title(f'i = {line} (dx={dx}, t={t}, σ={sigma})')
            plt.plot(x, np.zeros(len(x)), color='red')
            plt.plot(x, y)
            plt.pause(0.01)
            #writer.grab_frame()
def Cij(dx: str, t: str, sigma: str):
    parentfolder = 'data/UniformlyDistributed'
    path = f'{parentfolder}/dx={dx} {t}s/sigma_{sigma}.txt'
    #print(path)
    eigens = np.loadtxt(path)
    eigens = eigens.T

    fig = plt.figure()
    writer = PillowWriter(fps=10)
    i = 200
    with writer.saving(fig, f"C{i}j.gif", dpi=100):
        frames = []
        average = 401
        for timing in tqdm(np.linspace(average, len(eigens[0]), len(eigens[0]) - average)):
            timing = int(timing)

            e = []
            for lk in eigens:
                e.append(lk[timing-(average-1):timing])
            cor_matrix = np.corrcoef(e)
            #e = np.asarray(e)
            #e_centered = e - e.mean(axis=1, keepdims=True)
            #cor_matrix = e_centered @ e_centered.T

            frames.append(cor_matrix)
        for j in range(1000):
            C1j = [m[i-1][j-1] for m in frames]
            plt.clf()
            plt.plot(range(len(frames)), C1j)
            plt.ylim(-1, 1)
            plt.title(f'j = {j}; i = {i}')
            plt.xlabel('Time')
            plt.pause(0.01)
            #writer.grab_frame()

def MatrixHeatmap(dx: str, t: str, sigma: str):
    parentfolder = 'data/UniformlyDistributed'
    path = f'{parentfolder}/dx={dx} {t}s/sigma_{sigma}.txt'
    eigens = np.loadtxt(path)
    eigens = eigens.T

    fig = plt.figure()
    writer = PillowWriter(fps=10)
    i = 100
    with writer.saving(fig, f"C{i}j.gif", dpi=100):
        frames = []
        average = 301
        for timing in tqdm(np.linspace(average, len(eigens[0]), len(eigens[0]) - average)):
            timing = int(timing)

            e = []
            for lk in eigens:
                e.append(lk[timing-(average-1):timing])

            cor_matrix = np.corrcoef(e)
            frames.append(cor_matrix)
            #print(len(frames))
            plt.clf()
            plt.imshow(cor_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
            plt.colorbar(label='Correlation coefficient')
            plt.xlabel('Eigenvalue index')
            plt.ylabel('Eigenvalue index')
            plt.title(f'Correlation Matrix heatmap (dx={dx}, t={t}, σ={sigma}, t={timing})')
            plt.pause(0.01)

def heatmap(dx: str, t: str, sigma: str):
    parentfolder = 'data/UniformlyDistributed'
    path = f'{parentfolder}/dx={dx} {t}s/sigma_{sigma}.txt'

    eigens = np.loadtxt(path)
    cor_matrix = np.corrcoef(eigens.T)
    plt.clf()
    plt.imshow(cor_matrix, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Correlation coefficient')
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue index')
    plt.title(f'Correlation Matrix heatmap (dx={dx}, t={t}, σ={sigma})')
    plt.show()

def increase_sigma(dx, t):
    for s in np.logspace(1e-2, 1, 50):
        sigma = f'{s:.2e}'
        heatmap(dx, t, sigma)




dx = '1.00e-01'
t = '20.0'
sigma = '1.00e-01'
heatmap(dx, t, sigma)
#MatrixHeatmap(dx, t, sigma)
#graph(dx, t, sigma)
#get_gif()
