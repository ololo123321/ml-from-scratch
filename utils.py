import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(model, data2d, y, n_grid=100):
    xx = np.linspace(data2d[:, 0].min()-1, data2d[:, 0].max()+1, n_grid)
    yy = np.linspace(data2d[:, 1].min()-1, data2d[:, 1].max()+1, n_grid).T
    xx, yy = np.meshgrid(xx, yy)
    grid = np.c_[xx.ravel(), yy.ravel()]

    model.fit(data2d, y)
    Z = model.predict_proba(grid)[:, 1].reshape((n_grid, n_grid))
    # print(np.unique(Z))

    fig, ax = plt.subplots(figsize=(10, 5))
    contour = ax.contourf(xx, yy, Z, cmap=plt.cm.jet, alpha=.8)
    ax.scatter(data2d[:, 0], data2d[:, 1], marker='o', c=y, cmap=plt.cm.Blues, edgecolor='k')
    plt.colorbar(contour)
    plt.show()
