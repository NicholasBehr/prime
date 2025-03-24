import matplotlib.pyplot as plt
import numpy as np


def fig_feasibility():
    # make data
    u = np.linspace(-2.5, 1, 100)

    def f(x):
        return 2 * x**2 + x**3

    def f_grad(x):
        return 4 * x + 3 * x**2

    y = f(u)

    z1 = -1.2972
    z2 = -1.403

    z = -1.31
    g1 = f(z) + f_grad(z) * (u - z)
    print("intersect", (1 - f(z)) / f_grad(z) + z)
    print("intersect", (0.5 - f(z)) / f_grad(z) + z)

    # override
    plt.rcParams.update({"figure.figsize": (6, 3)})

    # plot
    fig, ax = plt.subplots(layout="constrained", sharex=True)

    ax.plot(u, y, color="C0", label=r"$y = h(u)$")
    ax.plot(u, g1, color="C1", label=r"$h(-1.31)$ linearization")

    # constraints
    ax.axvline(x=0, color="C3", linestyle="--", label="constraints")
    ax.axvline(x=-2, color="C3", linestyle="--")
    ax.axhline(y=0.5, color="C3", linestyle="--")
    ax.axhline(y=1, color="C3", linestyle="--")

    # linearization
    ax.axvspan(z1, z2, color="C3", alpha=0.3, label="primal infeasible region")

    ax.set(
        xlim=(-2.5, 1), xticks=np.arange(-2.5, 1.5, 0.5), ylim=(0, 1.5), yticks=np.arange(0, 2, 0.5)
    )
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$y$")
    ax.legend(loc="lower right")


if __name__ == "__main__":
    fig_feasibility()
    plt.show()
