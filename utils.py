import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(
    scores, avg_scores, thetas, title, figure_file=None, show=False
):
    x = np.arange(len(scores))
    fig, ax1 = plt.subplots()
    lines = []

    ax1.set_title(title)
    ax1.set_xlabel("Epoch")
    lines += ax1.plot(x, scores, "b", label="score")
    lines += ax1.plot(x, avg_scores, "g", label="avg")
    ax1.set_ylabel("Score", color="b")
    ax1.tick_params("y", colors="b")

    ax2 = ax1.twinx()
    lines += ax2.plot(x, thetas, "r", label="theta")
    ax2.set_ylabel("Theta", color="r")
    ax2.tick_params("y", colors="r")

    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc=2)
    if show:
        plt.show()
    if figure_file:
        plt.savefig(figure_file)


if __name__ == "__main__":
    plot_learning_curve(
        scores=[1, 2, 3, 4, 5],
        avg_scores=[1, 1, 2, 2, 3],
        thetas=[0, 0.2, 0.4, 0.6, 1.0],
        title="some title",
        show=True,
    )
