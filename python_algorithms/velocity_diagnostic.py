import numpy as np
import matplotlib.pyplot as plt


def plot_graph(x, kde_post, kde_prior, gaussian_prior, v_min, v_max,
               similarity=None, save_path="graph.png"):

    try:
        x = np.array(x)
        kde_post = np.array(kde_post)
        kde_prior = np.array(kde_prior)
        gaussian_prior = np.array(gaussian_prior)

        overlap = np.minimum(kde_post, kde_prior)
        # overlap_integral = np.trapz(overlap, x)

        product_integral = np.trapz(kde_post * kde_prior, x)
        self_post = np.trapz(kde_post**2, x)
        self_prior = np.trapz(kde_prior**2, x)
        product_integral_norm = product_integral / \
            np.sqrt(self_post * self_prior)

        bhattacharyya = np.trapz(
            np.sqrt(np.maximum(kde_post * kde_prior, 0)), x)

        plt.figure(figsize=(10, 7))

        plt.plot(x, kde_prior, linewidth=2, label="KDE Prior")
        plt.plot(x, kde_post, linewidth=2, label="KDE Posterior")
        plt.plot(x, gaussian_prior, linewidth=2, label="Gaussian Prior")
        plt.fill_between(x, overlap, alpha=0.3, label="Overlap")

        title = f"Product Norm: {product_integral_norm:.4f} \
        | Bhattacharyya: {bhattacharyya:.4f}"
        if similarity is not None:
            sim_val = float(np.atleast_1d(similarity)[0])
            title += f"\n Similarity: {sim_val:.4f}"

        plt.title(title)
        plt.xlim(v_min, v_max)
        plt.xlabel("Velocity")
        plt.ylabel("Density")
        plt.grid(True)
        plt.legend()

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close()

    except Exception as e:
        print("Exception inside plot_graph:", e)
        raise


def __test_args__(*args):
    """
    Тестовая функция для проверки передачи аргументов из C++ через PythonBridge
    """
    print("Python received args:")
    for i, a in enumerate(args):
        print(f"arg[{i}] =", a)
    return args
