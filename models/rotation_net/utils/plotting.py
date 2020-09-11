import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


def generate_confusion_matrix(labels, matrix_values):
    """
    Generate a pyplot confusion matrix for wandb

    Parameters
    ----------
    labels : Labels for the confusion matrix
    matrix_values : Confusion matrix values

    Returns
    -------
    Pyplot confusion matrix
    """
    plt.close()
    plot_confusion_matrix(conf_mat=matrix_values,
                          colorbar=False,
                          show_absolute=True,
                          show_normed=True,
                          class_names=labels,
                          figsize=(40, 40))
    return plt

