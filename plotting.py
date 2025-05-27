
import matplotlib.pyplot as plt


def plot_images(images, suptitle=None, titles=None, figsize=(20, 10), save_path=None):
    """
    Plot a list of images in a grid format.

    Args:
        images (list): List of images to plot.
        suptitle (str): Super title for the figure.
        titles (list): List of titles for each image.
        figsize (tuple): Size of the figure. 
        save_path (str): Path to save the figure. If None, the figure will not be saved.
    """
    plt.figure(figsize=figsize)
    rows = 1
    cols = len(images)

    if len(images) == 0:
        raise ValueError("No images to plot.")

    if suptitle:
        plt.suptitle(suptitle, fontsize=16)

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if titles:
            plt.title(titles[i])
        plt.imshow(img)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()