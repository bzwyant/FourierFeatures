
import matplotlib.pyplot as plt
import numpy as np


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


def plot_rgb_distribution_kde(pixels, x_range=None):
    """
    Plot RGB distributions using Kernel Density Estimation for very smooth curves
    """
    from scipy.stats import gaussian_kde
    
    # Extract RGB channels
    red_values = pixels[:, 0]
    green_values = pixels[:, 1] 
    blue_values = pixels[:, 2]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine the range for evaluation

    if x_range is None:
        all_values = np.concatenate([red_values, green_values, blue_values])
        x_min = np.min(all_values)
        x_max = np.max(all_values)
    else:
        x_min, x_max = x_range

    x_range = np.linspace(x_min, x_max, 300)
    
    # Calculate KDE for each channel
    red_kde = gaussian_kde(red_values)
    green_kde = gaussian_kde(green_values)
    blue_kde = gaussian_kde(blue_values)
    
    # Plot smooth curves
    ax.plot(x_range, red_kde(x_range), color='red', linewidth=2, label='Red', alpha=0.8)
    ax.plot(x_range, green_kde(x_range), color='green', linewidth=2, label='Green', alpha=0.8)
    ax.plot(x_range, blue_kde(x_range), color='blue', linewidth=2, label='Blue', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Density')
    ax.set_title('RGB Value Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()