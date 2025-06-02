import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, ax=None):
    """
    Plot a confusion matrix.
    
    Parameters:
    - cm: Confusion matrix (numpy array)
    - classes: List of class labels
    - normalize: Whether to normalize the confusion matrix
    - title: Title of the plot
    - cmap: Color map for the plot
    - ax: Matplotlib axis to plot on (optional)
    
    Returns:
    - None
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if ax is None:
        # Create a new figure and axis if ax is not provided
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        # If ax is provided, extract the figure from the axis
        fig = ax.figure
    
    # Plot the confusion matrix
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(cax)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    
    plt.tight_layout()
    plt.show()
