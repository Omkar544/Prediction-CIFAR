import matplotlib.pyplot as plt

def plot_metrics(history, train_metric, val_metric):
    plt.plot(history.history[train_metric], label='Training')
    plt.plot(history.history[val_metric], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.show()
