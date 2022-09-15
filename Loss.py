import matplotlib.pyplot as plt 

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def plotLoss(loss, paths, name):
    epochs = range(1, len(loss['kl']) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, loss['kl'], 'b', label='Training KLD')
    ax.plot(epochs, loss['val_kl'], 'r', label='Validation KLD')
    ax.plot(epochs, loss['rec'], 'cyan', label='Training recon.')
    ax.plot(epochs, loss['val_rec'], 'orange', label='Validation recon.')
    ax.plot(epochs, loss['total'], 'green', label='Training total')
    ax.plot(epochs, loss['val_total'], 'yellow', label='Validation total')
    ax.set_title('Training and validation losses')

    legend_without_duplicate_labels(ax)
    plt.savefig(paths["vis_path"]+"losses_" + name + ".png", bbox_inches='tight', format="png", dpi=400)