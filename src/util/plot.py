import numpy as np
import matplotlib.pyplot as plt
from config import cfg

opt = cfg


def plotting(history, model_name=opt.model, target=opt.target, num_epoch=opt.epochs):

    for key in history.keys():
        plt.plot(history[key], label=key)
        plt.legend()
    plt.xticks(np.arange(num_epoch, step=2))
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title(f'loss for {model_name} model on {target} prediction')
    plt.savefig(f'plots/{target}_{model_name}.png')
