import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import imageio
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.font_manager import FontProperties

fp = FontProperties(family='Tlwg Typo', size=10)


def plot_piechart(x, labels, title='', fig_size=(10, 5), save=None):
    fig = plt.figure(figsize=fig_size)

    ax1 = fig.add_subplot(121)
    wedges, texts = ax1.pie(x, labels=labels, startangle=90)

    percents = x / sum(x) * 100.
    annots = ['{} - {:.2f}% ({:d})'.format(c, p, n) for c, p, n
              in zip(labels, percents, x)]

    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    ax2.legend(wedges, annots, loc='center', fontsize=10)

    fig.suptitle(title)

    if save is not None:
        fig.savefig(save)
        plt.close()
    else:
        return fig


def plot_x(x, title='', fig_size=(12, 10)):
    fig = plt.figure(figsize=fig_size)
    x = np.squeeze(x)

    if len(x.shape) == 1:
        plt.plot(x)

    elif len(x.shape) == 2:
        plt.imshow(x, cmap='gray')
        plt.axis('off')

    elif len(x.shape) == 3:
        if x.shape[-1] == 3:
            plt.imshow(x)
            plt.axis('off')
        else:
            fig = plot_multiImage(x.transpose(2, 0, 1), fig_size=fig_size)

    elif len(x.shape) == 4:
        fig = plot_multiImage(x.transpose(3, 0, 1, 2), fig_size=fig_size)

    fig.suptitle(title)
    return fig


def plot_bars(x, y, title='', ylim=None, save=None):
    fig = plt.figure()
    bars = plt.bar(x, y)
    plt.ylim(ylim)
    plt.title(title)
    for b in bars:
        plt.annotate('{:.2f}'.format(b.get_height()),
                     xy=(b.get_x(), b.get_height()))

    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        return fig


def plot_graphs(x_list, legends, title, ylabel, xlabel='epoch', xlim=None, save=None):
    fig = plt.figure()
    for x in x_list:
        plt.plot(x)

    plt.legend(legends)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(xlim)

    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        return fig


# images in shape (amount, h, w, c).
def plot_multiImage(images, labels=None, pred=None, title=None, fig_size=(12, 10), tight_layout=False, save=None):
    n = int(np.ceil(np.sqrt(images.shape[0])))
    fig = plt.figure(figsize=fig_size)

    for i in range(images.shape[0]):
        ax = fig.add_subplot(n, n, i + 1)

        if len(images[i].shape) == 2 or images[i].shape[-1] == 1:
            ax.imshow(images[i], cmap='gray')
        else:
            ax.imshow(images[i])

        if labels is not None:
            ax.set_xlabel(labels[i], color='g', fontproperties=fp)
        if labels is not None and pred is not None:
            if labels[i] == pred[i]:
                clr = 'g'
            else:
                if len(labels[i]) == len(pred[i]):
                    clr = 'm'
                else:
                    clr = 'r'

            ax.set_xlabel('True: {}\nPred : {}'.format(u'' + labels[i], u'' + pred[i]),
                          color=clr, fontproperties=fp)

    if title is not None:
        fig.suptitle(title)

    if tight_layout:  # This make process slow if too many images.
        fig.tight_layout()

    if save is not None:
        plt.savefig(save)
        plt.close()
    else:
        return fig


def plot_confusion_metrix(y_true, y_pred, labels=None, title='', normalize=None,
                          fig_size=(10, 10), save=None):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    if labels is None:
        labels = list(set(y_true))

    disp = ConfusionMatrixDisplay(cm, labels)
    disp.plot(xticks_rotation=45)
    disp.figure_.set_size_inches(fig_size)
    disp.figure_.suptitle(title)
    disp.figure_.tight_layout()

    if save is not None:
        disp.figure_.savefig(save)
        plt.close()
    else:
        return disp.figure_


def get_fig_image(fig):  # figure to array of image.
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer._renderer)
    return img


def vid2gif(video_file, output_file, delay=0.05):
    with imageio.get_writer(output_file, mode='I', duration=delay) as writer:
        cap = cv2.VideoCapture(video_file)
        while True:
            ret, frame = cap.read()
            if ret:
                # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                writer.append_data(frame)
            else:
                break
