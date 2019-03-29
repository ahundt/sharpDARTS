# Cosine Power Annealing
# Author: Andrew Hundt (ATHundt@gmail.com)
import numpy as np


def cosine_power_annealing(
        epochs=None, max_lr=0.1, min_lr=1e-4, exponent_order=10,
        max_epoch=None, warmup_epochs=None, return_intermediates=False,
        start_epoch=1, restart_lr=True):
    """ Cosine Power annealing is designed to be an improvement on cosine annealing.

    Often the cosine annealing schedule decreases too slowly at the beginning
    and to quickly at the end. A simple exponential annealing curve does the
    opposite, decreasing too quickly at the beginning and too slowly at the end.
    Power cosine annealing strikes a configurable balance between the two.
    The larger the exponent order, the faster exponential decay occurs.
    The smaller the exponent order, the more like cosine annealing the curve becomes.

    # Arguments

    epochs: An integer indicating the number of epochs to train.
        If you are resuming from epoch 100 and want to run until epoch 300,
        specify 200.
    max_lr: The maximum learning rate which is also the initial learning rate.
    min_lr: The minimum learning rate which is also the final learning rate.
    exponent_order: Determines how fast the learning rate decays.
        A value of 1 will perform standard cosine annealing, while
        10 will decay with an exponential base of 10.
    max_epoch: The maximum epoch number that will be encountered.
        This is usually specified for when you are getting a single learning rate
        value at the current epoch, or for resuming training runs.
    return_intermediates: True changes the return value to be
        [cos_power_annealing, cos_power_proportions, cos_proportions]
        which is useful for comparing, understanding, and plotting several possible
        learning rate curves. False returns only the cos_power_annealing
        learning rate values.,
    start_epoch: The epoch number to start training from which will be at index 0
        of the returned numpy array.
    restart_lr: If True the training curve will be returned as if starting from
        epoch 1, even if you are resuming from a later epoch. Otherwise we will
        return with the learning rate as if you have already trained up to
        the value specified by start_epoch.

    # Returns

        A 1d numpy array cos_power_annealing, which will contain the learning
        rate you should use at each of the specified epochs.
    """
    if epochs is None and max_epoch is None:
        raise ValueError('cosine_power_annealing() requires either the "epochs" parameter, "max_epochs" parameter, '
                         'or both to be specified. "epochs" can be a single value or an array of epoch values, but got ' +
                         str(epochs) + '. "max_epoch" can be a single value, but got ' + str(max_epoch) + '.')
    elif epochs is None:
        epochs = np.arange(max_epoch) + 1
    elif isinstance(epochs, int):
        epochs = np.arange(epochs) + 1
    if max_epoch is None:
        max_epoch = np.max(epochs)

    if warmup_epochs is not None and warmup_epochs > 0:
        min_epoch = np.min(epochs)
        if min_epoch > 1:
            raise ValueError(
                'cosine_power_annealing(): '
                'Resuming training with warmup enabled is not yet directly supported! '
                'The workaround is to create a training curve starting from 1, '
                'then get your learning rates by indexing from your current epoch onwards. '
                'Expected: warmup_epochs=None or np.min(epochs)=1, '
                'but got warmup_epochs=' + str(warmup_epochs) + ' and np.min(epochs)=' + str(min_epoch))
        # change warmup epochs to do some warmup
        warmups = epochs <= warmup_epochs + min_epoch
        max_epoch -= warmup_epochs
        epochs -= warmup_epochs

    # first half of cosine curve scaled from 1 to 0
    cos_proportions = (1 + np.cos(np.pi * epochs / max_epoch)) / 2
    # power curve applied to cosine values
    if exponent_order < 1:
        raise ValueError('cosine_power_annealing() requires the "exponent order" parameter '
                         'to be greater than or equal to 1 but got ' + str(exponent_order) + '.')
    elif exponent_order == 1:
        cos_power_proportions = cos_proportions
    else:
        cos_power_proportions = np.power(exponent_order, cos_proportions + 1)
    # rescale the power curve from the current range to be from 1 to 0
    cos_power_proportions = cos_power_proportions - np.min(cos_power_proportions)
    cos_power_proportions = cos_power_proportions / np.max(cos_power_proportions)
    # check if we are doing warmup
    if warmup_epochs is not None and warmup_epochs > 0:
        # set the proportion values which apply during the warmup phase
        warmup_end = min_epoch + warmup_epochs
        cos_power_proportions[warmups] = np.arange(min_epoch, warmup_end + 1) / float(warmup_end)
    # rescale the power curve between the user specified min and max learning rate
    cos_power_annealing = ((cos_power_proportions * (max_lr - min_lr)) + min_lr)

    if start_epoch > 1:
        # If we are resuming, extract the portion of the curve
        # the user asked for.
        if restart_lr:
            cos_power_annealing = cos_power_annealing[:len(epochs)]
        else:
            cos_power_annealing = cos_power_annealing[start_epoch-1:]

    if return_intermediates:
        return cos_power_annealing, cos_power_proportions, cos_proportions
    else:
        return cos_power_annealing

def plot_power_annealing_schedule(epochs, max_lr, min_lr, exponent_order, restart_lr=True, y_scale='linear', save_filename=''):
    # standard cosine power annealing
    schedules = cosine_power_annealing(
           epochs, max_lr, min_lr, restart_lr=restart_lr,
           exponent_order=exponent_order, return_intermediates=True)

    [cos_power_annealing, cos_power_proportions,
     cos_proportions] = schedules

    # calculate the pure cosine annealing range limited values
    range_limited_cos_annealing_proportions = ((cos_proportions * (max_lr - min_lr)) + min_lr)

    # power cosine annealing with warmup
    warmup_schedules = cosine_power_annealing(
           epochs, max_lr, min_lr, warmup_epochs=5,
           exponent_order=exponent_order, restart_lr=restart_lr,
           return_intermediates=True)
    [warmup_cos_power_annealing, warmup_cos_power_proportions,
     warmup_cos_proportions] = warmup_schedules
    epochs = np.arange(epochs) + 1
    print("epochs")
    print(epochs)
    print("cos_proportions")
    print(cos_proportions)
    print("cos_power_proportions")
    print(cos_power_proportions)
    print("cos_power_annealing")
    print(cos_power_annealing)
    labels = ['epochs', 'cos_proportions', 'cos_power_proportions', 'cos_power_annealing',
              'warmup_cos_power_proportions', 'warmup_cos_power_annealing']
    np.savetxt(
        'power_cosine_annealing_schedule.csv',
        [epochs] + list(schedules) + list(warmup_schedules[1:]),
        header=', '.join(labels))

    import matplotlib
    import matplotlib.pyplot as plt

    # source for font selection code: http://jonathansoma.com/lede/data-studio/matplotlib/changing-fonts-in-matplotlib/
    # Say, "the default sans-serif font is COMIC SANS"
    matplotlib.rcParams['font.sans-serif'] = "Georgia"
    # Then, "ALWAYS use sans-serif fonts"
    matplotlib.rcParams['font.family'] = "serif"
    fontsize = '20'

    min_lr_str = '{0:.1E}'.format(min_lr)
    min_lr_str = min_lr_str.replace('E', 'e')
    min_lr_str = min_lr_str.replace('04', '4')
    fig, ax = plt.subplots()
    if y_scale == 'linear':
        plot_cos_proportions = ax.plot(cos_proportions, label='cos annealing, from 1 to 0', color='darkorange')
        warmup_plot_cos_power_proportions = ax.plot(warmup_cos_power_proportions, label='warmup + cos power annealing, from 1 to 0', color='darkviolet')
        plot_cos_power_proportions = ax.plot(cos_power_proportions, label='cos power annealing, from 1 to 0')
    plot_range_limited_cos_annealing_proportions = ax.plot(
        range_limited_cos_annealing_proportions, label='cos annealing, from ' + '{0:.1}'.format(max_lr) + ' to ' + min_lr_str, color='orange')

    # plt.plot(log_limited)
    plot_cos_power_annealing = ax.plot(
        cos_power_annealing, label='cos power annealing, from ' + '{0:.1}'.format(max_lr) + ' to ' + min_lr_str, color='royalblue')
    warmup_plot_cos_power_annealing = ax.plot(
        warmup_cos_power_annealing, label='warmup + cos power annealing, from ' + '{0:.1}'.format(max_lr) + ' to ' + min_lr_str, color='darkmagenta')
    # ax.legend(schedules, labels[1:])
    legend = ax.legend(loc='bottom left', shadow=False, fontsize='14')
    # plt.plot(result)
    lr_str = 'learning rate, ' + y_scale + ' scale'
    plt.ylabel(lr_str, fontsize=fontsize)
    plt.xlabel('epoch', fontsize=fontsize)
    plt.yscale(y_scale)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    plt.tight_layout()
    plt.show()
    if save_filename:
        fig.savefig(save_filename, bbox_inches='tight')


def main(plot_example='imagenet'):
    # example of how to set up cosine power annealing with a configuration designed for imagenet
    # plot_example = 'cifar10'
    # plot_example = 'resume_imagenet'
    # plot_example = 'sub1'
    if plot_example == 'imagenet':
        max_lr = 0.1
        exponent_order = 10
        # max_epoch = 300
        # epochs = np.arange(max_epoch) + 1
        epochs = 300
        min_lr = 7.5e-4
        restart_lr = True
    elif plot_example == 'resume_imagenet':
        max_lr = 0.2
        exponent_order = 10
        min_epoch = 300
        max_epoch = 600
        # epochs = np.arange(min_epoch, max_epoch) + 1
        epochs = max_epoch - min_epoch
        min_lr = 7.5e-4
        restart_lr = True
    elif plot_example == 'cifar10':
        max_lr = 0.025
        exponent_order = 2
        epochs = 1000
        min_lr = 7.5e-4
        restart_lr = True
    elif plot_example == 'sub1':
        max_lr = 0.1
        exponent_order = 0.5
        # max_epoch = 300
        # epochs = np.arange(max_epoch) + 1
        epochs = 300
        min_lr = 7.5e-4
        restart_lr = True
    else:
        raise ValueError('main(): unknown plot_example: ' + str(plot_example))

    # standard cosine power annealing
    plot_power_annealing_schedule(
        epochs, max_lr, min_lr, exponent_order, restart_lr=restart_lr,
        save_filename='cosine_power_annealing_' + plot_example + '.pdf')
    plot_power_annealing_schedule(
        epochs, max_lr, min_lr, exponent_order, restart_lr=restart_lr,
        y_scale='log',
        save_filename='cosine_power_annealing_' + plot_example + '_log.pdf')


if __name__ == '__main__':
    main()