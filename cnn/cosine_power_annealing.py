# Cosine Power Annealing
# Author: Andrew Hundt (ATHundt@gmail.com)
import numpy as np


def cosine_power_annealing(
        epochs=None, max_lr=0.1, min_lr=1e-4, exponent_order=10,
        max_epoch=None, warmup_epochs=None, return_intermediates=False):
    """ Cosine Power annealing is designed to be an improvement on cosine annealing.

    Often the cosine annealing schedule decreases too slowly at the beginning
    and to quickly at the end. A simple exponential annealing curve does the
    opposite, decreasing too quickly at the beginning and too slowly at the end.
    Power cosine annealing strikes a configurable balance between the two.
    The larger the exponent order, the faster exponential decay occurs.
    The smaller the exponent order, the more like cosine annealing the curve becomes.

    # Arguments

    epochs: 1d numpy array with one or more epoch numbers which you expect to run.
        This is set up so that you can start from the beginning or resume training.
        Typical input will be np.arange(total_epochs) + 1.
    max_lr: The maximum learning rate which is also the initial learning rate.
    min_lr: The minimum learning rate which is also the final learning rate.
    exponent_order: Determines how fast the learning rate decays.
        A value of 1 will perform standard cosine annealing, while
        10 will decay with an exponential base of 10.
    max_epoch: The maximum epoch number that will be encountered.
        This is usually specified for when you are getting a single learning rate
        value at the current epoch, or for resuming training runs.
    return_intermediates: True changes the return value to be
        [range_limited_cos_power_proportions, cos_power_proportions, cos_proportions]
        which is useful for comparing, understanding, and plotting several possible
        learning rate curves. False returns only the range_limited_cos_power_proportions
        learning rate values.

    # Returns

        A 1d numpy array range_limited_cos_power_proportions, which will contain the learning
        rate you should use at each of the specified epochs.
    """
    if epochs is None and max_epoch is None:
        raise ValueError('cosine_power_annealing() requires either the "epochs" parameter, "max_epochs" parameter, '
                         'or both to be specified. "epochs" can be a single value or an array of epoch values, but got ' +
                         str(epochs) + '. "max_epoch" can be a single value, but got ' + str(max_epoch) + '.')
    elif epochs is None:
        epochs = np.arange(max_epoch) + 1
    elif max_epoch is None:
        max_epoch = np.max(epochs)

    if warmup_epochs is not None and warmup_epochs > 0:
        # change warmup epochs to do some warmup
        warmups = epochs <= warmup_epochs
        max_epoch -= warmup_epochs
        epochs -= warmup_epochs

    # first half of cosine curve scaled from 1 to 0
    cos_proportions = (1 + np.cos(np.pi * epochs / max_epoch)) / 2
    # power curve applied to cosine values
    if exponent_order < 1:
        raise ValueError('cosine_power_annealing() requires the "exponent order" parameter'
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
        cos_power_proportions[warmups] = np.arange(1, warmup_epochs + 1) / warmup_epochs
    # rescale the power curve between the user specified min and max learning rate
    range_limited_cos_power_proportions = ((cos_power_proportions * (max_lr - min_lr)) + min_lr)

    if return_intermediates:
        return range_limited_cos_power_proportions, cos_power_proportions, cos_proportions
    else:
        return range_limited_cos_power_proportions


def main():
    # example of how to set up cosine power annealing
    max_lr = 0.8
    exponent_order = 10
    max_epoch = 100
    epochs = np.arange(max_epoch) + 1
    min_lr = 0.02

    # standard cosine power annealing
    schedules = cosine_power_annealing(
           epochs, max_lr, min_lr,
           exponent_order=exponent_order, return_intermediates=True)

    [range_limited_cos_power_proportions, cos_power_proportions,
     cos_proportions] = schedules

    # power cosine annealing with warmup
    warmup_schedules = cosine_power_annealing(
           epochs, max_lr, min_lr, warmup_epochs=5,
           exponent_order=exponent_order, return_intermediates=True)
    [warmup_range_limited_cos_power_proportions, warmup_cos_power_proportions,
     warmup_cos_proportions] = warmup_schedules

    print("epochs")
    print(epochs)
    print("cos_proportions")
    print(cos_proportions)
    print("cos_power_proportions")
    print(cos_power_proportions)
    print("range_limited_cos_power_proportions")
    print(range_limited_cos_power_proportions)
    labels = ['epochs', 'cos_proportions', 'cos_power_proportions', 'range_limited_cos_power_proportions',
              'warmup_cos_power_proportions', 'warmup_range_limited_cos_power_proportions']
    np.savetxt(
        'power_cosine_annealing_schedule.csv',
        [epochs] + list(schedules) + list(warmup_schedules[1:]),
        header=', '.join(labels))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot_cos_proportions = ax.plot(cos_proportions, label='cos annealing proportions')
    plot_cos_power_proportions = ax.plot(cos_power_proportions, label='cos power annealing proportions')
    warmup_plot_cos_power_proportions = ax.plot(warmup_cos_power_proportions, label='warmup + cos power annealing proportions')
    # plt.plot(log_limited)
    plot_range_limited_cos_power_proportions = ax.plot(
        range_limited_cos_power_proportions, label='range limited ' + str(max_lr) + ' to ' + str(min_lr))
    warmup_plot_range_limited_cos_power_proportions = ax.plot(
        warmup_range_limited_cos_power_proportions, label='warmup + range limited ' + str(max_lr) + ' to ' + str(min_lr))
    # ax.legend(schedules, labels[1:])
    legend = ax.legend(loc='upper right', shadow=True, fontsize='large')
    # plt.plot(result)
    plt.ylabel('learning rate')
    plt.xlabel('epoch')
    plt.show()

if __name__ == '__main__':
    main()