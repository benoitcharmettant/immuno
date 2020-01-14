from os.path import join

import matplotlib.pyplot as plt

log_info = {
    'train_loss': 9,
    'train_acc': 12,
    'val_loss': 15,
    'val_acc': 18,
    'val_acc_1': 20,
}


def parse_log_file(path_log_file, infos):
    ls = [[] for i in range(len(infos))]

    with open(path_log_file, 'r') as f:
        lines = f.readlines()

    for line in lines:

        if line[24: 35] == 'Train Epoch':

            line = line.replace(')', '')
            line = line.replace('\n', '')
            line = line.replace('\t', ' ')
            line = line.split(' ')

            for i, info in enumerate(infos):
                ls[i].append(float(line[log_info[info]]))

    return ls


def smooth(ls, weight):
    smoothed = [ls[0]]
    for x in ls[1:]:
        smoothed_x = weight * smoothed[-1] + (1 - weight) * x
        smoothed.append(smoothed_x)
    return smoothed


def plot_training(path_log_dir, log_file_name="console.log", random_pred_level=None):
    path_log_file = join(path_log_dir, log_file_name)
    path_target_file = join(path_log_dir, "training_visualisation.png")



    infos = ['train_loss', 'train_acc', 'val_loss', 'val_acc']


    [train_loss, train_acc, val_loss, val_acc] = parse_log_file(path_log_file, infos)

    weight = 0.9

    train_loss_smooth = smooth(train_loss, weight)
    val_loss_smooth = smooth(val_loss, weight)
    train_acc_smooth = smooth(train_acc, weight)
    val_acc_smooth = smooth(val_acc, weight)

    train_epoch = range(len(train_acc))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col')

    ax1.plot(train_epoch, train_loss, '.', color='#4083ff', alpha=0.1, label="Train")
    ax1.plot(train_epoch, train_loss_smooth, color='#4083ff', label=f"Train (smooth {weight})")

    ax1.plot(train_epoch, val_loss, '.', color='#ff9900', alpha=0.1, label="Validation")
    ax1.plot(train_epoch, val_loss_smooth, color='#ff9900', label=f"Validation (smooth {weight})")
    ax1.set(ylabel='Loss')
    ax1.grid(True)
    ax1.legend(prop={'size': 4})

    ax2.plot(train_epoch, train_acc, '.', alpha=0.1, color='#4083ff')
    ax2.plot(train_epoch, train_acc_smooth, color='#4083ff')

    ax2.plot(train_epoch, val_acc, '.', alpha=0.1, color='#ff9900')
    ax2.plot(train_epoch, val_acc_smooth, color='#ff9900')

    if random_pred_level is not None:
        random_pred = [random_pred_level for i in range(len(train_acc))]
        ax2.plot(train_epoch, random_pred, linewidth=1, color='grey')

    ax2.grid(True)
    ax2.set(xlabel='Training epochs', ylabel='Accuracy')

    plt.savefig(path_target_file)


if __name__ == "__main__":
    plot_training("/home/opis/bcharmme/logs/toy/convnet_lr0.0001_e4000_bs20_ps0.4_s40_r0.04")
