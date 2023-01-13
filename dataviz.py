"""Plotting/Visualization"""

import json
import matplotlib.pyplot as plt

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def visualize_loss_plot(experiment_folder):
    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
    iters_total_loss = [x['iteration'] for x in experiment_metrics if 'total_loss' in x]
    total_loss = [x['total_loss'] for x in experiment_metrics if 'total_loss' in x]
    iters_val_loss = [x['iteration'] for x in experiment_metrics if 'validation_loss' in x]
    val_loss = [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x]
    # iters_ap50 = [x['iteration'] for x in experiment_metrics if 'bbox/AP50' in x]
    # ap50 = [x['bbox/AP50'] for x in experiment_metrics if 'bbox/AP50' in x]

    fig, ax = plt.subplots()
    ax.plot(iters_total_loss, total_loss)
    ax.plot(iters_val_loss, val_loss)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.legend(['total_loss', 'validation_loss'], loc='best')
    iter = val_loss.index(min(val_loss))
    ax.vlines(iters_val_loss[iter], 0, float(max(val_loss)), color="red")
    ax.annotate('min val loss: %f at iter: %d' % (float(min(val_loss)), int(iters_val_loss[iter])),
                xy=(iters_val_loss[iter], min(val_loss)), xytext=(+15, +15), textcoords='offset points',
                fontsize=8)  # arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2')
    ax.set_xlim([0, max(max(iters_total_loss), max(iters_val_loss))])
    ax.set_ylim([0, 2.0])

    plt.show()
    path = experiment_folder + '/training_loss.jpg'
    plt.savefig(path)
    plt.close()


