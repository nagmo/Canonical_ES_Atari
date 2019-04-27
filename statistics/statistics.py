import matplotlib.pyplot as plt
import numpy as np

stat_row_indexes = {
    'steps': 0,
    'time': 1,
    'mean_reward': 2,
    'max_reward': 3,
    'train_mean_reward': 4,
    'train_max_reward': 5
}

labels = ['step since start', 'time since start', 'eval mean reward', 'eval max reward', 'train mean reward', 'train max reward']


def create_plots_from_file(path, output_dir):
    results = [[] for _ in range(6)]
    with open(path, 'r') as f:
        for line in f:
            for stat_value, res_arr in zip(line.split(','), results):
                res_arr.append(float(stat_value.strip()))
    create_plot(results[stat_row_indexes.get('time')], 'time', results[stat_row_indexes.get('mean_reward')],
                'mean reward', f'{output_dir}/mean_reward-time.plot.png')
    for res, label in zip(results, labels):
        create_plot([i for i in range(len(res))], 'iterations', res, label, f'{output_dir}/{label.replace(" ", "_")}.plot.png')


def create_plot(x, x_label, y, y_label, file_name):
    plt.plot(x, y, 'gray')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    min_y = int(min(y))
    max_y = int(max(y))
    plt.yticks(range(min_y, max_y, max((max_y - min_y) // 10, 1)))
    plt.savefig(file_name)
    plt.clf()


def create_plot_from_two_files(p1, p2, output_dir):
    r1 = [[] for _ in range(6)]
    r2 = [[] for _ in range(6)]
    with open(p1, 'r') as f1, open(p2, 'r') as f2:
        for l1, l2 in zip(f1, f2):
            for v1, a1, v2, a2 in zip(l1.split(','), r1, l2.split(','), r2):
                a1.append(float(v1.strip()))
                a2.append(float(v2.strip()))
    create_two_legends_plot(r1[stat_row_indexes.get('mean_reward')], [i for i in range(len(r1[stat_row_indexes.get('mean_reward')]))],
                            r2[stat_row_indexes.get('mean_reward')], [i for i in range(len(r1[stat_row_indexes.get('mean_reward')]))])


def create_two_legends_plot(x1, y1, x2, y2):
    n = 6
    cut = 20
    fig, ax = plt.subplots()
    ax.plot(y1[:-n-cut], smooth_array(x1, n)[:-cut], 'black')
    ax.plot(y1[:-n-cut], x1[:-n-cut], 'gray')
    # ax.plot(y2[:-n-cut], smooth_array(x2, n)[:-cut], 'gray')

    plt.xlabel('iterations')
    plt.ylabel('score')
    plt.legend(['OurES', 'CanonicalES'])
    plt.show()


def smooth_array(arr, n):
    new_arr = []
    for i in range(len(arr) - n):
        new_arr.append(np.mean(arr[i:i+n]))
    return new_arr


def make_stat(in_path, out_dir):
    create_plots_from_file(in_path, out_dir)


if __name__ == '__main__':
    origin_stat_path = '../logs_mpi/Qbert/Baseline/Nature/40/5/0.010000    /1.000000/1.000000/origin/stat.txt'
    novelty_stat_path = '../logs_mpi/Qbert/Baseline/Nature/40/5/0.010000    /1.000000/1.000000/novelty/stat.txt'
    # create_plots_from_file(origin_stat_path, 'origin')
    create_plot_from_two_files(novelty_stat_path, origin_stat_path, '')
