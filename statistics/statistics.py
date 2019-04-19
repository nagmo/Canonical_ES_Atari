import matplotlib.pyplot as plt

stat_row_indexes = {
    'steps': 0,
    'time': 1,
    'mean_reward': 2,
    'max_reward': 3,
    'train_mean_reward': 4,
    'train_max_reward': 5
}


def create_plots_from_file(path, output_dir):
    results = [[] for _ in range(6)]
    labels = ['step since start', 'time since start', 'eval mean reward', 'eval max reward', 'train mean reward', 'train max reward']
    with open(path, 'r') as f:
        for line in f:
            for stat_value, res_arr in zip(line.split(','), results):
                res_arr.append(float(stat_value.strip()))
    create_plot(results[stat_row_indexes.get('time')], 'time', results[stat_row_indexes.get('mean_reward')],
                'mean reward', f'{output_dir}/mean_reward-time.plot.png')
    for res, label in zip(results, labels):
        create_plot([i for i in range(len(res))], 'iterations', res, label, f'{output_dir}/{label.replace(" ", "_")}.plot.png')


def create_plot(x, x_label, y, y_label, file_name):
    plt.plot(x, y)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    min_y = int(min(y))
    max_y = int(max(y))
    plt.yticks(range(min_y, max_y, max((max_y - min_y) // 10, 1)))
    plt.savefig(file_name)
    plt.clf()

def make_stat(in_path, out_dir):
    create_plots_from_file(in_path, out_dir)


#if __name__ == '__main__':
#    origin_stat_path = '../logs_mpi/Qbert/Baseline/Nature/40/40/0.010000/1.000000/1.000000/None/stat.txt'
#    create_plots_from_file(origin_stat_path, 'origin')

