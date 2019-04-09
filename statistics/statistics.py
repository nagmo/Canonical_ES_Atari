import matplotlib.pyplot as plt


def create_plots_from_file(path):
    results = [[] for _ in range(6)]
    labels = ['step since start', 'time since start', 'eval mean reward', 'eval max reward', 'train mean reward', 'train max reward']
    with open(path, 'r') as f:
        for line in f:
            for stat_value, res_arr in zip(line.split(','), results):
                res_arr.append(float(stat_value.strip()))
    for res, label in zip(results, labels):
        create_plot([i for i in range(len(res))], 'iterations', res, label, path)


def create_plot(x, x_label, y, y_label, file_name):
    plt.plot(x, y)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    min_y = int(min(y))
    max_y = int(max(y))
    plt.yticks(range(min_y, max_y, int((max_y - min_y) / 10)))
    plt.show()


if __name__ == '__main__':
    origin_stat_path = '../logs_mpi/Qbert/Baseline/Nature/40/40/0.010000/1.000000/1.000000/None/stat.txt'
    create_plots_from_file(origin_stat_path)
