import itertools
import math
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np


def false_match_rate(bad: List[float], treshold):
    return len([e for e in bad if e > treshold]) / len(bad)


def true_match_rate(good: List[float], treshold):
    return len([e for e in good if e > treshold]) / len(good)


def split_good_bad(matrix: List[List[float]], id: List[int]) -> Tuple[List[float], List[float]]:
    """
    1  2  3  4  5
    6  7  8  9  10
    11 12 13 14 15
    16 17 18 19 20
    21 22 23 24 25

    with id = [1 1 2 2 2] and a symmetry
    we have all below the diagonal 1 7 13 19 25 to classify (the diagonal is not taken into account)
    so 6 is good for 1
    11 is bad because 1 against 2
    12 is bad because 1 against 2
    16 is bad because 1 against 2
    17 is bad because 1 against 2
    18 is good because 2 against 2
    21 is bad because 1 against 2
    22 is bad because 1 against 2
    23 is good because 2 against 2
    24 is good because 2 against 2
    so we return [6,18,23,24], [11,12,16,17,21,22]

    :param matrix:
    :param id:
    :return:
    """
    good: List[float] = []
    bad: List[float] = []

    squares_size: Dict[int, Tuple[int, int]] = {}
    length = 0
    for key, iter in itertools.groupby(id):
        current_square = len(list(iter))
        squares_size[key] = (length, current_square)
        length += current_square
    for _, v in squares_size.items():
        bad_length, good_length = v
        # for all lines inside the good square offset by the bad length
        for y in range(bad_length, bad_length + good_length):
            for x in range(0, bad_length):
                # noinspection PyPackageRequirements
                if x < y:  # if below diagonal
                    bad.append(matrix[y][x])
            for x in range(bad_length, bad_length + good_length):
                if x < y:  # if below diagonal
                    good.append(matrix[y][x])

    return good, bad


def test():
    m = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25],
    ]
    id = [1, 2, 2, 2, 2]
    assert split_good_bad(m, id) == ([12, 17, 18, 22, 23, 24], [6, 11, 16, 21])
    id = [1, 1, 2, 2, 2]
    assert split_good_bad(m, id) == ([6, 18, 23, 24], [11, 12, 16, 17, 21, 22])
    id = [1, 1, 1, 2, 2]
    assert split_good_bad(m, id) == ([6, 11, 12, 24], [16, 17, 18, 21, 22, 23])
    id = [1, 1, 1, 1, 2]
    assert split_good_bad(m, id) == ([6, 11, 12, 16, 17, 18], [21, 22, 23, 24])
    id = [1, 1, 1, 1, 1]
    assert split_good_bad(m, id) == ([6, 11, 12, 16, 17, 18, 21, 22, 23, 24], [])


def load_matrix(filename: str) -> List[List[float]]:
    matrix: List[List[float]] = []
    with open(filename) as f:
        for line in f:
            row: List[float] = list(map(float, line.split()))
            matrix.append(row)
    return matrix


def load_id(filename: str) -> List[int]:
    id: List[int] = []
    with open(filename) as f:
        id = list(map(int, f.readline().split()))
    return id


def plot_hists(good: List[float], bad: List[float], size: int):
    bin_size = round(math.sqrt(size * size / 2))
    print("Using binsize", bin_size)
    plt.figure(1)
    plt.hist(good, bins=bin_size, stacked=True, histtype='bar', alpha=0.5, label='Genuine distribution',
             density=True)  # arguments are passed to np.histogram
    plt.hist(bad, bins=bin_size, stacked=True, histtype='bar', alpha=0.5, label='Imposter Distribution',
             density=True)  # arguments are passed to np.histogram
    plt.title("GENUINE AND IMPOSTER SCORE DISTRIBUTION")
    plt.xlabel('Match Score')
    plt.ylabel('Probability')
    plt.legend()


def get_low_high(_good, _bad):
    return min(min(_good), min(_bad)), max(max(_good), max(_bad))


def plot_FMR(_good: List[float], _bad: List[float]):
    low, high = get_low_high(_good, _bad)
    r = np.linspace(low, high)
    y = []
    for i in range(len(r)):
        y.append(false_match_rate(_bad, r[i]))
    plt.figure(2)
    plt.xlabel('t')
    plt.ylabel('FMR(t)')
    plt.title('FMR compared to t')
    plt.plot(r, y)


def plot_FNMR(_good: List[float], _bad: List[float]):
    low, high = get_low_high(_good, _bad)
    r = np.linspace(low, high)
    y = []
    for i in range(len(r)):
        y.append(1 - true_match_rate(_good, r[i]))
    plt.figure(3)
    plt.xlabel('t')
    plt.ylabel('FNMR(t)')
    plt.title('FNMR compared to t')
    plt.plot(r, y)


def plot_DET(_good: List[float], _bad: List[float]):
    low, high = get_low_high(_good, _bad)
    r = np.linspace(low, high)
    x = []
    y = []
    for i in range(len(r)):
        x.append(false_match_rate(_bad, r[i]))  # FMR
        y.append(1 - true_match_rate(_good, r[i]))  # FNMR
    plt.figure(4)
    plt.xlabel('FMR(t)')
    plt.ylabel('FNMR(t)')
    plt.title('DET')
    plt.plot(x, y)


def plot_ROC(_good: List[float], _bad: List[float]):
    low, high = get_low_high(_good, _bad)
    r = np.linspace(low, high)
    x = []
    y = []
    for i in range(len(r)):
        x.append(false_match_rate(_bad, r[i]))  # FMR
        y.append(true_match_rate(_good, r[i]))  # TMR
    plt.figure(5)
    plt.xlabel('FMR(t)')
    plt.ylabel('TMR(t)')
    plt.title('ROC')
    plt.plot(x, y)


def compute_eer(_good: List[float], _bad: List[float]):
    fpr = []
    tpr = []
    low, high = get_low_high(_good, _bad)
    thresholds = np.linspace(low, high)
    for i in range(len(thresholds)):
        fpr.append(false_match_rate(_bad, thresholds[i]))  # FMR
        tpr.append(true_match_rate(_good, thresholds[i]))  # TMR
    err = 0, (0, 0)
    mini = 2
    for item_fpr, item_tpr, threshold in zip(fpr, tpr, thresholds):
        item_fnr = 1 - item_tpr
        s = abs(item_fnr - item_fpr)  # FNR -FPR
        if s < mini:
            mini = s
            err = (item_fpr + item_fnr) / 2, (item_fpr, item_tpr, threshold)
    return err


def trace_matrix(id,size=200):
    arr = np.zeros((size, size))
    for i, y in enumerate(id[:size]):
        for j, x in enumerate(id[:size]):
            if x == y:
                arr[i][j] = 1
    plt.figure(0)
    plt.imshow(arr, interpolation='none')


def get_metrics(name: str):

    print(name)
    import os
    os.makedirs(f"images_metrics/{name}", exist_ok=True)
    matrix: List[List[float]] = load_matrix(f"data_metrics/matrix_{name}.txt")
    id: List[int] = load_id(f"data_metrics/id_{name}.txt")
    trace_matrix(id,len(id))
    #plt.savefig(f'images_metrics/{name}/matrix_id.eps', format='eps', dpi=100)
    plt.savefig(f'images_metrics/{name}/matrix_id.png', format='png', dpi=100)
    good, bad = split_good_bad(matrix, id)
    from pyeer.eer_info import get_eer_stats
    from pyeer.report import generate_eer_report, export_error_rates
    from pyeer.plot import plot_eer_stats

    # Calculating stats for classifier A
    stats_a = get_eer_stats(good, bad)
    print(stats_a)
    plot_eer_stats([stats_a], ['A'])
    print(len(good), len(bad))
    print(compute_eer(good, bad))
    plot_hists(good, bad, len(matrix))
    plt.savefig(f'images_metrics/{name}/hists.png')
    print("histogram done")
    plot_FMR(good, bad)
    plt.savefig(f'images_metrics/{name}/FMR.png')
    print("FMR done")
    plot_FNMR(good, bad)
    plt.savefig(f'images_metrics/{name}/FNMR.png')
    print("FNMR done")
    plot_DET(good, bad)
    plt.savefig(f'images_metrics/{name}/DET.png')
    print("DET done")
    plot_ROC(good, bad)
    plt.savefig(f'images_metrics/{name}/ROC.png')
    print("ROC done")
    plt.show()


if __name__ == '__main__':
    get_metrics("originals")
    for i in range(1, 7):
        get_metrics("improved" + str(i))
