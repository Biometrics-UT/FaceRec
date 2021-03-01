from src.fft import run_fft
from src.get_best import get_best
from src.get_matrix import get_matrices
from src.get_metrics import run_metrics
from src.get_scores import get_scores

if __name__ == '__main__':
    get_best()
    get_scores()
    get_matrices()
    run_metrics()
    run_fft()
