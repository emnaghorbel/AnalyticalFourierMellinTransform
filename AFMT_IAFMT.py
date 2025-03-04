import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_AFMT(f_polar, R_max, N_theta):
    """
    Approximation de la Transformation de Fourier-Mellin Analytique (AFMT).

    :param f_polar: Image en coordonnées polaires (r, theta).
    :param R_max: Rayon maximal.
    :param N_theta: Nombre d’angles.
    :return: Coefficients AFMT sous forme d’un tableau complexe.
    """
    M_f = np.zeros((R_max, N_theta), dtype=complex)

    for r in range(R_max):
        for k in range(N_theta):
            sum_m = 0
            for m in range(N_theta):
                sum_m += f_polar[r, m] * np.exp(-2j * np.pi * m * k / N_theta)
            M_f[r, k] = sum_m / N_theta

    return M_f

def compute_IAFMT(M_f, R_max, N_theta):
    """
    Approximation de la Transformation de Fourier-Mellin Analytique Inverse (IAFMT).

    :param M_f: Coefficients AFMT.
    :param R_max: Rayon maximal.
    :param N_theta: Nombre d’angles.
    :return: Image reconstruite en coordonnées polaires.
    """
    f_reconstructed = np.zeros((R_max, N_theta), dtype=complex)

    for r in range(R_max):
        for m in range(N_theta):
            sum_k = 0
            for k in range(N_theta):
                sum_k += M_f[r, k] * np.exp(2j * np.pi * m * k / N_theta)
            f_reconstructed[r, m] = sum_k

    return np.real(f_reconstructed)
