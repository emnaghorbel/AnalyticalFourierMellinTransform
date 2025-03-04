def compute_invariant_features(M_f, sigma=1):
    """
    Computes similarity-invariant features for all (k, v) pairs from AFMT.

    :param M_f: 2D AFMT coefficients (matrix).
    :param sigma: Strictly positive parameter.
    :return: 2D matrix of invariant features I_f.
    """
    N, M = M_f.shape
    I_f = np.zeros((N, M), dtype=complex)  # Create a full matrix

    M_f_00 = M_f[0, 0]  # Central AFMT coefficient
    M_f_10 = M_f[1, 0]  # First order coefficient

    if M_f_00 == 0 or M_f_10 == 0:
        raise ValueError("M_f(0,0) and M_f(1,0) must be nonzero.")

    # Compute for all (k, v)
    for k in range(N):
        for v in range(M):
            I_f[k, v] = (M_f_00 ** (-sigma + 1j * v)) * np.exp(1j * k * np.angle(M_f_10)) * M_f[k, v]

    return I_f

def compute_inverse_invariant_features(I_f, M_f_10, M_f_00, sigma=1):
    """
    Computes the full AFMT matrix from invariant features.

    :param I_f: 2D invariant feature matrix.
    :param M_f_10: AFMT coefficient Mf(1, 0).
    :param M_f_00: AFMT coefficient Mf(0, 0).
    :param sigma: Strictly positive parameter.
    :return: Reconstructed 2D AFMT matrix.
    """
    N, M = I_f.shape
    M_f_reconstructed = np.zeros((N, M), dtype=complex)

    if M_f_00 == 0 or M_f_10 == 0:
        raise ValueError("M_f(0,0) and M_f(1,0) must be nonzero.")

    # Compute for all (k, v)
    for k in range(N):
        for v in range(M):
            M_f_reconstructed[k, v] = I_f[k, v] * (M_f_00 ** (sigma - 1j * v)) * np.exp(-1j * k * np.angle(M_f_10))

    return M_f_reconstructed
