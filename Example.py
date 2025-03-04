import numpy as np
import cv2
import matplotlib.pyplot as plt
def find_object_center(image):
    """
    Find the centroid of the main object in the image.

    :param image: Grayscale image
    :return: (cx, cy) coordinates of the object center
    """
    _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    M = cv2.moments(binary)

    if M["m00"] == 0:
        return (image.shape[1] // 2, image.shape[0] // 2)

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return cx, cy

def cartesian_to_polar(image, N, M):
    """
    Convert an image to polar coordinates using fixed radial and angular steps.

    :param image: Grayscale image.
    :param N: Number of radial lines.
    :param M: Number of concentric circles.
    :return: Polar transformed image.
    """
    h, w = image.shape
    cx, cy = find_object_center(image)  # Compute object center
    R = min(cx, cy, w - cx, h - cy)  # Smallest radius containing object

    # Compute angular and radial sampling steps
    d_theta = 2 * np.pi / M
    d_rho = R / N

    # Create an empty polar image [N x M]
    polar_image = np.zeros((N, M))

    for n in range(N):  # Radial samples
        rho = n * d_rho  # Fixed radial step
        for m in range(M):  # Angular samples
            theta = m * d_theta  # Fixed angular step

            # Convert to Cartesian coordinates
            x = cx + rho * np.cos(theta)
            y = cy + rho * np.sin(theta)

            if 0 <= x < w - 1 and 0 <= y < h - 1:
                x0, y0 = int(x), int(y)
                dx, dy = x - x0, y - y0

                # Bilinear interpolation
                polar_image[n, m] = (1 - dx) * (1 - dy) * image[y0, x0] + \
                                     dx * (1 - dy) * image[y0, x0 + 1] + \
                                     (1 - dx) * dy * image[y0 + 1, x0] + \
                                     dx * dy * image[y0 + 1, x0 + 1]

    return polar_image
def polar_to_cartesian(polar_image, original_shape):
    H, W = original_shape
    N, M = polar_image.shape

    # Use the same center as in cartesian_to_polar()
    cx, cy = W // 2, H // 2  
    R = min(cx, cy, W - cx, H - cy)  # Match max radius from forward transform

    cartesian_image = np.zeros((H, W), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            dx, dy = x - cx, y - cy
            rho = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            if theta < 0:
                theta += 2 * np.pi

            # Normalize rho to match N (radial samples)
            r_idx = rho * (N - 1) / R  
            theta_idx = theta * (M - 1) / (2 * np.pi)

            # Bilinear interpolation
            r0, r1 = int(np.floor(r_idx)), int(np.ceil(r_idx))
            t0, t1 = int(np.floor(theta_idx)), int(np.ceil(theta_idx))

            if 0 <= r0 < N and 0 <= t0 < M and 0 <= r1 < N and 0 <= t1 < M:
                f00 = polar_image[r0, t0]
                f01 = polar_image[r0, t1]
                f10 = polar_image[r1, t0]
                f11 = polar_image[r1, t1]

                dr, dt = r_idx - r0, theta_idx - t0  # Fractional part

                value = (f00 * (1 - dr) * (1 - dt) + f01 * (1 - dr) * dt +
                         f10 * dr * (1 - dt) + f11 * dr * dt)

                cartesian_image[y, x] = np.clip(value, 0, 255)  

    return cartesian_image
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


### **🔹 Run Full Processing Pipeline**
# 📌 Load grayscale image
image = cv2.imread("Cancer (1).jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Error loading image! Check the file path.")

# Define parameters
N = image.shape[0]//2  # Number of radial samples
M = 360  # Number of angular samples

# Convert Cartesian to Polar
polar_image= cartesian_to_polar(image, N, M)

# Compute AFMT
AFMT_coeffs = compute_AFMT(polar_image, N, M)

# Compute Invariant Features
k, v = 10, 10  # Example indices
I_f = compute_invariant_features(AFMT_coeffs, 1)
IAFMT_coeffs = compute_inverse_invariant_features(I_f, AFMT_coeffs[1, 0], AFMT_coeffs[0, 0], 1)
# Reconstruct via IAFMT
reconstructed_polar = compute_IAFMT(IAFMT_coeffs , N, M)

# Convert back to Cartesian
reconstructed_image = polar_to_cartesian(reconstructed_polar, image.shape)

# ✅ Save & Display
cv2.imwrite("polar_transformed.jpg", polar_image)
cv2.imwrite("AFMT_coeffs.jpg", np.abs(AFMT_coeffs))
cv2.imwrite("I_f.jpg", np.abs(I_f))
cv2.imwrite("polar_rec.jpg", reconstructed_polar)
cv2.imwrite("reconstructed_image.jpg", reconstructed_image)

plt.imshow(reconstructed_image, cmap="gray")
plt.title("Reconstructed Image")
plt.axis("off")
plt.show()
