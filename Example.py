import numpy as np
import cv2
import matplotlib.pyplot as plt
def cartesian_to_polar(image, R_max, N_theta):
    """
    Convertit une image en coordonnées polaires avec interpolation bilinéaire.
    
    :param image: Image en niveaux de gris sous forme de tableau 2D.
    :param R_max: Rayon maximal considéré.
    :param N_theta: Nombre d’échantillons angulaires.
    :return: Image interpolée en coordonnées polaires.
    """
    h, w = image.shape
    center = (w // 2, h // 2)
    
    polar_image = np.zeros((R_max, N_theta))
    
    for r in range(R_max):
        for theta in range(N_theta):
            angle = 2 * np.pi * theta / N_theta
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            
            if 0 <= x < w and 0 <= y < h:
                polar_image[r, theta] = image[y, x]
    
    return polar_image

def polar_to_cartesian(polar_image, original_shape):
    """
    Convert an image from polar to Cartesian coordinates using inverse mapping.
    
    :param polar_image: The image in polar coordinates.
    :param original_shape: The shape (height, width) of the original Cartesian image.
    :return: The reconstructed Cartesian image.
    """
    H, W = original_shape
    center = (W // 2, H // 2)
    R_max, N_theta = polar_image.shape

    cartesian_image = np.zeros((H, W), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            dx = x - center[0]
            dy = y - center[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            if theta < 0:
                theta += 2 * np.pi

            r_idx = int(r)
            theta_idx = int(theta * N_theta / (2 * np.pi))

            if 0 <= r_idx < R_max and 0 <= theta_idx < N_theta:
                cartesian_image[y, x] = polar_image[r_idx, theta_idx]

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

# Charger une image en niveaux de gris
image = cv2.imread("Cancer (1).jpg", cv2.IMREAD_GRAYSCALE)

# Définition des paramètres
R_max = 300  # Rayon maximal
N_theta = 200  # Nombre de points angulaires

# Conversion en coordonnées polaires
polar_image = cartesian_to_polar(image, R_max, N_theta)

# Calcul de la AFMT
AFMT_coeffs = compute_AFMT(polar_image, R_max, N_theta)

# Reconstruction via IAFMT
reconstructed_polar = compute_IAFMT(AFMT_coeffs, R_max, N_theta)


# Convert Polar back to Cartesian
reconstructed_image = polar_to_cartesian(reconstructed_polar, image.shape)
# Affichage des résultats
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Image originale")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(polar_image, cmap='gray', aspect='auto')
plt.title("Image en coordonnées polaires")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_image, cmap='gray', aspect='auto')
plt.title("Image reconstruite (IAFMT)")
plt.axis("off")

plt.tight_layout()
plt.show()
