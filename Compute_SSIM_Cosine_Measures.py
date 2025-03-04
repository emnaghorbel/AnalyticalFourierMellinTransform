import numpy as np
import cv2
import matplotlib.pyplot as plt
from find_object_center import find_object_center
from Cartesian2polar2cartesian import cartesian_to_polar
from AFMT_IAFMT import compute_AFMT
from invariant_features import compute_invariant_features
# Load original grayscale image
image = cv2.imread("Cancer (1).jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Error loading image! Check the file path.")

# Define parameters
N = image.shape[0] // 2  # Number of radial samples
M = 360  # Number of angular samples

# Convert Cartesian to Polar
polar_image = cartesian_to_polar(image, N, M)

# Compute AFMT for original image
AFMT_coeffs_orig = compute_AFMT(polar_image, N, M)

# Compute invariant features
I_f_orig = compute_invariant_features(AFMT_coeffs_orig, sigma=1)

# 🔄 Apply Rotation
angle = 45  # Rotate by 45 degrees
rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # Example rotation
polar_rotated = cartesian_to_polar(rotated_image, N, M)

# Compute AFMT for rotated image
AFMT_coeffs_rot = compute_AFMT(polar_rotated, N, M)

# Compute invariant features for rotated image
I_f_rot = compute_invariant_features(AFMT_coeffs_rot, sigma=1)
# 🔹 Normalize Invariant Features
def normalize_features(I_f, method="minmax"):
    """Normalize the invariant feature matrix using different methods."""
    if method == "minmax":
        return (I_f - np.min(I_f)) / (np.max(I_f) - np.min(I_f))
    elif method == "l2":
        return I_f / np.linalg.norm(I_f, ord=2)
    elif method == "meanvar":
        return (I_f - np.mean(I_f)) / np.std(I_f)
    else:
        raise ValueError("Unknown normalization method!")
I_f_orig = normalize_features(I_f_orig, method="minmax")
I_f_rot= normalize_features(I_f_rot, method="minmax")

# Compute Cosine Similarity
cosine_similarity = np.dot(I_f_orig.flatten().real, I_f_rot.flatten().real) / (
    np.linalg.norm(I_f_orig.flatten().real) * np.linalg.norm(I_f_rot.flatten().real)
)

def complex_distance(I_f1, I_f2):
    """
    Compute a distance between two complex-valued invariant feature matrices.

    :param I_f1: First invariant matrix (complex-valued).
    :param I_f2: Second invariant matrix (complex-valued).
    :return: Scalar distance value.
    """
    if I_f1.shape != I_f2.shape:
        raise ValueError("Both feature matrices must have the same shape!")

    # Compute magnitude and phase differences
    magnitude_diff = np.abs(np.abs(I_f1) - np.abs(I_f2))  # Difference in magnitudes
    phase_diff = np.abs(np.angle(I_f1) - np.angle(I_f2))  # Difference in angles

    # Normalize phase difference to be in [0, π]
    phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)

    # Weighted combination of magnitude and phase differences
    alpha = 0.5  # Weight for magnitude vs. phase
    distance = np.sqrt((1 - alpha) * np.sum(magnitude_diff**2) + alpha * np.sum(phase_diff**2))

    return distance

# Compute SSIM
distance = complex_distance(I_f_orig, I_f_rot)

# Print SSIM score
print(f"🔹 complex_distance: {distance:.4f}")

print(f"🔹 Cosine Similarity: {cosine_similarity}")
