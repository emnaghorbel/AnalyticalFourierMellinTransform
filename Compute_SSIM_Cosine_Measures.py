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
def compute_ssim(I_f_orig, I_f_rot):
    """
    Computes the Structural Similarity Index (SSIM) between two invariant feature matrices.

    :param I_f_orig: Original invariant features (complex matrix).
    :param I_f_rot: Rotated invariant features (complex matrix).
    :return: SSIM value
    """
    # Convert to real values (taking absolute values)
    I_f_orig_real = np.abs(I_f_orig)
    I_f_rot_real = np.abs(I_f_rot)

    # Compute SSIM
    ssim_value = ssim(I_f_orig_real, I_f_rot_real, data_range=I_f_orig_real.max() - I_f_orig_real.min())

    return ssim_value
# Compute SSIM
ssim_score = compute_ssim(I_f_orig, I_f_rot)

# Print SSIM score
print(f"🔹 Structural Similarity Index (SSIM): {ssim_score:.4f}")

print(f"🔹 Cosine Similarity: {cosine_similarity}")
