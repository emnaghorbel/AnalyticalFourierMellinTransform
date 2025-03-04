from find_object_center import find_object_center
import numpy as np

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
