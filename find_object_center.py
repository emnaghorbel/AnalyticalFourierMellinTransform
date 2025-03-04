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
