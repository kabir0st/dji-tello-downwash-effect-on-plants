import cv2
import numpy as np

IMAGE_PATH = "image.jpg"

DISTANCE_MM = 750.0
TAG_SIZE_MM = 50.0


def make_detector():
    # used 4x4 aruco marker
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def measure_tag(gray, detector):
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return None
    pts = corners[0][0].astype(np.float64)
    edge_lengths = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    tag_width_px = float(np.mean(edge_lengths))
    return corners, ids, edge_lengths, tag_width_px


def estimate(tag_width_px):
    fx_est = tag_width_px * DISTANCE_MM / TAG_SIZE_MM
    scale_px_per_mm = tag_width_px / TAG_SIZE_MM
    return fx_est, scale_px_per_mm


def print_results(tag_width_px, fx_est, scale_px_per_mm):
    print(f"Mean tag width: {tag_width_px:.2f} px")
    print(f"Estimated focal length: {fx_est:.2f} px")
    print(f"Scale at {DISTANCE_MM:.0f} mm: {scale_px_per_mm:.4f} px/mm")


if __name__ == "__main__":
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Cannot open {IMAGE_PATH}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    print(f"Image resolution: {width} x {height} pixels")

    result = measure_tag(gray, make_detector())
    if result is None:
        raise RuntimeError("No marker detected.")
    corners, ids, edge_lengths, tag_width_px = result
    fx_est, scale_px_per_mm = estimate(tag_width_px)

    print(f"Detected marker ID: {ids.flatten().tolist()}")
    print(f"Tag edge lengths (px): {edge_lengths}")
    print_results(tag_width_px, fx_est, scale_px_per_mm)

    display = image.copy()
    cv2.aruco.drawDetectedMarkers(display, corners, ids)
    cv2.imshow("Detected ArUco marker", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
