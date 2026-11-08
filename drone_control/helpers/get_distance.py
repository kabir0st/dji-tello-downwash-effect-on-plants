import cv2
from calibrate_camera import TAG_SIZE_MM, make_detector, measure_tag

FOCAL_LENGTH_PX = 2817.82
CALIB_WIDTH_PX = 2268  # width of image.jpg used for calibration

_detector = make_detector()


def focal_for_width(width_px):
    """Focal length in px scales with image width."""
    return FOCAL_LENGTH_PX * width_px / CALIB_WIDTH_PX


def get_distance_mm(frame, tag_size_mm=TAG_SIZE_MM):
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = measure_tag(gray, _detector)
    if result is None:
        return None
    tag_width_px = result[3]
    focal_px = focal_for_width(gray.shape[1])
    return focal_px * tag_size_mm / tag_width_px


if __name__ == "__main__":
    frame = cv2.imread("get_dist.png")
    distance = get_distance_mm(frame)
    if distance is None:
        print("No marker detected.")
    else:
        print(f"Distance: {distance:.1f} mm")
