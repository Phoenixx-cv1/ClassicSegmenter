import cv2
import numpy as np
import os

# ---------------- CONFIG ----------------
MODE = "color"  # choose "threshold" or "hsv"
TARGET_COLOR = "red"  # only used in hsv mode: red, green, or blue
IMG_PATH = "apples.jpg"
SAVE_ROIS = True
MIN_AREA = 300
# ----------------------------------------

# Folder naming
roi_folder = f"ROIs_{TARGET_COLOR.capitalize()}" if MODE == "color" else "ROIs_Threshold"
if SAVE_ROIS and not os.path.exists(roi_folder):
    os.makedirs(roi_folder)

# Load image
image = cv2.imread(IMG_PATH)
if image is None:
    print(f"❌ Cannot read image: {IMG_PATH}")
    exit()

roi_count = 0


def apply_watershed_segmentation(img, mask):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    img_copy = img.copy()
    markers = cv2.watershed(img_copy, markers)

    # Extract object contours from watershed result
    object_contours = []
    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue
        region = np.uint8(markers == marker_id)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_AREA:
                object_contours.append(cnt)
    return object_contours


def process_threshold(img):
    global roi_count
    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:
            roi_count += 1

            # Draw contour (shape-based)
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

            x, y, w, h = cv2.boundingRect(cnt)
            roi = img[y:y + h, x:x + w]
            if SAVE_ROIS:
                cv2.imwrite(os.path.join(roi_folder, f"threshold_roi_{roi_count}.jpg"), roi)

            label_text= f"#{roi_count}:{int(area)}"

            cv2.putText(output, label_text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    return output, thresh


def get_hsv_mask(img, color):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color == "red":
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 100])
        upper2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    elif color == "green":
        lower = np.array([35, 40, 40])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif color == "blue":
        lower = np.array([100, 150, 0])
        upper = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    else:
        print(f"❌ Invalid color: {color}")
        exit()

    return mask


def process_hsv_with_watershed(img, target_color):
    global roi_count
    output = img.copy()

    mask = get_hsv_mask(img, target_color)
    contours = apply_watershed_segmentation(img, mask)

    for cnt in contours:
        roi_count += 1
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # Draw rectangle
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)

        roi = img[y:y + h, x:x + w]
        if SAVE_ROIS:
            cv2.imwrite(os.path.join(roi_folder, f"{target_color}roi{roi_count}.jpg"), roi)

        label_text=f"#{roi_count}:{int(area)}"

        cv2.putText(output, label_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    return output, mask


# --- Run processing ---
if MODE == "threshold":
    output_img, mask_img = process_threshold(image)
elif MODE == "color":
    output_img, mask_img = process_hsv_with_watershed(image, TARGET_COLOR.lower())
else:
    print("❌ Invalid MODE. Use 'threshold' or 'color'")
    exit()


# Show results
print(f"✅ Total objects detected: {roi_count}")

output_path = os.path.join(roi_folder, "output_image.jpg")
mask_path = os.path.join(roi_folder, "mask_image.jpg")
cv2.imwrite(output_path, output_img)
cv2.imwrite(mask_path, mask_img)
print(f"Saved output and mask to: {roi_folder}")
    
cv2.imshow("Segmented Objects", output_img)
cv2.imshow("Mask", mask_img)
cv2.waitKey(0)
cv2.destroyAllWindows()