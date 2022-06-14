import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from math import atan2, cos, sin, sqrt, pi
import cv2 as cv
from typing import Tuple
from unet import UNet
from torchvision import transforms
from utils.data_loading import BasicDataset

MODEL_PATH = "MODEL.pth"  # change this to your path


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(
        BasicDataset.preprocess(full_img, scale_factor, is_mask=False)
    )
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((full_img.size[1], full_img.size[0])),
                transforms.ToTensor(),
            ]
        )

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return (
            F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
        )


def calculate_orientation(pts):
    # Calculate the orientation of the object without Drawing the axis
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

    # Store the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    ## [pca]
    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    return cntr, angle


def get_orientation_of_detected_objects(gray):
    # Do not draw anything
    # Convert image to grayscale
    # gray = img_mask.copy() * 255
    # gray = gray.astype("uint8")

    # Convert image to binary
    _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # Find all the contours in the thresholded image
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):

        # Calculate the area of each contour
        area = cv.contourArea(c)

        # Ignore contours that are too small or too large
        if area < 3700 or 100000 < area:
            continue

        _, angle = calculate_orientation(c)

        rect = cv.minAreaRect(c)
        longest_side_length = max(rect[1])
        center_coordinates = (int(rect[0][0]), int(rect[0][1]))
        return center_coordinates, angle, longest_side_length


net = UNet(n_channels=3, n_classes=2, bilinear=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net.to(device=device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray(
            (np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8)
        )


def get_mask(img):
    return predict_img(net, img, device, scale_factor=1, out_threshold=0.5)


def get_orientation(img: Image) -> Tuple[Tuple[int, int], float, float]:
    """
    Returns the center coordinates and angle of the detected object

    :param img: Image to detect the object in
    :return:
        center_coordinates: Tuple of the center coordinates of the object (x, y)
        angle: Angle of the object in radians
        longest_side_length: Length of the longest side of the object in ints
    """
    predicted_mask = get_mask(img)
    predicted_mask = np.array(mask_to_image(predicted_mask))
    (
        center_coordinates,
        angle,
        longest_side_length,
    ) = get_orientation_of_detected_objects(predicted_mask)
    return center_coordinates, angle, longest_side_length
