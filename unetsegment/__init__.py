from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import pytesseract
import re
import math
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler


def get_prediction(imagep, modelp, encoding="Image"):
    """
    Predicts the scale bar details and the binary image from the given image using the provided model.

    This function reads an image, converts it to grayscale, and uses OCR to extract text from the image.
    It then finds contours in the binary image and calculates the scale bar details. The function also
    predicts the binary image using the provided model.

    Parameters:
    imagep (str): The path to the input image.
    modelp (str or keras.Model): The path to the model or the model itself.
    encoding (str, optional): The type of encoding for the output image. It can be either "Image" (PIL Image) or "ndarray" (numpy ndarray). Defaults to "Image".

    Returns:
    tuple: A tuple containing the following elements:
        - new_pred (PIL.Image.Image or numpy.ndarray): The predicted binary image.
        - img_in (numpy.ndarray): The original input image.
        - um_per_pixel (float): The micrometers per pixel value calculated from the scale bar.
        - error_string (str): A string containing error messages if any errors occurred during the process.

    Raises:
    ValueError: If the encoding is not either "Image" or "ndarray".
    """
    if type(modelp) == str:
        model = load_model(modelp)  # load the model
    else:
        model = modelp

    img_in = cv2.imread(imagep)
    gray_image = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    ocr_result = pytesseract.image_to_string(img_in, config="-l eng --psm 11")
    _, binary_image = cv2.threshold(
        gray_image, 250, 255, cv2.THRESH_BINARY
    )  # Use thresholding to get a binary image
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Find contours in the binary image
    error_string = ""

    # Initialize variables to store the scale bar details
    scale_bar_text = None
    scale_bar_position = None
    w_max = 0
    um_per_pixel = None

    if len(contours) > 0:
        # Iterate over the contours to find the scale bar
        for contour in contours:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            if w > w_max:
                w_max = w
                scale_bar_position = (x, y, w, h)
        if "um" not in ocr_result:
            x, y, w, h = scale_bar_position
            y_bounds = [max(y - w // 2, 0), max(y + 3 * w // 2, 0)]
            x_bounds = [max(x - w // 2, 0), max(x + 3 * w // 2, 0)]
            ocr_image = img_in[y_bounds[0] : y_bounds[1], x_bounds[0] : x_bounds[1]]
            ocr_result = pytesseract.image_to_string(ocr_image, config="-l eng --psm 6")
        if "um" in ocr_result:
            scale_bar_text = ocr_result.replace("\n", "").replace(" ", "")
            scale_bar_text = re.findall(r"(\d+)um", scale_bar_text)
            if scale_bar_text:
                scale_bar_text = scale_bar_text[0]
                if int(scale_bar_text) > 0:
                    um_per_pixel = int(scale_bar_text) / w_max

    if um_per_pixel == None:
        error_string += f"\nNo scale bar found in the image: {imagep}."

    orig_size = gray_image.shape[::-1]
    img = Image.fromarray(gray_image)
    img = img.resize((512, 512))
    img = np.array(img)[np.newaxis, ..., np.newaxis]
    img = img / 255

    pred = model.predict(img, verbose=0)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    pred = (pred[0, ..., 0]) * 255
    pred = pred.astype("uint8")
    if encoding == "Image":
        new_pred = Image.fromarray(pred).resize(orig_size, resample=Image.NEAREST)
    elif encoding == "ndarray":
        pred.resize(orig_size)
        new_pred = pred
    else:
        error_string += '\nOnly "Image" (PIL Image) and "ndarray" (numpy ndarray) encoding supported'
    return new_pred, img_in, um_per_pixel, error_string


def get_metrics(prediction):
    """
    Calculate the metrics of the largest contour in a binary image.

    This function finds contours in the input binary image, sorts them by area in descending order,
    and calculates the area, width, height, and the top-left coordinates (x1, y1) of the bounding rectangle
    of the largest contour.

    Parameters:
    prediction (numpy.ndarray): A binary image where the contours are to be found.

    Returns:
    tuple: A tuple containing the following elements:
        - area (float): The area of the largest contour.
        - width (int): The width of the bounding rectangle of the largest contour.
        - height (int): The height of the bounding rectangle of the largest contour.
        - x1 (int): The x-coordinate of the top-left corner of the bounding rectangle of the largest contour.
        - y1 (int): The y-coordinate of the top-left corner of the bounding rectangle of the largest contour.
        - contour (numpy.ndarray): The largest contour.

    Note:
    If no contours are found in the image, all output values except 'contour' will be None, and 'contour' will be an empty list.
    """
    area, width, height, x1, y1 = [
        None,
    ] * 5
    contours = cv2.findContours(
        np.array(prediction), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # find contours in image
    if np.all(contours[1]) != None:
        contours = contours[0] if len(contours) == 2 else contours[1]
        sorted_contours = sorted(
            contours, key=cv2.contourArea, reverse=True
        )  # sort contours by area
        area = cv2.contourArea(sorted_contours[0])  # get the area
        x1, y1, width, height = cv2.boundingRect(
            sorted_contours[0]
        )  # get the width and height
    return area, width, height, x1, y1, sorted_contours[0]


def get_baseline(img_in, prediction, contour, x1, y1, w, h):
    """
    Calculates the baseline of the melt pool in the image.

    This function takes an image and a prediction mask, and calculates the baseline of the melt pool.
    It uses edge detection and regression to find the interface line, and identifies the common points between
    the melt pool boundary and the interface.

    Parameters:
    img_in (numpy.ndarray): The input image.
    prediction (PIL.Image.Image): The prediction mask.
    contour (numpy.ndarray): The contour of the melt pool.
    x1, y1, w, h (int): The x and y coordinates, width, and height of the bounding rectangle of the melt pool.

    Returns:
    tuple: A tuple containing the following elements:
        - y_split (int or None): The y-coordinate of the split between the top and bottom of the melt pool.
        - common_points (numpy.ndarray or None): The common points between the melt pool boundary and the interface.
        - model_dict (dict): A dictionary containing the regression model and the scalers used for standardization.
    """
    # make scalebar dark
    image = np.where(img_in == 255, 0, img_in)
    # deal with case where top half of image is bright
    if np.mean(image[: image.shape[0] // 2, :]) > 150:
        image[: np.argmin(np.mean(image, axis=1)), :] = np.array(
            image.shape[1]
            * [
                min(np.mean(image, axis=1))
                + -10 * np.random.rand(np.argmin(np.mean(image, axis=1))),
            ]
        ).T
    # convert prediction from PIL.Image to np.array containing 0s and 1s
    pred_mask = np.array(prediction, dtype=np.uint8) // 255
    # remove the mask region from the image and blur it
    kernel_size = min(img_in.shape) // 150 + 1 * (min(img_in.shape) // 150 + 1) % 2
    blurred_image = cv2.GaussianBlur(
        np.where(pred_mask == 1, np.nan, image), (kernel_size, kernel_size), 0
    )
    # detect edges in the image using Y-gradient Sobel filter
    edges = cv2.Sobel(
        src=blurred_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5, scale=0.1
    )
    abs_grad_x = cv2.convertScaleAbs(edges)
    threshold = 75
    filter = abs_grad_x > threshold
    # ideally, should be the top interface of the sample
    indices = np.argmax(filter, axis=0)
    # get the coordinates of the interface
    pixel_coordinates = np.stack((np.arange(len(indices)), indices), axis=-1)
    pixel_coordinates = pixel_coordinates.astype(int)
    # get coordinates of the melt pool boundary
    contour_coords = contour.reshape(-1, 2)
    contour_coords = contour_coords.astype(int)
    # create a boolean mask for the length of the x-axis. Make the region of the melt pool False
    mask = np.ones(len(indices), dtype=bool)
    mask[x1 : x1 + w + 1] = False

    # find the common points between the melt pool boundary and the interface
    common_points = []
    for point1 in pixel_coordinates:
        for point2 in contour_coords:
            # check if the points are close to each other
            if (
                abs(point1[0] - point2[0]) < img_in.shape[1] // 200
                and abs(point1[1] - point2[1]) < img_in.shape[0] // 100
            ):
                common_points.append(point1)
    common_points = np.unique(np.array(common_points), axis=0)
    min_x = np.min(common_points[:, 0])
    max_x = np.max(common_points[:, 0])
    if max_x - min_x > w // 4:
        min_x_mean_y = int(np.mean(common_points[common_points[:, 0] == min_x][:, 1]))
        max_x_mean_y = int(np.mean(common_points[common_points[:, 0] == max_x][:, 1]))
        common_points = np.array([[min_x, min_x_mean_y], [max_x, max_x_mean_y]])
    else:
        common_points = None

    near_interface = np.where(mask.reshape(-1, 1) == 0, np.nan, pixel_coordinates[:])
    near_interface = near_interface[~np.isnan(near_interface[:, 1])]
    inter_y = near_interface[:, 1]
    inter_x = near_interface[:, 0]

    # standardize
    inter_x_scaler, inter_y_scaler = StandardScaler(), StandardScaler()
    inter_x_train = inter_x_scaler.fit_transform(inter_x[..., None])
    inter_y_train = inter_y_scaler.fit_transform(inter_y[..., None])

    # fit model
    inter_model = HuberRegressor(epsilon=1)
    inter_model.fit(inter_x_train, inter_y_train.ravel())

    y_split = None
    slope = None
    intercept = None
    if common_points is not None:
        inter_test_x = common_points[:, 0].reshape(-1, 1)
        inter_test_y = common_points[:, 1].reshape(-1, 1)
        inter_predictions = inter_y_scaler.inverse_transform(
            inter_model.predict(inter_x_scaler.transform(inter_test_x)).reshape(-1, 1)
        )
        if np.all(np.abs(inter_predictions - inter_test_x) < h // 50):
            y_split = int(
                inter_y_scaler.inverse_transform(
                    inter_model.predict(
                        inter_x_scaler.transform([[x1 + w // 2]])
                    ).reshape(-1, 1)
                )[0, 0]
            )
            y_at_01 = inter_y_scaler.inverse_transform(
                inter_model.predict(inter_x_scaler.transform([[0], [1]])).reshape(-1, 1)
            ).ravel()
            slope = y_at_01[1] - y_at_01[0]
            intercept = y_at_01[0]
        else:
            y_split = int(common_points[:, 1].mean())
    # organise all model information
    model_dict = {
        "model": inter_model,
        "x_scaler": inter_x_scaler,
        "y_scaler": inter_y_scaler,
        "slope": slope,
        "intercept": intercept,
    }

    return y_split, common_points, model_dict


def get_mask(image_path, model):
    """
    Generates a prediction mask and calculates the metrics of the melt pool.

    This function takes an image path and a model, generates a prediction mask for the image, and calculates
    the metrics of the melt pool, including the width, height, and scale.

    Parameters:
    image_path (str): The path to the input image.
    model (keras.Model): The model used for prediction.

    Returns:
    tuple: A tuple containing the following elements:
        - prediction (PIL.Image.Image): The prediction mask.
        - metrics (tuple): A tuple containing the metrics of the melt pool, including the width, height, and scale.
    """
    # Step 1: Get the prediction mask and scale
    prediction, raw_image, um_per_pixel, errors = get_prediction(image_path, model)
    # Step 2: Get the metrics of the melt pool
    area, width, height, x1, y1, contour = get_metrics(prediction)
    # Step 3: Get the baseline of the melt pool
    y_split, end_points, baseline_model_dict = get_baseline(
        cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY),
        prediction,
        contour,
        x1,
        y1,
        width,
        height,
    )
    # Step 4: Split the height into above and below the baseline
    if y_split is not None:
        ha = y_split - y1
        hb = height - ha
    else:
        ha = height
        hb = None
    # Step 5: Get the angles of the melt pool
    alpha_l, beta_l, alpha_r, beta_r, line_params = get_angles(
        contour, baseline_model_dict, end_points
    )
    # Step 6: Build image with angle lines (not yet ready to be displayed or saved)
    # angle_image = make_angle_image(raw_image, line_params)
    w, alpha, beta, scale = (
        width,
        np.mean([alpha_l, alpha_r]),
        np.mean([beta_l, beta_r]),
        um_per_pixel,
    )
    return prediction, (w, ha, hb, alpha, beta, scale)


def get_tangent(point_index, blob_contour, direction, window=200):
    """
    Calculate the tangent of the melt pool boundary at a given point.

    This function calculates the tangent of the melt pool boundary at a given point
    by fitting a line to the points in a window around the given point.

    Parameters:
    point_index (int): The index of the point on the contour.
    blob_contour (numpy.ndarray): The contour of the melt pool.
    window (int, optional): The size of the window around the point. Defaults to 200.

    Returns:
    tuple: A tuple containing the following elements:
        - slope (float): The slope of the tangent line.
        - intercept (float): The y-intercept of the tangent line.
    """
    multiplier = 1 if point_index < blob_contour.shape[0] // 2 else -1
    # print(multiplier)
    if direction == "upper":
        multiplier = -1 * multiplier
    prev_point = blob_contour[(point_index) % len(blob_contour)]
    next_point = blob_contour[
        (point_index + multiplier * window // 20) % len(blob_contour)
    ]
    slope = (next_point[1] - prev_point[1]) / (next_point[0] - prev_point[0])
    intercept = prev_point[1] - slope * prev_point[0]
    return slope, intercept


def angle_between_lines(slope_baseline, slope_tangent):
    tan_theta = (slope_tangent - slope_baseline) / (1 + slope_tangent * slope_baseline)
    theta = math.atan(tan_theta)
    angle = math.degrees(theta) % 180
    return angle


def make_angle_image(img_in, line_params):
    """
    Overlay lines on an image based on provided line parameters.

    Parameters:
    img_in (np.array): The input image on which to overlay the lines.
    line_params (dict): A dictionary containing the slopes and intercepts of the lines to be drawn. The keys should be 'slope_base', 'intercept_base', 'slope_alpha_l', 'intercept_alpha_l', 'slope_beta_l', 'intercept_beta_l', 'slope_alpha_r', 'intercept_alpha_r', 'slope_beta_r', and 'intercept_beta_r'.

    Returns:
    np.array: The input image with the lines overlaid.
    """
    img_overlay = img_in

    x = np.array([0, img_overlay.shape[1]])
    y_base = line_params["slope_base"] * x + line_params["intercept_base"]
    y_a_l = line_params["slope_alpha_l"] * x + line_params["intercept_alpha_l"]
    y_b_l = line_params["slope_beta_l"] * x + line_params["intercept_beta_l"]
    y_a_r = line_params["slope_alpha_r"] * x + line_params["intercept_alpha_r"]
    y_b_r = line_params["slope_beta_r"] * x + line_params["intercept_beta_r"]

    cv2.line(
        img_overlay, (x[0], int(y_base[0])), (x[1], int(y_base[1])), (255, 0, 0), 4
    )  # red line
    cv2.line(
        img_overlay, (x[0], int(y_a_l[0])), (x[1], int(y_a_l[1])), (0, 255, 0), 4
    )  # cyan alpha line
    cv2.line(
        img_overlay, (x[0], int(y_b_l[0])), (x[1], int(y_b_l[1])), (0, 0, 255), 4
    )  # magenta beta line
    cv2.line(
        img_overlay, (x[0], int(y_a_r[0])), (x[1], int(y_a_r[1])), (0, 255, 0), 4
    )  # green alpha line
    cv2.line(
        img_overlay, (x[0], int(y_b_r[0])), (x[1], int(y_b_r[1])), (0, 0, 255), 4
    )  # blue beta line

    return img_overlay


def get_angles(contour, baseline_model_dict, end_points, window=200):
    """
    Calculate the angles between the baseline and the tangent lines at the endpoints of a contour.

    Parameters:
    contour (np.array): The contour for which to calculate the angles.
    baseline_model_dict (dict): A dictionary containing the slope and intercept of the baseline. If either the slope or intercept is None, the function will use the end_points to calculate them.
    end_points (np.array): A 2D array containing the x and y coordinates of the endpoints of the contour. If the baseline_model_dict does not contain valid slope and intercept, these points will be used to calculate them.
    window (int, optional): The size of the window to use when calculating the tangent lines. Default is 200.

    Returns:
    tuple: A tuple containing the following elements:
        - alpha_l (float): The angle between the baseline and the upper tangent line at the left endpoint.
        - beta_l (float): The angle between the baseline and the lower tangent line at the left endpoint.
        - alpha_r (float): The angle between the baseline and the upper tangent line at the right endpoint.
        - beta_r (float): The angle between the baseline and the lower tangent line at the right endpoint.
        - line_params (dict): A dictionary containing the slopes and intercepts of all lines calculated in the function.
    """
    blob_contour = contour[:, 0, :]

    # Red line
    if (
        baseline_model_dict["slope"] is None or baseline_model_dict["intercept"] is None
    ) and end_points is None:
        print("No baseline info found")
        # print("Mode 1: No baseline info found")
        return None, None, None, None, None
    elif (
        baseline_model_dict["slope"] is None or baseline_model_dict["intercept"] is None
    ) and end_points is not None:
        slope_base, intercept_base = np.polyfit(end_points[:, 0], end_points[:, 1], 1)
        # print("Mode 2: End points used")
    else:
        slope_base, intercept_base = (
            baseline_model_dict["slope"],
            baseline_model_dict["intercept"],
        )
        # print("Mode 3: Baseline model used")

    # For left point
    point = end_points[0]
    point_index_l = np.argmin(
        np.sum((blob_contour - point) ** 2, axis=1)
    )  # find the points on the contour closest to the end points

    # alpha line
    slope_alpha_l, intercept_alpha_l = get_tangent(
        point_index_l, blob_contour, "upper", window
    )

    # Beta line
    slope_beta_l, intercept_beta_l = get_tangent(
        point_index_l, blob_contour, "lower", window
    )

    # For right point
    point = end_points[1]
    point_index_r = np.argmin(
        np.sum((blob_contour - point) ** 2, axis=1)
    )  # find the points on the contour closest to the end points

    # alpha line
    slope_alpha_r, intercept_alpha_r = get_tangent(
        point_index_r, blob_contour, "upper", window
    )

    # Beta line
    slope_beta_r, intercept_beta_r = get_tangent(
        point_index_r, blob_contour, "lower", window
    )

    alpha_l = angle_between_lines(slope_alpha_l, slope_base)
    beta_l = angle_between_lines(slope_base, slope_beta_l)
    alpha_r = angle_between_lines(slope_base, slope_alpha_r)
    beta_r = angle_between_lines(slope_beta_r, slope_base)

    line_params = {
        "slope_base": slope_base,
        "intercept_base": intercept_base,
        "slope_alpha_l": slope_alpha_l,
        "intercept_alpha_l": intercept_alpha_l,
        "slope_beta_l": slope_beta_l,
        "intercept_beta_l": intercept_beta_l,
        "slope_alpha_r": slope_alpha_r,
        "intercept_alpha_r": intercept_alpha_r,
        "slope_beta_r": slope_beta_r,
        "intercept_beta_r": intercept_beta_r,
    }

    return alpha_l, beta_l, alpha_r, beta_r, line_params
