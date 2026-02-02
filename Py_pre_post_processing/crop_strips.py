import cv2, os
import numpy as np
# from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# don't run this line in ipython. It allows us to run python TV/tv_CLS_train.py from the root directory on Linux
import sys
in_ipython = 'get_ipython' in globals()
if not in_ipython:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Py_pre_post_processing.fit_edge import fit_edge, find_all_intersections, find_top_flat_lines


def crop_strips (img, masks, crop_version, strip_width=23, strip_height=420):
    """
    Crop sS strips from the binary masks. Using binary masks as input instead of json files for gt allows the function to be used for both gt and predicted masks.

    fit a line with slope 0 to the top edge
    fit a line with slope 0 to the bottom edge
    reshape the quadrilateral to a rectangle

    Parameters:
        img: array of size [Ly x Lx x nchan]
        masks: an array storing the mask_id of each pixel
        crop_version:
            v6: 6-chan
            v5: add 10px white space between strips
            v4: affine shearing transformation on patch
            v3: perspective transformation on patch 1px wider than v2
            v2: perspective transformation on patch
            v1: perspective transformation on image
        strip_width: width of the strip
        strip_height: height of the strip

    Returns:
        strips (list): List of array of shape (6, strip_height, strip_width) or (3, strip_height, strip_width x 2). The corresponding mask_id should be 255, 254, ...
    """

    # make sure that the last dimension is channel
    assert img.shape[2] == min(img.shape)

    # compute a height to split the image into a top half and a bottom half
    # if we use image_height // 2 instead of sep, the top half may be too small
    y_mask = np.unique(np.argwhere(masks >0)[:, 0])
    if max(y_mask) - min(y_mask) < 800:
        # there is only 1 row. it does not really matter where we cut the image
        sep = max(y_mask) + 50
    else:
        sep = int(np.mean(np.quantile(y_mask, [1 / 4, 3 / 4])))

    top_half = masks[:sep, :]
    bottom_half = masks.copy()
    bottom_half[:sep, :] = 0
    # np.histogram(bottom_half, bins=np.append(np.unique(bottom_half), np.inf))

    # masks_image = Image.new('L', (image_width, image_height), 0) # Create a blank image with the same dimensions
    # draw = ImageDraw.Draw(masks_image)
    # width=[] # to study the distribution of strip widths

    strips=[]

    indices_top=None

    for half in [top_half, bottom_half]:

        ## deprecated
        # # assume they all share the same top edge and bottom edge
        # points = np.argwhere(half > 0)
        # if len(points) == 0:
        #     print("no points found")
        #     break
        # top_edge = find_top_flat_lines(points, plot=plot)
        # # define bottom edge at a fixed distance from the top edge
        # bottom_edge = (0, top_edge[1] + 420)

        indices = np.unique(half)[1:][::-1]  # exclude 0 and be in descending order, e.g. [255, 254, ..., 1]
        # assert that there are even number of indices
        assert len(indices) % 2 == 0

        plot = False  # set to True to plot the transformed strip. takes much memory
        debug = False  # set to True to print debug info

        if len(indices) == 0:
            print("no indices found in this half")
            break

        # verify that indices starts from 255 and decreases by 1 if half is top_half
        if half is top_half:
            indices_top = indices
            assert indices[0] == 255 and all(indices[i] == indices[i - 1] - 1 for i in range(1, len(indices)))
        else:
            # verify that indices starts from 1 less than the last indices_top
            assert indices[0] == indices_top[-1] - 1

        for i in indices:

            if i % 2 == 1:
                points_pair = np.argwhere((masks == i) | (masks == i - 1))

                points = np.argwhere(masks == i)
                strip1 = crop1strip(points_pair, points, img, crop_version, strip_width, strip_height, plot, debug)

                points = np.argwhere(masks == i-1)
                strip2 = crop1strip(points_pair, points, img, crop_version, strip_width, strip_height, plot, debug)

                if crop_version == 5:
                    # add some white space between the two strips put side by side
                    double_strips = np.concatenate((strip1[:,:strip_width-5,:], np.ones((strip_height, 10, 3), dtype=np.uint8) * 255, strip2[:,:strip_width-5,:]), axis=1)
                elif crop_version == 6:
                    # put strip1 and strip2 together top and bottom (420 x 23 x 6)
                    double_strips = np.concatenate((strip1, strip2), axis=-1)
                else:
                    # put strip1 and strip2 together either side by side (420 x 46 x 3)
                    double_strips = np.concatenate((strip1, strip2), axis=1)

                # permute the dimensions to follow the convention of tv_utils
                double_strips = np.transpose(double_strips, (2, 0, 1))

                strips.append(double_strips)

            else:
                continue # since HSV2 strip already processed under if i % 2 == 1


    # study the distribution of strip widths

    # plt.hist(np.ceil(width), bins=[i+0.5 for i in range(15, 27)], color='blue', edgecolor='black')
    # plt.xlabel('Strip width')
    # plt.ylabel('Frequency')
    # plt.show()

    # np.histogram(np.ceil(width), bins=np.append(np.unique(np.ceil(width)), np.inf))

    # masks_image.show()

    return strips


def crop1strip(points_pair, points, img, crop_version, strip_width=23, strip_height=420, plot=False, debug=False):
    # crop a single strip

    # find top edge using points_pair
    top_edge = find_top_flat_lines(points_pair, plot=plot)
    # define bottom edge at a fixed distance from the top edge
    bottom_edge = (0, top_edge[1] + strip_height-1)

    # find left and right edges for each strip
    left_edge = fit_edge(points, mode='left', plot=debug)
    right_edge = fit_edge(points, mode='right', plot=debug)

    # list of 4 tuples of the 4 corners of the quadrilateral: top-left, top-right, bottom-right, bottom-left
    quadrilateral = find_all_intersections([left_edge, top_edge, right_edge, bottom_edge])
    # draw.polygon(quadrilateral, outline=255, fill=255)
    # Convert the quadrilateral list to a numpy array of floats
    quad_pts = np.array(quadrilateral, dtype="float32")
    x, y, w, h = cv2.boundingRect(quad_pts)

    if crop_version <= 2:
        patch = img[y:y + h, x:x + w]
    else:
        # removes black pixels on the right edge but leads to poor classification performance
        # the extra 1 pixel in width is needed to include the right edge b/c boundingRect round down the coordinates
        # no need for to add 1 pixel to height since the top and bottom edges are always integers
        patch = img[y:y + h, x:x + w + 1]

    if debug:
        print(quad_pts)
        print(x, y, w, h)

    # width.append(quadrilateral[1][0] - quadrilateral[0][0])
    # width.append(quadrilateral[2][0] - quadrilateral[3][0])

    if crop_version <= 3:
        src = np.array([
            [quadrilateral[0][0] - x, 0],  # Top-left corner of the output rectangle
            [quadrilateral[1][0] - x, 0],  # Top-right corner
            [quadrilateral[2][0] - x, strip_height - 1],  # Bottom-right corner
            [quadrilateral[3][0] - x, strip_height - 1]  # Bottom-left corner
        ], dtype="float32")

        rect = np.array([
            [0, 0],  # Top-left corner of the output rectangle
            [strip_width - 1, 0],  # Top-right corner
            [strip_width - 1, strip_height - 1],  # Bottom-right corner
            [0, strip_height - 1]  # Bottom-left corner
        ], dtype="float32")

        warped_img = cv2.warpPerspective(patch, cv2.getPerspectiveTransform(src, rect), (strip_width, strip_height))

    else:
        corners = [
            (quadrilateral[0][0] - x, 0),  # Top-left corner of the output rectangle
            (quadrilateral[1][0] - x, 0),  # Top-right corner
            (quadrilateral[2][0] - x, strip_height - 1),  # Bottom-right corner
            (quadrilateral[3][0] - x, strip_height - 1)  # Bottom-left corner
        ]
        warped_img = trapezoid_to_rectangle_with_corners(patch, corners, strip_width, strip_height)

    if plot:
        # plot the original strip
        plt.imshow(patch)
        plt.axis('off')
        plt.show()

        # plot the transformed strip
        plt.imshow(warped_img)  # Display the image
        plt.axis('off')  # Hide axes for better visualization
        plt.show()

        # # plot the original image with the quadrilateral
        # # Find the bounding box of the quadrilateral (minimum enclosing rectangle)
        # img_with_quad = img.copy()
        # quad_pts_int = np.array(quadrilateral, dtype="int32") # redefine as int32
        # cv2.polylines(img_with_quad, [quad_pts_int], isClosed=True, color=(0, 255, 0), thickness=2)
        # plt.figure(figsize=(24, 12))  # Increase figure size for better visualization
        # plt.imshow(img_with_quad)
        # plt.axis('off')
        # plt.show()

    return warped_img

def reorder_strip_id (masks):
    """
    Reorder the mask indices so that they start from 255 and decrease by 1 if half is top_half.
    """

    masks_cpy = masks.copy()

    # compute a height to split the image into a top half and a bottom half
    # if we use image_height // 2 instead of sep, the top half may be too small
    y_mask = np.unique(np.argwhere(masks >0)[:, 0])

    if len(y_mask) == 0:
        return None

    if max(y_mask) - min(y_mask) < 800:
        # there is only 1 row. it does not really matter where we cut the image
        sep = max(y_mask) + 50
    else:
        sep = int(np.mean(np.quantile(y_mask, [1 / 4, 3 / 4])))

    top_half = masks[:sep, :]
    # np.histogram(top_half, bins=np.append(np.unique(top_half), np.inf))
    bottom_half = masks.copy()
    bottom_half[:sep, :] = 0
    # np.histogram(bottom_half, bins=np.append(np.unique(bottom_half), np.inf))

    indices_new = None
    for half in [top_half, bottom_half]:

        indices = np.unique(half)[1:][::-1]  # exclude 0 and be in descending order, e.g. [255, 254, ..., 1]
        # np.histogram(half, bins=np.append(np.unique(half), np.inf))

        if len(indices) == 0:
            print("no indices found in this half")
            break

        # get the mean of x coordinate of the points in half whose values equal to index
        x_points = [np.mean(np.argwhere(half == index)[:, 1]) for index in indices]
        # get the position of each item in x_points in a new sorted x_points (ascending order)
        order = [sorted(x_points).index(x) for x in x_points]

        if half is top_half:
            # create a new set of indices starting from 255 and decreasing by 1 from left to right based on the order of the mean of the x coordinates of the strip masks
            # create a list of the same length as x_points starting at 255 and decreasing by 1
            indices_new = [255 - i for i in range(len(x_points))]
        else:
            indices_new = [indices_new[-1] - i - 1 for i in range(len(x_points))]

        for i, index in enumerate(indices):
            masks_cpy[masks == index] = indices_new[order[i]]

    return masks_cpy



def trapezoid_to_rectangle_with_corners(patch, corners, strip_width, strip_height):
    """
    Transforms a trapezoid specified by its corners into a rectangle with the same height by scaling each row.

    Args:
        patch (numpy.ndarray): The source patch containing the trapezoid.
        corners (list of tuples): List of coordinates of the trapezoid corners [(top_left), (top_right), (bottom_left), (bottom_right)].
        strip_width (int): The desired width of the rectangle.
        strip_height (int): The desired height of the rectangle.

    Returns:
        numpy.ndarray: A new patch with the trapezoid transformed into a rectangle.
    """
    # Unpack the corners
    top_left, top_right, bottom_right, bottom_left = corners

    # check that the four corners in corners form a trapezoid
    assert top_left[1] == top_right[1] and bottom_left[1] == bottom_right[1]
    # check that the height of the trapezoid is the same as the target height
    assert bottom_left[1] - top_left[1] == strip_height - 1
    # check that the four corners are in the correct order
    assert top_left[0] < top_right[0] and bottom_left[0] < bottom_right[0]

    # Initialize the output patch with target dimensions
    output_image = np.zeros((strip_height, strip_width, 3), dtype=np.uint8)

    for y in range(strip_height):
        # Calculate the proportional position of the current row
        row_ratio = y / (strip_height - 1)

        # Interpolate the start and end x-coordinates for this row in the trapezoid
        start_x = round(top_left[0] + (bottom_left[0] - top_left[0]) * row_ratio)
        end_x = round(top_right[0] + (bottom_right[0] - top_right[0]) * row_ratio)

        # Crop the current row from the trapezoid in the original patch
        row = patch[y:y + 1, start_x:end_x]

        # Resize the row to fit the strip width
        scaled_row = cv2.resize(row, (strip_width, 1), interpolation=cv2.INTER_LINEAR)

        # Place the scaled row in the output patch
        output_image[y:y + 1, :] = scaled_row

    return output_image



def crop_strips_dS_box (img, masks, strip_width=23, strip_height=420): # noqa
    """
    Crop dS strips from the binary masks of a rectangle shape

    fit a line with slope 0 to the top edge
    fit a line with slope 0 to the bottom edge
    reshape the quadrilateral to a rectangle

    Parameters:
        img: array of size [Ly x Lx x nchan]
        masks: an array storing the mask_id of each pixel
        strip_width: width of the strip
        strip_height: height of the strip

    Returns:
        strips (list): List of array of shape (3, strip_height, strip_width x 2). The corresponding mask_id should be 255, 254, ...
    """

    # make sure that the last dimension is channel
    assert img.shape[2] == min(img.shape)

    strips=[]

    indices = np.unique(masks)[1:][::-1]  # exclude 0 and be in descending order, e.g. [255, 254, ..., 1]

    if len(indices) == 0:
        # raise Exception("no indices found in the mask file")
        print ("no indices found in the mask file")
        return None

    assert indices[0] == 255 and all(indices[i] == indices[i - 1] - 1 for i in range(1, len(indices)))

    for i in indices:

        points = np.argwhere(masks == i)
        # get a bounding box for points
        # note that because points are x,y coordinates, the width is the height of the strip and the height is the width of the strip
        y, x, h, w = cv2.boundingRect(points)
        patch = img[y:y + h, x:x + w]

        # scale the patch to 2*strip_width x strip_height
        double_strips = cv2.resize(patch, (2*strip_width, strip_height), interpolation=cv2.INTER_LINEAR)

        # permute the dimensions to follow the convention of tv_utils
        double_strips = np.transpose(double_strips, (2, 0, 1))

        strips.append(double_strips)

    return strips
