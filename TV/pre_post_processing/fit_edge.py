import math, warnings
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning


def find_intersection(m1, b1, m2, b2):
    """
    Find the intersection point of two lines given by slopes (m1, m2) and intercepts (b1, b2).
    For a vertical line, the slope is infinite, and the intercept is the x-intercept.
    """

    if m1 == m2:
        raise ValueError("The lines are parallel and do not intersect.")

    if math.isinf(m1):
        x = b1 * 1.0
        y = b2 + m2 * x
    elif math.isinf(m2):
        x = b2 * 1.0
        y = b1 + m1 * x
    else:
        # Calculate x-coordinate of the intersection
        x = (b2 - b1) / (m1 - m2)

        # Calculate y-coordinate of the intersection by plugging x into either line equation
        y = m1 * x + b1

    return x, y


def find_all_intersections(lines):
    """Find the intersections of four lines

    Returns
        Four tuples, each representing the intersections of Line 1 and 2, Line 2 and 3, Line 3 and 4, Line 4 and 1.
    """
    if len(lines) != 4:
        raise ValueError("Exactly 4 lines must be provided.")

    intersections = []

    # Define the specific pairs of lines we are interested in
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for i, j in pairs:
        m1, b1 = lines[i]
        m2, b2 = lines[j]
        try:
            intersection = find_intersection(m1, b1, m2, b2)
            intersections.append(intersection)
        except ValueError as e:
            print(f"Lines {i + 1} and {j + 1} are parallel: {e}")

    return intersections


# noqa is to suppress flake8 warning about using upper case in function names and variable names
def remove_outliers_IQR(vals): # noqa
    """Remove outliers from a 1D array using the IQR method."""
    Q1 = np.percentile(vals, 25) # noqa
    Q3 = np.percentile(vals, 75) # noqa
    IQR = Q3 - Q1 # noqa
    lower_bound = Q1 - 1.5 * IQR # noqa
    upper_bound = Q3 + 1.5 * IQR # noqa
    return (vals >= lower_bound) & (vals <= upper_bound)

def reset_outliers_IQR_median(vals): # noqa
    """After applying IQR method, further restrict to within median +- 1."""
    Q1 = np.percentile(vals, 25) # noqa
    Q3 = np.percentile(vals, 75) # noqa
    IQR = Q3 - Q1 # noqa
    lower_bound = Q1 - 1.5 * IQR # noqa
    upper_bound = Q3 + 1.5 * IQR # noqa
    tmp = (vals >= lower_bound) & (vals <= upper_bound)
    # set bounds to median +-1
    lower_bound = np.median(vals[tmp])-1
    upper_bound = np.median(vals[tmp])+1
    vals[vals<lower_bound]=lower_bound
    vals[vals>upper_bound]=upper_bound
    return vals

def reset_outliers_max(vals, mode):
    if mode=='top':
        tmp=np.max(vals)
        vals[vals<=tmp-2]=tmp-2
    elif mode=='bottom':
        tmp=np.min(vals)
        vals[vals>=tmp+2]=tmp+2
    return vals

def robust_regression(X, y): # noqa
    """
    Try to fit a Huber Regressor first. If it fails, use Linear Regression.

    Parameters:
    X (array-like): Feature matrix (input data).
    y (array-like): Target values (output data).

    Returns:
    model: The fitted model (HuberRegressor or LinearRegression).
    """
    try:
        # Suppress warnings from the Huber Regressor (e.g., ConvergenceWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Attempt to fit Huber Regressor
        huber = HuberRegressor()
        huber.fit(X, y)
        # print("Huber Regressor fit successfully.")
        return huber

    except Exception as e:
        # If any error occurs in fitting Huber Regressor, fall back to Linear Regression
        len (f"Huber Regressor failed with error: {e}") # use len to avoid warnings
        # print("Falling back to Linear Regression.")

        linear = LinearRegression()
        linear.fit(X, y)
        return linear



def reverse_slope_intercept(m, b):
    """Compute the slope and intercept when x and y are reversed."""
    if m == 0:
        new_slope = float('inf')
        new_intercept = b
    else:
        new_slope = 1 / m
        new_intercept = -b / m
    return new_slope, new_intercept


def fit_edge(points, mode='left', plot=True):
    """
    Fits a straight line to either:
    - Leftmost `x` for each unique `y` if mode is 'left'
    - Rightmost `x` for each unique 'y' if mode is 'right'
    - Topmost `y' for each unique `x` if mode is 'top'
    - Bottommost `y' for each unique `x` if mode is 'bottom'

    Removes outliers from the response variable using the Interquartile Range (IQR) method.

    Parameters:
        points (numpy array): nx2 array containing n points with coordinates.
        mode (str): 'left' for leftmost x at each y, 'right' for rightmost x at each y,
                    'top' for topmost y at each x, 'bottom' for bottommost y at each x.
        plot (bool): Whether to display the plot or not.

    Returns:
        slope (float): Slope of the fitted line.
        intercept (float): Intercept of the fitted line.
    """

    # Step 1: Extract the coordinates. y coords are the row indices, x coords are the column indices
    x_coords, y_coords = points[:, 1], points[:, 0]

    if mode == 'left' or mode == 'bottom':
        # Leftmost 'x' for each unique 'y'
        unique_ys = np.unique(y_coords)
        selected_points = []

        for y in unique_ys:
            xs_at_y = x_coords[y_coords == y]
            leftmost_x = np.min(xs_at_y)
            selected_points.append([y, leftmost_x])

        selected_points = np.array(selected_points)

        # Swap x and y so that x is always the independent variable
        X = selected_points[:, 0].reshape(-1, 1) # noqa  # x as independent
        y = selected_points[:, 1]  # y as dependent (even though here it's the response variable)

    elif mode == 'right' or mode == 'top':
        # Rightmost 'x' for each unique 'y'
        unique_ys = np.unique(y_coords)
        selected_points = []

        for y in unique_ys:
            xs_at_y = x_coords[y_coords == y]
            rightmost_x = np.max(xs_at_y)
            selected_points.append([y, rightmost_x])

        selected_points = np.array(selected_points)

        # Swap x and y so that x is always the independent variable
        X = selected_points[:, 0].reshape(-1, 1) # noqa  # x as independent
        y = selected_points[:, 1]  # y as dependent (but here it's the response variable)

    else:
        raise ValueError("Invalid mode. Choose from 'left', 'right', 'top', 'bottom'.")

    # Step 2: Remove outliers

    # def remove_outliers_top_bottom(vals, mode):
    #     # constrain to within a distance of the bounding box
    #     box = cv2.boxPoints(cv2.minAreaRect(points))
    #     ymax = max([x[1] for x in box])
    #     ymin = min([x[1] for x in box])
    #
    #     if mode=='top':
    #         # within 5 px of extreme values
    #         return vals >= ymax-5
    #     elif mode=='bottom':
    #         # within 5 px of extreme values
    #         return vals <= ymin+5

    # Apply the outlier removal mask to the response variable (y or x, depending on the mode)
    if mode in ['left', 'right']:
        outlier_mask = remove_outliers_IQR(X.ravel())
        X = X[outlier_mask] # noqa
        y = y[outlier_mask]
    else:
        # y = reset_outliers_IQR_median(y)
        y = reset_outliers_max(y, mode)

    # Step 3: Fit a straight line using Linear Regression
    if mode in ['left', 'right']:
        model = robust_regression (X, y)
    else:
        model = LinearRegression()
        model.fit(X, y)

    ## not necessary anymore after we turn 90 degrees for left and right
    # if len(np.unique(X))==1:
    #     # a vertical line
    #     slope=float('inf')
    #     intercept=np.unique(X)[0] # X intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    if mode=='left' or mode=='right':
        # turn 90 degrees
        slope_old, intercept_old = slope, intercept
        slope, intercept = reverse_slope_intercept(slope_old, intercept_old)
    else:
        slope_old, intercept_old = None, None


    # Step 4: Plot the results if 'plot' argument is True
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(x_coords, y_coords, color='blue')  # Plot all points

        if mode=='left' or mode=='right':
            ax.scatter(selected_points[:, 1], selected_points[:, 0], color='red')  # Plot selected points
        else:
            ax.scatter(selected_points[:, 0], selected_points[:, 1], color='red')  # Plot selected points

        # Plot the fitted line
        if mode=='left' or mode=='right':
            line_range = np.linspace(X.min(), X.max(), 100)
            line_y = slope_old * line_range + intercept_old
            ax.plot(line_y, line_range, color='green')
        else:
            line_range = np.linspace(X.min(), X.max(), 100)
            line_y = slope * line_range + intercept
            ax.plot(line_range, line_y, color='green')

        # Set the aspect ratio to 1
        ax.set_aspect('equal', 'box')

        # Remove labels and legend
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Suppress warning about missing labels in the legend
        if ax.get_legend():
            ax.get_legend().remove()

        # invert the y axis so that the plot looks like the image
        ax.invert_yaxis()

        plt.show()


    return slope, intercept


def find_top_flat_lines (points, plot=True):
    """
    Fits a horizontal straight line topmost `y' for each unique `x`, defined by top 20th percentile.

    Parameters:
        points (numpy array): nx1x2 array containing n points with coordinates.
        plot (bool): Whether to display the plot or not.

    Returns:
        slope (float): always 0
        intercept (float): Intercept of the fitted line.
    """

    # Step 1: Extract the coordinates. y coords are the row indices, x coords are the column indices
    y_coords, x_coords = points[:, 0], points[:, 1]

    # Topmost 'y' for each unique 'x'
    unique_xs = np.unique(x_coords)
    selected_points = []

    for x in unique_xs:
        ys_at_x = y_coords[x_coords == x]
        # min corresponds to topmost y in the photo
        topmost_y = np.min(ys_at_x)
        selected_points.append([x, topmost_y])

    selected_points = np.array(selected_points)

    X = selected_points[:, 0].reshape(-1, 1) # noqa  # x as independent
    y = selected_points[:, 1]
    intercept = np.quantile(y,0.2) # 20th percentile, which is 20% from the top
    slope = 0

    # optional plot
    if plot:
        fig, ax = plt.subplots()
        ax.scatter(x_coords, y_coords, color='blue', s=10)  # Plot all points

        ax.scatter(selected_points[:, 0], selected_points[:, 1], color='red', s=10)  # Plot selected points

        # Plot the fitted line
        line_range = np.linspace(X.min(), X.max(), 100)
        line_y = slope * line_range + intercept
        ax.plot(line_range, line_y, color='green')
        ax.plot(line_range, line_y+420, color='green')

        # Set the aspect ratio to 1
        ax.set_aspect('equal', 'box')

        # Remove labels and legend
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Suppress warning about missing labels in the legend
        if ax.get_legend():
            ax.get_legend().remove()

        # invert the y-axis so that the plot looks like the image
        ax.invert_yaxis()

        plt.show()

    return slope, intercept
