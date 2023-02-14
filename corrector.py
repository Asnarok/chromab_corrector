import PIL
import cv2
import matplotlib
import json

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from math import sqrt

files = []

result = {"name": [], "factors": []}

settings = {"file": "", "max_distance": 10, "origin_max_distance": 100, "max_intersect_distance": 10,
            "max_dot_product": 10}


def sift(img):
    # Find the features (i.e. keypoints) and feature descriptors he imagein t
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    # Draw circles to indicate the location of features and the feature's orientation
    img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img, kp


def dilate_and_search(img):
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(img, kernel)
    img, p = sift(gray)
    pt = [[kp.pt[0], kp.pt[1], kp.size] for kp in p]
    return img, pt



def analysis(i, j):
    print("Performing dilated SIFT research...")
    res_r, pti = dilate_and_search(i)
    res_g, ptj = dilate_and_search(j)

    return pti, ptj


def associate(pt1, pt2):
    print("Correlating found features...")
    associations = []
    for p in pt1:
        min_p, min_dist = get_nearest(p, pt2)
        if min_dist <= settings["max_distance"]:
            associations.append([p, min_p])
    return associations


def vectorize(associations):
    print("Vectorizing correlations...")
    v = []  # [[vx, vy], o]
    for p in associations:
        v.append([[(p[0][0] - p[1][0]), (p[0][1] - p[1][1])], p[0][0:2]])
    return v


def get_nearest(p, array):
    min_dist = 100000000000
    min_p = []
    for pt in array:
        d = distance(p, pt)
        if d < min_dist:
            min_p = pt
            min_dist = d
    return min_p, min_dist


def distance(p1, p2):
    return sqrt(pow((p2[0] - p1[0]), 2) + pow((p2[1] - p1[1]), 2))


def draw_vectors(vectorized_data):
    array = []

    for x in vectorized_data:
        array.append([x[1][0], x[1][1], x[0][0] * 20, x[0][1] * 20])

    x = np.linspace(0, width, 100)
    y = []

    a = np.array(array)
    X, Y, V, W = zip(*a)
    fig = plt.figure()
    plt.ylabel('Y-axis')
    plt.xlabel('X-axis')
    ax = plt.gca()
    ax.quiver(X, Y, V, W, angles='xy', scale_units='xy', color=['r', 'b'], scale=1)
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])
    ax.invert_yaxis()
    plt.grid()
    fig.canvas.draw()
    temp_canvas = fig.canvas

    pil_image = PIL.Image.frombytes('RGB', temp_canvas.get_width_height(), temp_canvas.tostring_rgb())
    return pil_image


def scatter(x, y):
    plt.figure()
    plt.ylabel('Y-axis')
    plt.xlabel('X-axis')
    ax = plt.gca()
    ax.scatter(x, y)
    ax.invert_yaxis();
    plt.grid()
    plt.draw()


def length(x, y):
    return sqrt(x * x + y * y)


def compute_origin(vectorized_data):
    print("Computing barycenter...")
    x_intersects = []
    y_intersects = []
    for i in vectorized_data:
        m = i[0][1] / i[0][0]
        b = i[1][1] - m * i[1][0]
        for w in vectorized_data:
            if w != i:
                mw = w[0][1] / w[0][0]
                bw = w[1][1] - mw * w[1][0]
                if m - mw != 0:
                    x_intersect = (bw - b) / (m - mw)

                    y_intersect = mw * x_intersect + bw

                    adimx = x_intersect / width
                    adimy = y_intersect / width

                    if not np.isnan(x_intersect) and not np.isnan(y_intersect) and length(adimx,
                                                                                          adimy) <= settings[
                        "max_intersect_distance"]:
                        x_intersects.append(x_intersect)
                        y_intersects.append(y_intersect)
    x_mean = np.nanmean(x_intersects)
    y_mean = np.nanmean(y_intersects)
    scatter(x_intersects, y_intersects)
    truncated_x = []
    truncated_y = []
    for i in range(0, len(x_intersects)):
        if sqrt(pow(x_mean - x_intersects[i], 2) + pow(y_mean - y_intersects[i], 2)) < settings["origin_max_distance"]:
            truncated_x.append(x_intersects[i])
            truncated_y.append(y_intersects[i])
    if len(truncated_x) > 0 and len(truncated_y) > 0:
        return [np.nanmean(truncated_x), np.nanmean(truncated_y)]
    else:
        return None


def normalize(v):
    n = sqrt(pow(v[0], 2) + pow(v[1], 2))
    return [v[0] / n, v[1] / n]


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1]


def linear_regression(vectorized_data, origin):
    print("Performing linear regression...")
    r = []
    r_y = []
    for i in vectorized_data:
        normalized_vec = normalize(i[0])
        normalized_d_to_o = normalize([i[1][0] - origin[0], i[1][1] - origin[1]])
        dot_prod = dot(normalized_vec, normalized_d_to_o)
        if dot_prod < 0 and abs(-1 - dot_prod) < settings["max_dot_product"]:
            r.append(sqrt(pow((i[1][0] - origin[0]), 2) + pow((i[1][1] - origin[1]), 2)))
            r_y.append(sqrt(pow(i[0][0], 2) + pow(i[0][1], 2)))

    x = np.array(r).reshape((-1, 1))
    y = np.array(r_y)

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print(f"coefficient of determination: {r_sq}")

    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

    fig = plt.figure()
    plt.ylabel('Norm (px)')
    plt.xlabel('Distance to origin (px)')
    ax = plt.gca()

    x_ = np.linspace(0, np.max(r), 2)
    y_ = x_ * model.coef_ + model.intercept_
    ax.scatter(r, r_y)
    ax.plot(x_, y_)

    plt.grid()

    fig.canvas.draw()
    t = fig.canvas

    i = PIL.Image.frombytes('RGB', t.get_width_height(), t.tostring_rgb())

    return model.coef_, i


def resize(img, factor):
    dsize = (int(len(img[0]) * factor), int(len(img) * factor))

    return cv2.resize(img, dsize)
    # resize image


def center_crop(img, target_w, target_h):
    h = len(img)
    w = len(img[0])

    offset_y = int(h / 2 - target_h / 2)  # oé continuer en gros
    offset_x = int(w / 2 - target_w / 2)
    i = img[offset_y:target_h + offset_y, offset_x:target_w + offset_x]

    return i


def perform_correction():
    global width, height

    pil_image = PIL.Image.open(settings["file"]).convert('RGB')
    open_cv_image = np.array(pil_image)
    img = open_cv_image[:, :, ::-1].copy()

    width = len(img[0])
    height = len(img)
    blue, green, red = cv2.split(img)

    ptr, ptb = analysis(red, green) #
    associated = associate(ptr, ptb)
    v = vectorize(associated)
    vector_plot = draw_vectors(v)
    vector_plot.save("vector_plot.png")

    o = compute_origin(v)
    if o is not None:
        print("Found origin : " + str(o))
        rg_expansion_factor, rg_graph = linear_regression(v, o) #compute enlargement factor

        #same for blue channel
        ptg, ptb = analysis(green, blue)
        associated = associate(ptg, ptb)
        v = vectorize(associated)

        gb_expansion_factor, gb_graph = linear_regression(v, o)
        resized_green = resize(green, 1 / (1 + rg_expansion_factor))

        rg_graph.save("rg_plot.png")
        gb_graph.save("gb_plot.png")

        resized_blue = resize(blue, 1 / ((1 + rg_expansion_factor) * (1 + gb_expansion_factor)))

        target_w = len(resized_blue[0])
        target_h = len(resized_blue)

        #crop enlarged channels to fit with the reference green channel
        cropped_green = center_crop(resized_green, target_w, target_h)

        cropped_red = center_crop(red, target_w, target_h)

        merged = cv2.merge([resized_blue, cropped_green, cropped_red])

        cv2.imwrite("corrected.png", merged) #output image

        #for measuring purposes
        result["factors"].append(np.array([rg_expansion_factor, gb_expansion_factor]).tolist())
        with open('results.json', 'w') as convert_file:
            convert_file.write(json.dumps(result))
