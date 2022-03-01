import cv2
import math
import numpy as np


def get_lines(lines_in):
    if cv2.__version__ < '3.0':
        return lines_in[0]
    return [l[0] for l in lines_in]


def process_lines(img, pred_box):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 50
    max_line_gap = 20
    line_image = np.copy(img) * 0

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    _lines = []

    for _line in get_lines(lines):
        _lines.append([(_line[0], _line[1]), (_line[2], _line[3])])

    _lines_x = []
    _lines_y = []
    for line_i in _lines:
        orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
            _lines_y.append(line_i)
        else:
            _lines_x.append(line_i)

    _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

    merged_lines_x = merge_lines_pipeline_2(_lines_x)
    merged_lines_y = merge_lines_pipeline_2(_lines_y)

    merged_lines_all = []
    merged_lines_all.extend(merged_lines_x)
    merged_lines_all.extend(merged_lines_y)

    if len(merged_lines_all) > 1:
        tuple_src, tuple_dst = get_longest_line(merged_lines_all)
    else:
        tuple_src = merged_lines_all[0][0]
        tuple_dst = merged_lines_all[0][1]

    x_init = int(pred_box[0])
    y_init = int(pred_box[3])

    k_x1 = x_init + tuple_src[0]
    k_y1 = y_init - tuple_src[1]

    k_x2 = x_init + tuple_dst[0]
    k_y2 = y_init - tuple_dst[1]

    # k_1 = np.array([k_x1, k_y1])
    # k_2 = np.array([k_x2, k_y2])

    k_1 = np.array([k_x1, k_y2])
    k_2 = np.array([k_x2, k_y1])

    return k_1, k_2


def get_longest_line(merged_lines):
    list_dists = []

    for pos, line in enumerate(merged_lines):
        point_a = np.array([line[0][0], line[0][1]])
        point_b = np.array([line[1][0], line[1][1]])
        distance = np.linalg.norm(point_a - point_b)
        list_dists.append((pos, distance, line))

    tuple_max = max(list_dists, key=lambda t: t[1])

    return tuple_max[2][0], tuple_max[2][1]


def merge_lines_pipeline_2(lines):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 30
    min_angle_to_merge = 30

    for line in lines:
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        group.append(line)
                        create_new_group = False
                        group_updated = True
                        break

            if group_updated:
                break

        if create_new_group:
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                if get_distance(line2, line) < min_distance_to_merge:
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge:
                        new_group.append(line2)
            super_lines.append(new_group)

    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))

    return super_lines_final


def get_distance(line1, line2):
    dist1 = get_distance_point_line(line1[0][0], line1[0][1], line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = get_distance_point_line(line1[1][0], line1[1][1], line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = get_distance_point_line(line2[0][0], line2[0][1], line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = get_distance_point_line(line2[1][0], line2[1][1], line1[0][0], line1[0][1], line1[1][0], line1[1][1])

    return min(dist1,dist2,dist3,dist4)


def get_distance_point_line(px, py, x1, y1, x2, y2):
    LineMag = get_line_magnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = get_line_magnitude(px, py, x1, y1)
        iy = get_line_magnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = get_line_magnitude(px, py, ix, iy)

    return DistancePointLine


def get_line_magnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude


def merge_lines_segments1(lines, use_log=False):
    if len(lines) == 1:
        return lines[0]

    line_i = lines[0]
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))

    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])

    if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
        points = sorted(points, key=lambda point: point[1])
        if use_log:
            print("use y")
    else:
        points = sorted(points, key=lambda point: point[0])

        if use_log:
            print("use x")

    return [points[0], points[len(points)-1]]


def get_endpoints_for_line(points, img):
    extremes = []
    for p in points:
        x = p[0]
        y = p[1]
        n = 0
        n += img[y - 1, x]
        n += img[y - 1, x - 1]
        n += img[y - 1, x + 1]
        n += img[y, x - 1]
        n += img[y, x + 1]
        n += img[y + 1, x]
        n += img[y + 1, x - 1]
        n += img[y + 1, x + 1]
        n /= 255
        if n == 1:
            extremes.append(p)
    return extremes


# def detect_keypoints_for_edge_v1(img, bbox):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     kernel_size = 5
#     blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
#
#     low_threshold = 50
#     high_threshold = 150
#     edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
#
#     rho = 1
#     theta = np.pi / 180
#     threshold = 15
#     min_line_length = 50
#     max_line_gap = 20
#     line_image = np.copy(img) * 0
#
#     lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
#
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
#
#     lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

