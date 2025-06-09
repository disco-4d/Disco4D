from controlnet_aux.util import HWC3
import cv2
import math
import numpy as np
import matplotlib
import cv2


eps = 0.01

src_idxs = [55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8, 60, \
                             61, 62, 63, 64, 65, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, \
                                137, 138, 139, 140, 141, 142, 143, 76, 77, 78, 79, 80, 81, 82, 83, 84, \
                                    85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, \
                                        102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, \
                                            116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 20, 37, 38, 39, \
                                                66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 21, \
                                                    52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]


def draw_bodypose(canvas, candidate, subset):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas


def draw_handpose(canvas, all_hand_peaks):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas



def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = draw_bodypose(canvas, candidate, subset)
    canvas = draw_handpose(canvas, hands)
    # canvas = draw_facepose(canvas, faces)

    return canvas

def get_dwpose_map(out):

    max_ind = 0
    candidate = out['proj_lmk'][src_idxs].unsqueeze(0).detach().cpu().numpy()

    # draw in moore niamte keypoint format
    # _, H, W = out['alpha'].shape
    H, W, _ = out['alpha'].shape

    candidate[..., 0] /= float(W)
    candidate[..., 1] /= float(H)



    # new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    # new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]


    neck = np.mean(candidate[:, [5, 6]], axis=1)

    candidate = np.insert(candidate, 17, neck, axis=1)

    candidate[:, openpose_idx] = candidate[:, mmpose_idx]

    foot = candidate[:, 18:24]
    body = candidate[:, :18]

    faces = candidate[[max_ind], 24:92]

    hands = candidate[[max_ind], 92:113]
    hands = np.vstack([hands, candidate[[max_ind], 113:]]) # 113

    score = np.array([range(18)])

    bodies = dict(candidate=body[0], subset=score)
    pose = dict(bodies=bodies, hands=hands, faces=faces)

    detected_map = draw_pose(pose, H, W)
    detected_map = HWC3(detected_map)

    # img = resize_image(input_image, image_resolution)
    # H, W, C = img.shape

    detected_map = cv2.resize(
        detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    
    return detected_map