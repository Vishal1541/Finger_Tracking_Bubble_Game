import cv2
import time
import numpy as np
from random import randrange


#  X0, Y0 ------------ X1, Y1
#        |            |
#        |            |
#        |            |
#  X3, Y3 ------------ X2, Y2

game_started = False
score_increment = 10
score = 0
game_is_over = False
bubble_blast = True
bubble_start_time = 0
bubble_end_time = 0
bubble_radius = 20
bubble_threshold_time = 5
bubble_threshold_drop_multiplier = 0.9
bubble_X = None
bubble_Y = None

hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None

def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont

def max_contour(contour_list):
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

        return contour_list[max_i]

def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))

    return cv2.bitwise_and(frame, thresh)

def centroid(max_contour):
    moment = cv2.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None

def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)

def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)
    contour_list = contours(hist_mask_image)
    max_cont = max_contour(contour_list)

    cnt_centroid = centroid(max_cont)
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_circles(frame, traverse_point)
        return far_point
    return (-1, -1)

def show_start_game_message(frame):
    start_x0 = 220
    start_y0 = 160
    start_x2 = 300
    start_y2 = 200

    start_coordinates_0 = (start_x0, start_y0)
    start_coordinates_2 = (start_x2, start_y2)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (start_x0, start_y2)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2


    frame = cv2.putText(frame, "START GAME", 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    return frame, start_coordinates_0, start_coordinates_2

def user_starts_the_game(current_finger_point, start_coordinates_0, start_coordinates_2):
    if(current_finger_point == None): return False
    return (start_coordinates_0[0] <= current_finger_point[0] and start_coordinates_2[0] >= current_finger_point[0] 
        and start_coordinates_0[1] <= current_finger_point[1] and start_coordinates_2[1] >= current_finger_point[1])

def ask_to_start_the_game(frame, current_finger_point):
    global game_started
    global score, score_increment
    frame, start_point_0, start_point_2 = show_start_game_message(frame)
    if(user_starts_the_game(current_finger_point, start_point_0, start_point_2)):
        print("New Game Started")
        score = -score_increment
        game_started = True

def display_score(frame):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 40)
    fontScale              = 0.8
    fontColor              = (255,255,255)
    lineType               = 2
    global score

    frame = cv2.putText(frame,str('SCORE: ') + str(score), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    return frame

def create_new_bubble(frame, bubble_X = None, bubble_Y = None):
    global bubble_blast
    global bubble_start_time
    global bubble_threshold_drop_multiplier
    global bubble_threshold_time
    global bubble_radius
    bubble_blast = False
    
    x0, x3 = 60, 60
    x1, x2 = 450, 450
    y0, y1 = 60, 60
    y2, y3 = 250, 250

    if(bubble_X == None or bubble_Y == None):
        X = randrange(x0, x1)
        Y = randrange(y0, y2)
        bubble_threshold_time *= bubble_threshold_drop_multiplier
        bubble_start_time = time.time()

    else: X, Y = bubble_X, bubble_Y

    frame = cv2.circle(frame,(X,Y), bubble_radius, (0,255,0), -1)

    return frame, X, Y

def does_point_lie_inside_circle(X, Y, X_cicle, Y_circle, radius):
    distance_from_radius = (X - X_cicle) * (X - X_cicle) + (Y - Y_circle) * (Y - Y_circle)
    return distance_from_radius <= radius * radius

def check_for_bubble_blast(current_finger_point):
    global bubble_X, bubble_Y, bubble_radius
    if(bubble_X == None or bubble_Y == None): return True
    if(current_finger_point == None): return False
    (finger_X, finger_Y) = current_finger_point
    return does_point_lie_inside_circle(finger_X, finger_Y, bubble_X, bubble_Y, bubble_radius)

def is_game_over():
    global bubble_start_time, bubble_threshold_time, game_is_over
    time_elapsed = time.time() - bubble_start_time
    print("Time elapsed: " + str(time_elapsed))
    game_is_over = time_elapsed > bubble_threshold_time
    return (game_is_over)

def show_end_game_message(frame):
    start_x0 = 220
    start_y0 = 160
    start_x2 = 300
    start_y2 = 170
    global score

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (start_x0, start_y2)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    
    frame = cv2.putText(frame, "GAME OVER", 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    start_x0 = 185
    start_y2 = 220
    bottomLeftCornerOfText = (start_x0, start_y2)

    frame = cv2.putText(frame, "YOUR SCORE: " + str(score), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    return frame


def play_the_game(frame, current_finger_point):
    frame = display_score(frame)
    global bubble_blast, score
    global bubble_X, bubble_Y, game_is_over
    if(game_is_over):
        frame = show_end_game_message(frame)
        return frame
    bubble_blast = check_for_bubble_blast(current_finger_point)
    if(bubble_blast):
        score += 10
        frame = display_score(frame)
        frame, bubble_X, bubble_Y = create_new_bubble(frame)
    else:
        if(is_game_over() == False):
            frame, bubble_X, bubble_Y = create_new_bubble(frame, bubble_X, bubble_Y)
        else:
            frame = show_end_game_message(frame)

    return frame

# X-> 60 450
# Y-> 60 250

def main():
    global hand_hist
    global game_started
    global score
    global bubble_threshold_time

    is_hand_hist_created = False
    url = 'http://192.168.43.1:8080/video'
    capture = cv2.VideoCapture(url)

    while capture.isOpened():
        pressed_key = cv2.waitKey(1)
        _, frame = capture.read()
        frame = cv2.resize(frame,(600,400))
        frame = cv2.flip(frame, 1)
        if pressed_key & 0xFF == ord('z'):
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)

        if is_hand_hist_created:
            current_finger_point = manage_image_opr(frame, hand_hist)
            if(game_started == False):
                print("Game Not started yet")
                ask_to_start_the_game(frame, current_finger_point)
            else:
                print("Game already started")
                frame = play_the_game(frame, current_finger_point)
        else:
            frame = draw_rect(frame)

        cv2.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
