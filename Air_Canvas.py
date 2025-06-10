import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Setup for drawing colors and canvas
bpoints = [deque(maxlen=1024)]
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255  # White canvas

# Flags for modes
draw_mode = True
shapes_mode = False
shape_selected = None

# Shape properties (for draggable shapes
shapes = {
    "rect": {"pos": ((50, 50), (150, 150)), "selected": False, "fixed": False},
    "circle": {"pos": (350, 100, 50), "selected": False, "fixed": False},
    "rectangle": {"pos": ((200, 50), (300, 100)), "selected": False, "fixed": False},
    "arrow": {"pos": ((400, 200), (500, 200)), "selected": False, "fixed": False}
}

# Initialize Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)


# Reset the shapes
def reset_shapes():
    global shapes
    shapes = {
        "rect": {"pos": ((50, 50), (150, 150)), "selected": False, "fixed": False},
        "circle": {"pos": (350, 100, 50), "selected": False, "fixed": False},
        "rectangle": {"pos": ((200, 50), (300, 100)), "selected": False, "fixed": False},
        "arrow": {"pos": ((400, 200), (500, 200)), "selected": False, "fixed": False}
    }


# clear the paint window
def clear_canvas():
    global paintWindow, bpoints
    paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255
    bpoints = [deque(maxlen=1024)]
    reset_shapes()



# Draw buttons for CLEAR, DRAW, and SHAPES
def draw_buttons(frame):
    buttons = [("CLEAR", (40, 1), (140, 65)),
               ("DRAW", (160, 1), (255, 65)),
               ("SHAPES", (275, 1), (370, 65))]

    for label, start, end in buttons:
        cv2.rectangle(frame, start, end, (0, 0, 0), 2)
        cv2.putText(frame, label, (start[0] + 10, start[1] + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Here we made use Cv2 to draw the arrow
def draw_arrow(img, start_point, end_point, color, thickness):
    cv2.arrowedLine(img, start_point, end_point, color, thickness, tipLength=0.2)

def smooth_points(points):
    smoothed = []
    for i in range(len(points)):
        if i == 0:
            smoothed.append(points[i])  # Keep the first point
        elif i == len(points) - 1:
            smoothed.append(points[i])  # Keep the last point
        else:
            #avg_x is calculated by summing the x-coordinates of the previous, current, and next points
            avg_x = (points[i-1][0] + points[i][0] + points[i+1][0]) // 3
            avg_y = (points[i-1][1] + points[i][1] + points[i+1][1]) // 3
            smoothed.append((avg_x, avg_y))
    return smoothed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    draw_buttons(frame)

    # Process hand landmarks
    result = hands.process(framergb)
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # lm.x is horizontal and lm.y is vertical
            # this is done to extract the fingertip

            # frame.shape[1] gives the width of the frame in pixels, and frame.shape[0] gives the height.
            # The expression lm.x * frame.shape[1] scales the normalized x-coordinate
            # to the actual pixel width of the frame, effectively converting the value from a range of [0, 1]
            # to [0, frame width].


            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in handslms.landmark]
            fore_finger = landmarks[8]  # Index finger tip

            # Count raised fingers
            # [8,12,16,20] are the coordinates for fingertips
            raised_fingers = sum(1 for i in [8, 12, 16, 20] if landmarks[i][1] < landmarks[i - 2][1])

            # Button Press Detection
            if fore_finger[1] <= 65 and raised_fingers == 1:
                if 40 <= fore_finger[0] <= 140:  # Clear Button
                    clear_canvas()  # Use the new clear_canvas function
                    draw_mode, shapes_mode = True, False
                elif 160 <= fore_finger[0] <= 255:  # Draw Button
                    draw_mode, shapes_mode = True, False
                    shape_selected = None
                elif 275 <= fore_finger[0] <= 370:  # Shapes Button
                    shapes_mode, draw_mode = True, False
                    shape_selected = None

            # Drawing in Draw Mode
            if draw_mode and raised_fingers == 1 and not shape_selected:
                bpoints[0].appendleft(fore_finger)

            # Shape Selection and Dragging
            if shapes_mode and raised_fingers == 1:
                if shape_selected is None:
                    # Check for shape selection

                    # This makes sure that forefinger in the Rectangle
                    #         top left
                    if (shapes["rect"]["pos"][0][0] <= fore_finger[0] <= shapes["rect"]["pos"][1][0] and
                            shapes["rect"]["pos"][0][1] <= fore_finger[1] <= shapes["rect"]["pos"][1][1] and
                            not shapes["rect"]["fixed"]):
                        shape_selected = "rect"

                    elif ((fore_finger[0] - shapes["circle"]["pos"][0]) ** 2 +
                          (fore_finger[1] - shapes["circle"]["pos"][1]) ** 2 <= shapes["circle"]["pos"][2] ** 2 and
                          not shapes["circle"]["fixed"]):
                        shape_selected = "circle"

                    elif (shapes["rectangle"]["pos"][0][0] <= fore_finger[0] <= shapes["rectangle"]["pos"][1][0] and
                          shapes["rectangle"]["pos"][0][1] <= fore_finger[1] <= shapes["rectangle"]["pos"][1][1] and
                          not shapes["rectangle"]["fixed"]):
                        shape_selected = "rectangle"

                    elif (shapes["arrow"]["pos"][0][0] - 10 <= fore_finger[0] <= shapes["arrow"]["pos"][1][0] + 10 and
                          shapes["arrow"]["pos"][0][1] - 10 <= fore_finger[1] <= shapes["arrow"]["pos"][1][1] + 10 and
                          not shapes["arrow"]["fixed"]):
                        shape_selected = "arrow"

                # Move selected shape
                if shape_selected == "rect":
                    shapes["rect"]["pos"] = ((fore_finger[0] - 50, fore_finger[1] - 50),
                                           (fore_finger[0] + 50, fore_finger[1] + 50))

                elif shape_selected == "circle":
                    shapes["circle"]["pos"] = (fore_finger[0], fore_finger[1], shapes["circle"]["pos"][2])

                elif shape_selected == "rectangle":
                    shapes["rectangle"]["pos"] = ((fore_finger[0] - 50, fore_finger[1] - 25),
                                                (fore_finger[0] + 50, fore_finger[1] + 25))

                elif shape_selected == "arrow":
                    # Move arrow while maintaining its length and direction
                    start_x, start_y = shapes["arrow"]["pos"][0]
                    end_x, end_y = shapes["arrow"]["pos"][1]
                    dx = end_x - start_x
                    dy = end_y - start_y
                    shapes["arrow"]["pos"] = ((fore_finger[0], fore_finger[1]),
                                            (fore_finger[0] + dx, fore_finger[1] + dy))

            # Fix the shape if fingers are released
            if raised_fingers == 0 and shape_selected:
                shapes[shape_selected]["fixed"] = True
                shape_selected = None

    # Draw lines in Draw mode on both paint window and frame
    if draw_mode:
        smoothed_points = smooth_points(list(bpoints[0]))
        for j in range(len(smoothed_points)):
            for k in range(1, len(smoothed_points)):
                if smoothed_points[k - 1] is None or smoothed_points[k] is None:
                    continue
                cv2.line(paintWindow, smoothed_points[k - 1], smoothed_points[k], (0, 0, 0), thickness=4)
                cv2.line(frame, smoothed_points[k - 1], smoothed_points[k], (0, 0, 0), thickness=4)


    # Draw shapes only when in shapes mode or when fixed
    if shapes_mode or any(shape["fixed"] for shape in shapes.values()):
        # Draw Square
        rect_start, rect_end = shapes["rect"]["pos"]
        cv2.rectangle(frame, rect_start, rect_end, (0, 0, 255), 2)

        # Draw Circle
        cx, cy, radius = shapes["circle"]["pos"]
        cv2.circle(frame, (cx, cy), radius, (0, 0, 255), 2)

        # Draw Rectangle
        rect_start, rect_end = shapes["rectangle"]["pos"]
        cv2.rectangle(frame, rect_start, rect_end, (255, 0, 0), 2)

        # Draw Arrow
        arrow_start, arrow_end = shapes["arrow"]["pos"]
        draw_arrow(frame, arrow_start, arrow_end, (0, 255, 0), 2)

        # Draw shapes in paint window only if they are fixed
        if shapes["rect"]["fixed"]:
            cv2.rectangle(paintWindow, rect_start, rect_end, (0, 0, 255), 2)

        if shapes["circle"]["fixed"]:
            cv2.circle(paintWindow, (cx, cy), radius, (0, 0, 255), 2)

        if shapes["rectangle"]["fixed"]:
            cv2.rectangle(paintWindow, rect_start, rect_end, (255, 0, 0), 2)

        if shapes["arrow"]["fixed"]:
            draw_arrow(paintWindow, arrow_start, arrow_end, (0, 255, 0), 2)

    # Display frames
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == 27:  # Esc key to clear
        clear_canvas()  # Use the new clear_canvas function

# Release webcam and destroy windows
cap.release()
cv2.destroyAllWindows()


# testsigma