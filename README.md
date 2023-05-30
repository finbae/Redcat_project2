# Redcat_project2
### opencv를 이용해 '안녕' 인식 시키기

    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    my_hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    gesture = []
    threshold = 28 # 흔들림을 감지하는 거리 임계값

    while True:
        success, img = cap.read()
        h, w, c = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = my_hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                landmarks = []
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))
                gesture.append(landmarks)

                if len(gesture) > 5:
                    gesture = gesture[-5:]

                if len(gesture) == 5:
                    diff_x = gesture[4][8][0] - gesture[0][8][0]
                    diff_y = gesture[4][8][1] - gesture[0][8][1]
                    distance = (diff_x ** 2 + diff_y ** 2) ** 0.5

                    if distance > threshold:
                        cv2.putText(img, "Hi", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cv2.imshow("HandTracking", img)
        cv2.waitKey(1)
