# 핸드 트래킹을 사용하는데 필요한 최소한의 코드

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# 손을 인식하여 표현
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#fps 표시
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR을 RGB 순으로 바꾼다
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks) # 손의 위치 좌표 표시
    
    # 손의 위치를 계산하여 표시
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm, in enumerate(handLms.landmark):
                # print(id, lm) # 손 위치 id를 x, y, z를 통해 출력
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                
                if id == 4 or id == 8: # 4, 8번 위치를 표시해준다.
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # 손의 좌표를 점으로 표시
    
    # 프레임 연산
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    # 프레임 표시
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    
    if not success: # 카메라가 없으면 종료
        break
    
    cv2.imshow("camera", img)
    if cv2.waitKey(1) == ord('q'): # 사용자가 q를 입력한 경우 종료
        break

cap.release()
cv2.destroyAllWindows()