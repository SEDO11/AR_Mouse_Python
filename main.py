# 해당 코드를 보기전에 Explanation 파일을 보고 와라

import cv2
import numpy as np
import HandTracking as htm
from time import *
import autopy

# 변수
wCam, hCam = 640, 480 # 창 가로, 세로 길이 변수

cap = cv2.VideoCapture(0)
cap.set(3, wCam) # 창 가로 길이
cap.set(4, hCam) # 창 세로 길이
pTime = 0
detector = htm.hand

while True:
    # 1. find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img)
    
    # 2. get the tip of the index and middle fingers
    
    # 3. check which fingers are up
    
    # 4. only index finger : moving mode
    
    # 5. covert coordicates
    
    # 6. smooth values
    
    # 7. move mouse
    
    # 8. both index and middle fingers are up : clicking mode
    
    # 9. find distance between fingers
    
    # 10. click mouse if distance short
    
    # 11. frame rate
    cTime = time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, 
                (255, 0, 0), 3)
    
    # 12. display
    if not success: # 카메라가 없으면 종료
        break
    
    cv2.imshow("camera", img)
    if cv2.waitKey(1) == ord('q'): # 사용자가 q를 입력한 경우 종료
        break

cap.release()
cv2.destroyAllWindows()