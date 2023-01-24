# 해당 코드를 보기전에 Explanation 파일을 보고 와라 

import cv2
import numpy as np
import HandTrakingModule as htm
from time import *
import autopy

# 변수
wCam, hCam = 640, 480 # 창 가로, 세로 길이 변수
frameR = 100 # Frame Reduction

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam) # 창 가로 길이
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam) # 창 세로 길이

fps = htm.FPS()
detector = htm.HandDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(wScr, hScr)

while True:
    # 1. find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findpostion(img, blue=0, green=0, red=150)
    
    # 2. get the tip of the index and middle fingers
    if len(lmList)!=0:
        x1, y1 = lmList[8][1:] # 검지 끝 부분
        x2, y2 = lmList[12][1:] # 중지 끝 부분
        
        #print(x1, y1, x2, y2)
    
        # 3. check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)
        
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255, 0, 255), 2)
        
        # 4. only index finger : moving mode
        if fingers[1]==1 and fingers[2]==0: # 검지만 올릴 경우
            
            # 5. covert coordicates
            x3 = np.interp(x1, (0, wCam), (0, wScr))
            y3 = np.interp(y1, (0, hCam), (0, hScr))
            
            # 6. smooth values
            
            # 7. move mouse
            try:
                autopy.mouse.move(wScr-x3, y3)
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            except:
                pass
            
        # 8. both index and middle fingers are up : clicking mode
        
        # 9. find distance between fingers
        
        # 10. click mouse if distance short
    
    # 11. frame rate
    fps.get_fps(img, blue=100, green=0, red=0)
    
    # 12. display
    if not success: # 카메라가 없으면 종료
        break
    
    cv2.imshow("camera", img)
    if cv2.waitKey(1) == ord('q'): # 사용자가 q를 입력한 경우 종료
        break

cap.release()
cv2.destroyAllWindows()