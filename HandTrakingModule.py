import cv2
import mediapipe as mp
import time
import numpy as np
import math

cap = cv2.VideoCapture(0)

# 손에 관련된 클래스
class HandDetector():
    def __init__(self, mode = False, maxHands = 2, modelC=1, detectionCon=0.5, trackCon=0.5):
            self.mode = mode
            self.maxHands = maxHands
            self.modelC = modelC
            self.detectionCon = detectionCon
            self.trackCon = trackCon
            
            self.mpHands = mp.solutions.hands
            # 새 버전에서 modelC가 추가 됨, 오류 수정 https://github.com/google/mediapipe/issues/2818
            self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC, self.detectionCon, self.trackCon)
            self.mpDraw = mp.solutions.drawing_utils
            self.tipIds = [4, 8, 12, 16, 20] # 엄지, 검지, 중지, 약지, 소지 끝 부분 id

    # 손을 찾아서 해당 위치를 그림 그려주는 함수
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR을 RGB 순으로 바꾼다
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks) # 손의 위치 좌표 표시
        
        # 손의 위치를 계산하여 표시
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # 손의 좌표를 점으로 표시
        
        return img # return이 있어야 제대로 실행된다.
    
    # 손의 위치를 연산해서 그림 그려주는 함수
    def findpostion(self, img, handNo=0, draw=True, blue=255, green=255, red=255):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm, in enumerate(myHand.landmark):
                # print(id, lm) # 손 위치 id를 x, y, z를 통해 출력
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw: # 위치를 그려줌
                    cv2.circle(img, (cx, cy), 5, (blue, green, red), cv2.FILLED)
            
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lmList, bbox
    
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # totalFingers = fingers.count(1)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        
        return length, img, [x1, y1, x2, y2, cx, cy]

# 현재 프레임을 보여주는 클래스
class FPS():
    def __init__(self, cTime=0, pTime=0):
        self.cTime = cTime
        self.pTime = pTime
        
    def get_fps(self, img, blue=255, green=255, red=255):
        # 프레임 연산
        self.cTime = time.time()
        fps = 1/(self.cTime-self.pTime)
        self.pTime = self.cTime
        
        # 프레임 표시
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (blue, green, red), 3)
        
def main():
    fps = FPS()
    detector = HandDetector()
    
    while True:
        success, img = cap.read()
        detector.findHands(img)
        fps.get_fps(img, blue=100, green=0, red=0)
        lmList = detector.findpostion(img, blue=150, green=200, red=0, draw=False)
        if len(lmList) !=0:
            print(lmList[9]) # 찾을 번호를 입력하면 해당 번호의 위치를 출력
        
        if not success: # 카메라가 없으면 종료
            break
        
        cv2.imshow("camera", img)
        if cv2.waitKey(1) == ord('q'): # 사용자가 q를 입력한 경우 종료
            break

if __name__ == '__main__':
    main()

cap.release()
cv2.destroyAllWindows()

