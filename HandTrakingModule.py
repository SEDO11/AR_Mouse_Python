import cv2
import mediapipe as mp
import time

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
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm, in enumerate(myHand.landmark):
                # print(id, lm) # 손 위치 id를 x, y, z를 통해 출력
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw: # 위치를 그려줌
                    cv2.circle(img, (cx, cy), 10, (blue, green, red), cv2.FILLED)
                
        return lmList

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

