# pyautogui 라이브러리를 활용한 코드
# pyauto에 비해서 기능은 많은데 마우스 이동하는 게 딜레이가 심함

import cv2  # OpenCV 라이브러리
import numpy as np  # Numpy 라이브러리
import HandTrakingModule as htm  # 손 추적 모듈
import time  # 시간 관련 기능
import pyautogui  # 화면 제어 라이브러리
import threading  # 스레드 실행 라이브러리

# 변수 설정
# 카메라 설정 및 해상도 가져오기
cap = cv2.VideoCapture(0)
# wCam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 카메라의 실제 가로 해상도 가져오기
# hCam = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 카메라의 실제 세로 해상도 가져오기
wCam = 640
hCam = 480
wScr, hScr = pyautogui.size()  # 화면 해상도 가져오기

cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)  # 창 가로 길이
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)  # 창 세로 길이

frameW = wCam // 2  # 창 가로 크기
frameH = hCam // 2  # 창 세로 크기

centerX, centerY = wCam // 2, (hCam // 2 - (hCam//5))  # 화면의 중앙
startX, startY = centerX - frameW // 2, centerY - frameH // 2  # 사각형의 시작점
endX, endY = centerX + frameW // 2, centerY + frameH // 2  # 사각형의 끝점

smoothening = 5  # 부드럽게 이동
plocX, plocY = 0, 0  # 이전 마우스 위치
clocX, clocY = 0, 0  # 현재 마우스 위치

# 손가락 id 값
thumb = 4  # 엄지
indexFig = 8  # 검지
middle = 12  # 중지
ring = 16  # 약지
pinky = 20  # 소지

fps = htm.FPS()  # FPS 계산 객체
detector = htm.HandDetector(maxHands=1)  # 손 탐지 객체 (최대 1손)

# 좌클릭 함수 (스레드로 실행)
def left_click():
    pyautogui.click(interval=1)  # 좌클릭 실행
    print('Left click')

# 페이지 다운 함수 (스레드로 실행)
def page_down():
    pyautogui.press('pagedown', interval=1)  # 페이지 다운 키 입력
    print('Page down')

# 페이지 업 함수 (스레드로 실행)
def page_up():
    pyautogui.press('pageup', interval=1)  # 페이지 업 키 입력
    print('Page up')

while True:
    # 1. find hand landmarks
    success, img = cap.read()  # 웹캠에서 프레임 읽기
    img = detector.findHands(img)  # 손 찾기
    lmList, bbox = detector.findpostion(img, blue=0, green=0, red=150)  # 손 위치 찾기

    # 2. get the tip of the index and middle fingers
    if len(lmList) != 0:  # 손이 인식되었으면
        x1, y1 = lmList[indexFig][1:]  # 검지 끝 부분 좌표
        x2, y2 = lmList[middle][1:]  # 중지 끝 부분 좌표

        # 3. check which fingers are up
        fingers = detector.fingersUp()  # 손가락 상태 체크

        # 중앙에 위치한 사각형 그리기
        cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 255), 2)

        # 손가락이 사각형 영역 내에 있는지 확인
        if (startX-10) < x1 < endX and (startY-10) < y1 < endY:
            # 4. only index finger : moving mode, 마우스 포인터 이동
            if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
                x3 = np.interp(x1, (startX, endX), (0, wScr))  # x 좌표 변환
                y3 = np.interp(y1, (startY, endY), (0, hScr))  # y 좌표 변환

                # 5. smoothen values, 마우스 포인터가 흔들리는걸 방지
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 6. move mouse
                try:
                    pyautogui.moveTo(wScr - clocX, clocY)  # 좌우 반전을 주기 위해 wScr - clocX 사용
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)  # 검지 끝부분에 원 그리기
                    plocX, plocY = clocX, clocY  # 이전 위치 업데이트
                except:
                    pass

            # 7. both index and thumb fingers are up : clicking mode, 마우스 좌클릭
            if fingers[1] == 1 and fingers[0] == 1:
                length, img, lineInfo = detector.findDistance(indexFig, thumb, img)  # 손가락 사이 거리 계산

                if length < 35:  # 거리 짧으면 좌클릭 실행
                    cv2.circle(img, (lineInfo[-2], lineInfo[-1]), 15, (0, 255, 0), cv2.FILLED)  # 손가락 사이에 원 그리기
                    threading.Thread(target=left_click).start()  # 좌클릭 스레드 실행

            # 8. 페이지 업과 다운 기능
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(indexFig, middle, img)  # 손가락 사이 거리 계산

                if 60 < length < 150:  # 거리 범위로 페이지 다운 실행
                    cv2.circle(img, (lineInfo[-2], lineInfo[-1]), 15, (0, 255, 0), cv2.FILLED)  # 손가락 사이에 원 그리기
                    threading.Thread(target=page_down).start()  # 페이지 다운 스레드 실행
                    
                if length > 150:  # 거리 길면 페이지 업 실행
                    cv2.circle(img, (lineInfo[-2], lineInfo[-1]), 15, (255, 0, 0), cv2.FILLED)  # 손가락 사이에 원 그리기
                    threading.Thread(target=page_up).start()  # 페이지 업 스레드 실행
        else:
            print("Hand outside the rectangle")  # 디버깅용 메시지

    # 9. frame rate
    fps.get_fps(img, blue=200, green=0, red=0)  # FPS 표시

    # 10. display
    if not success:  # 카메라에서 프레임을 읽을 수 없으면 종료
        break

    cv2.imshow("camera", img)  # 이미지 출력
    cv2.resizeWindow("camera", wCam, hCam)

    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break
    if cv2.getWindowProperty("camera", cv2.WND_PROP_VISIBLE) < 1:  # X 버튼으로 종료
        break

cap.release()  # 카메라 자원 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
