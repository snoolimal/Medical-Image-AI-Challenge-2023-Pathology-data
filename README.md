**현재 작업 중!** <br>


**시도해 본 것들**
1. Path 라이브러리
2. skimage 라이브러리 <br>
   전처리 과정에서는 이미지의 채널 변환을 위해 cv2 라이브러리를 사용했다. <br>
   근데 PIL 라이브러리나 skimage 라이브리리와 달리 cv2 라이브러리는 RGB가 아닌 BGR의 채널 순서로 이미지를 처리한다. <br>
    PyTorch는 이미지의 RGB 채널 순서를 기대한다. <br>
   그러므로, cv2 라이브러리를 사용해서 전처리를 하되 저장 시에는 skimage 라이브러리를 사용하여 학습 시 헷갈리지 않도록 채널 순서를 다시 바꾸지 않는다.
3. 정적 메소드


cf. config를 argument parsing으로 대체할 예정이다. <br>
cf. Logger를 달아 볼 예정이다.
