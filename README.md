**현재 작업 중** <br>

1. Path 라이브러리를 사용해 보았다.
2. skimage 라이브러리를 사용해 보았다.
    2.1. 전처리 과정에서는 이미지의 채널 변환을 위해 cv2 라이브러리를 일단 사용했다.
    그러나 PIL, skimage 라이브러리와 달리 cv2 라이브러리는 RGB의 채널 순서가 아닌 BGR의 채널 순서로 이미지를 처리한다.
     PyTorch는 RGB 채널 순서를 기대한다. 그러므로, 그러므로, cv2 라이브러리를 사용해서 전처리를 수행하고 그대로 전처리된 이미지를 저장하여 학습 시 채널 순서를 다시 바꿔주는 것은 귀찮으니, 최종 이미지를 저장할 때는 skimage 라이브러리를 사용해 보았다.
3. 정적 메소드를 사용해 보았다.


cf. config 부분을 argument parsing으로 대체할 예정이다. <br>
cf. Logger를 달아 볼 예정이다.