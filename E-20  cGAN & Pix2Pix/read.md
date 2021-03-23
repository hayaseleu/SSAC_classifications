# 20-13. 프로젝트 : Segmentation map으로 도로 이미지 만들기

## 프로젝트 수행
----------------------------------
프로젝트를 진행하면서 필수로 수행해야 할 사항은 다음과 같습니다.

데이터에 한 가지 이상의 augmentation 방법을 적용하여 학습해주세요.

(어떠한 방법을 사용했는지 적어주세요.)

이전에 구현했던 두 개의 Generator 중 Encoder와 Decoder간에 skip connection이 있는 U-Net Generator를 사용해주세요.

모델 학습 후, 학습된 Generator를 이용해 테스트합니다. 테스트 데이터는 다운받았던 "val" 폴더 내 이미지를 사용해주세요.

1개 이상의 이미지에 대해 테스트 과정을 거친 후 그 결과를 스케치, 생성된 사진, 실제 사진 순서로 나란히 시각화해 주세요.

모델을 충분히 학습하기에 시간이 부족할 수 있습니다. 적어도 10 epoch 이상 학습하며 중간 손실 값에 대한 로그를 남겨주세요. 

좋은 결과를 얻기 위해선 긴 학습 시간이 필요하므로 테스트 결과는 만족스럽지 않아도 괜찮습니다.




데이터 



![Screenshot from 2021-03-23 16-30-32](https://user-images.githubusercontent.com/76804810/112109784-5557d500-8bf5-11eb-9337-718e227922b3.png)



## 데이터 augmentation

``` python
from tensorflow import image
from tensorflow.keras.preprocessing.image import random_rotation

@tf.function() # 빠른 텐서플로 연산을 위해 @tf.function()을 사용합니다. 
def apply_augmentation(sketch, colored):
    stacked = tf.concat([sketch, colored], axis=-1)  # 두 이미지를 채널축으로 연결 ex 각각 3채널일때 6채널
    
    _pad = tf.constant([[30,30],[30,30],[0,0]])
    if tf.random.uniform(()) < .9:                         # 50% 확률로 30픽셀 pad width 만큼 
        padded = tf.pad(stacked, _pad, "REFLECT")            # reflection padding하거나 
    else:
        padded = tf.pad(stacked, _pad, "CONSTANT", constant_values=1.) #아니면 constant padding

    out = image.random_crop(padded, size=[256, 256, 6])       # 거기에 256, 256, 6 크기의 이미지를 임의로 잘라내기 
    
    out = image.random_flip_left_right(out)                # 50% 확률로 가로 뒤집기 
#     out = image.random_flip_up_down(out)                   # 50% 확률로 세로 뒤집기 
    
#     if tf.random.uniform(()) < .5:
#         degree = tf.random.uniform([], minval=1, maxval=4, dtype=tf.int32)
#         out = image.rot90(out, k=degree)                                  # 50% 확률로 90도 회전 
    
    return out[...,:3], out[...,3:]   

print("✅")
```

두 이미지를 채널축으로 연결 ex 각각 3채널일때 6채널

90% 확률로 30픽셀 pad width 만큼 reflection padding하거나 아니면 constant padding함

거기에 256, 256, 6 크기의 이미지를 임의로 잘라내고

50% 확률로 세로 뒤집기를 해줍니다.

가로 뒤집기나 회전을 하지 않는 이유는 데이터들이 고정된 카메라에서 촬영되었기때문에 이미지가 회전될 가능성이 없다고 보았기 때문입니다.



## 결과

![Screenshot from 2021-03-23 16-30-19](https://user-images.githubusercontent.com/76804810/112109789-57219880-8bf5-11eb-878a-a284acb346a6.png)


# 루브릭

|평가문항|상세기준|
|---|---|
|1. pix2pix 모델 학습을 위해 필요한 데이터셋을 적절히 구축하였다.|데이터 분석 과정 및 한 가지 이상의 augmentation을 포함한 데이터셋 구축 과정이 체계적으로 제시되었다.|
|2. pix2pix 모델을 구현하여 성공적으로 학습 과정을 진행하였다.|U-Net generator, discriminator 모델 구현이 완료되어 train_step이 안정적으로 진행됨을 확인하였다.|
|3. 학습 과정 및 테스트에 대한 시각화 결과를 제출하였다.|10 epoch 이상의 학습을 진행한 후 최종 테스트 결과에서 진행한 epoch 수에 걸맞은 정도의 품질을 확인하였다.|


## 소고

사실 뭔가 만들어내는 모델은 정말 학습을 많이 해야 가능할것 같습니다. 뭔가를 만든다는게 정말 어려운것같아요..

그래도 이전에 배웠던 gan 계열 모델에 비해서 U-net이 훨씬 짧은 시간이지만 잘 학습을 시킨다는 인상을 받았습니다.

사실 input data중에 도로의 비중이 큰데 차선 같은건 구별이 안되어있으니 잘 안되고

작은 물체들은 잘 인식하지 않을까 추측했는데 작은 사람같은 것은 아예 제대로 생성을 못한 결과를 낳은것이 다소 아쉽습니다.
