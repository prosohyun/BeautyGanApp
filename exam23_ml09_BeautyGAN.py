#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#응용하면 고객한테 어울리는 화장이나 헤어스타일 고르기 가능 


# In[1]:


import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


# In[2]:


#detector는 얼굴을 찾아주는 detector 사진에서 얼굴 영역 찾기 / 얼굴의 랜드마크 찾아주는 모델 - face_landmark
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat') 
#모양을 예측해주는 모델 - 얼굴의 5개의 랜드마크 예측해주는 모델 


# In[3]:


img = dlib.load_rgb_image('./imgs/11.jpg')#이미지 부르기 
plt.figure(figsize=(16,10))
plt.imshow(img)#이미지니까 imshow
plt.show()


# In[4]:


#얼굴 영역에 사각형 그려주기 
img_result = img.copy()#이미지 원본 손상시키면 안되니까 하나 파일 복사
dets = detector(img, 1)
if len(dets) == 0:
    print('cannot find face!')
else:
    fig, ax = plt.subplots(1,figsize =(16,10))
    #dets 는 얼굴의 영역들이 들어가있음. 원점의 맨위 영역의 폭과 높이에 대한 정보를 영역이 가지고 있는데 얼굴 하나당 영역 1개 
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height() # x,y,w,h 좌표/ 영역의 폭과 넓이/ 하나하나 꺼내서 해줌
        rect = patches.Rectangle((x, y),w,h,linewidth = 2, edgecolor = 'r', facecolor='none') # 얼굴에 색 안 채우고 사각형만 해줌 
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()


# In[5]:


fig, ax  = plt.subplots(1, figsize = (16,10)) 
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle  = patches.Circle((point.x,point.y), radius = 3,
                                edgecolor = 'r', facecolor = 'r')#랜드마크에 빨간색으로 채워진 동그라미 그리기 - 눈이랑 인중 
        ax.add_patch(circle)
ax.imshow(img_result)


# In[6]:


#얼굴들만 부르기 / 얼굴이 몇 개든 간에 다 부를 수 있다.
#패딩이 들어가는 이유 이미지들 사이 간격 주기 위해서 
faces = dlib.get_face_chips(img,objs, size = 256, padding = 0.3)
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)


# In[7]:


#이 코드에 거의 모든 내용 들어있다. 위에 있는 디텍터랑 sp 코드만 위에 있다  
#얼굴 쭉 뽑아주는 함수 만들기 / 옆, 뒤 얼굴은 인식 제대로 못해준다
def align_faces(img):
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    #디텍터가 얼굴 찾아서 넣어줌 얼굴 갯수만큼 포문 돌아감 
    for detection in dets:
        s = sp(img, detection) #이미지에 얼굴 주고/ 여기서 s는 얼굴 랜드마크 5개 점. 
        objs.append(s) #오브젝트에 s의 점 얼굴5개 넣어줌. 예를 들어 얼굴이 4개면 점 20개가 append된다. 
    faces = dlib.get_face_chips(img, objs, size =256, padding=0.35) #face chip은 얼굴 영역만 잘려서 해주는 거. 얼굴 이미지s가 들어있음 
    
    return faces

#위에 있는 함수 실행 
#읽어서 테스트 함수 만들어줌 
test_img = dlib.load_rgb_image('./imgs/02.jpg')
test_faces = align_faces(test_img) #faces는 얼굴이미지들을 리턴해주는 함수
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20, 16)) #얼굴갯수+1 만큼 subplot그려주기 
axes[0].imshow(test_img)
for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)


# In[8]:


#INFO:tensorflow:Restoring parameters from ./models\model 이거 뜨면 모델 잘 불러온 거다
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


# In[9]:


#스케일링 작업 
def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2


# In[68]:


#이미지 보여주고 시각화 하기 

img1 = dlib.load_rgb_image('./imgs/no_makeup/xfsy_0405.png')#노메이크업 , 소스
img1_faces = align_faces(img1)

img2 = dlib.load_rgb_image('./imgs/makeup/5.jpg') #레퍼런스 이미지 
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1,2,figsize =(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()


# In[69]:


src_img = img1_faces[0]
ref_img = img2_faces[0]

#x입력으로 줄 거
X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

#y 입력으로 줄 거
Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict = {X:X_img, Y:Y_img})
output_img = deprocess(output[0])

fig, axes = plt.subplots(1, 3, figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()


# In[ ]:




