import os
# directory with our training brain tumor yes images
brain_tumor_detected_dir=os.path.join("brain_tumor_dataset/yes/") 
# directory with our training brain tumor no images
brain_tumor_not_detected_dir=os.path.join("brain_tumor_dataset/no/") 
brain_tumor_detected=os.listdir(brain_tumor_detected_dir) 
brain_tumor_not_detected=os.listdir(brain_tumor_not_detected_dir) 
print(brain_tumor_detected[:10]) 
print(brain_tumor_not_detected[:10]) 
print("Total images of brain_tumor_detected : ", len(os.listdir(brain_tumor_detected_dir))) 
print("Total images of brain_tumor_not_detected : ", len(os.listdir(brain_tumor_not_detected_dir))) 
%matplotlib inline 
importmatplotlib.pyplotasplt
importmatplotlib.imageasmpimg
nrows=4
ncols=4
pic_index=0
#Set up matplotlib fig, and size it to fit 4x4 pics
fig =plt.gcf() 
fig.set_size_inches(ncols*4, nrows*4) 
pic_index+=8
next_tumor_detected= [os.path.join(brain_tumor_detected_dir, fname) 
forfnameinbrain_tumor_detected[pic_index-8:pic_index]] 
next_not_tumor_detected= [os.path.join(brain_tumor_not_detected_dir, fname) 
forfnameinbrain_tumor_not_detected[pic_index-8:pic_index]] 
fori, img_pathinenumerate(next_tumor_detected+next_not_tumor_detected): 
# Set up subplot; subplot indices start at 1
sp=plt.subplot(nrows, ncols, i+1) 
sp.axis('Off') # Don't show axes (or gridlines)
img=mpimg.imread(img_path) 
plt.imshow(img) 
plt.show() 
!pip install tensorflow 
import tensor flowast
fromtensorflowimportkeras 
model =keras.Sequential([ 
keras.layers.Conv2D(32,(3,3),activation="relu",input_shape=(300,300,3)), 
keras.layers.MaxPooling2D(2,2), 
keras.layers.Conv2D(32,(3,3),activation="relu"), 
keras.layers.MaxPooling2D(2,2), 
keras.layers.Conv2D(64,(3,3),activation="relu"), 
keras.layers.MaxPooling2D(2,2), 
keras.layers.Conv2D(64,(3,3),activation="relu"), 
keras.layers.MaxPooling2D(2,2), 
keras.layers.Flatten(), 
keras.layers.Dense(512,activation="relu"), 
keras.layers.Dense(1,activation="sigmoid") 
]) 
model.summary() 
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), 
 loss="binary_crossentropy", 
 metrics=["accuracy"]) 
fromkeras.preprocessing.imageimportImageDataGenerator 
# All images will be rescaled by 1./255
train_datagen=ImageDataGenerator(rescale=1/255) 
train_generator=train_datagen.flow_from_directory("brain_tumor_dataset/", 
target_size=(300,300), 
batch_size=32, 
class_mode="binary")
history =model.fit(train_generator, 
steps_per_epoch=8, 
 epochs=15, 
 verbose=1) 
importnumpyasnp
fromkeras.preprocessingimport image 
path ="Y182.jpg"
img=image.load_img(path,target_size=(300,300)) 
x =image.img_to_array(img) 
x =np.expand_dims(x,axis=0) 
images =np.vstack([x]) 
classes =model.predict(images,batch_size=10) 
print(classes[0]) 
ifclasses[0] >0.5: 
print("This is Tumor.. Tumor Detected") 
else: 
print("This is not tumor.. Tumor not Detected") 
image.load_img(path) 