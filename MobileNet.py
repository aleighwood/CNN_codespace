#PLAN
#Implement MobileNet using keras
#Revise architecture
#Use tf depthwise separable convolution to build MobileNet layers
#Build custom depthwise seperable conv layer 
#Buid remaining custom layers such as relu, batchnorm, pooling, flatten, dense,padding,dropout 

import tensorflow as tf

#Prebuilt mobile net

mobilenet = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    include_top=True,       # True = includes final dense classifier
    weights='imagenet'      # Load pretrained weights
)

mobilenet.summary()


# re watch mobile net video 

