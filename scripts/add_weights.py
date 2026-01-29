import tensorflow as tf
model = tf.keras.applications.MobileNet(weights="imagenet", input_shape=(224,224,3), classes=1000)
model.save_weights("mobilenet_imagenet.weights.h5")