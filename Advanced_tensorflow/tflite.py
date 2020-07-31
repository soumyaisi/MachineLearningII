"""
convert the model to tflite format.
"""

import tensorflow as tf

#create a converter object
conveter = tf.lite.TFLiteConverter.from_keras_model_file(model)

#convert the model
tflite_model = conveter.convert()

#save the file
with open("converted_model.tflite", 'wb') as f:
	f.write(tflite_model)

