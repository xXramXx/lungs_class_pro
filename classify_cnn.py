import tensorflow as tf
from tensorflow.keras import layers,models

def cnn_model (input_shape,classes):
	if classes == 1:
		model=tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(16,(3,3),input_shape=input_shape,activation='relu'),
		tf.keras.layers.MaxPool2D(2,2),
		tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
		tf.keras.layers.MaxPool2D(2,2),
		tf.keras.layers.Dropout(.2, input_shape=(2,)),
		tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
		tf.keras.layers.MaxPool2D(2,2),
		tf.keras.layers.Conv2D(204,(3,3),activation='relu'),
		tf.keras.layers.MaxPool2D(2,2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512,activation='relu'),
		tf.keras.layers.Dense(classes,activation='sigmoid')
	])
	
	else:
		model=tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(16,(3,3),input_shape=input_shape,activation='relu'),
		tf.keras.layers.MaxPool2D(2,2),
		tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
		tf.keras.layers.MaxPool2D(2,2),
		tf.keras.layers.Dropout(.2, input_shape=(2,)),
		tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
		tf.keras.layers.MaxPool2D(2,2),
		tf.keras.layers.Conv2D(204,(3,3),activation='relu'),
		tf.keras.layers.MaxPool2D(2,2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(512,activation='relu'),
		tf.keras.layers.Dense(classes,activation='softmax')
	])
	#model1 = models.Model( model,name="cnn_model")
	#model1=model.build(input_shape)
	
	return model

if __name__ == "__main__" :
	x,y,c = [int(x) for x in input ("enter model input size as x,y and number of channela :").split()]
	input_shape = (x,y,c)
	classes = input ("enter number of output classes:")
	print("Building model...")
	model=cnn_model(input_shape,int(classes))
	#model.build(input_shape)
	print(model.summary())
	
	
