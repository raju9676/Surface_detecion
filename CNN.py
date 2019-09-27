output_Y = np.array(output_Y)
output_Y = output_Y.reshape(len(output_Y),1)
output_Y = np_utils.to_categorical(output_Y)


input_X = np.array(input_X)
input_X = input_X.reshape(len(input_X),256,256,1)
print(input_X.shape)
input_X = np.append(input_X,np.zeros(shape=input_X.shape),axis = 3)
input_X = np.append(input_X,np.zeros(shape=(2103,256,256,1)),axis = 3)
#input_X = np.repeat(input_X[..., np.newaxis], 3, -1)
input_X = input_X/255

X_train, X_test, y_train, y_test = train_test_split(input_X , output_Y, test_size=0.1, random_state=42)

input_X.shape
from keras.layers import Dropout,AveragePooling2D
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


#parameters for architecture
input_shape = (256, 256, 3)
num_classes = 7
conv_size = 32

# parameters for training
batch_size = 32
num_epochs = 20
# build the model
model = Sequential()

model.add(Conv2D(conv_size, (5, 5), activation='relu', padding='same', input_shape=input_shape)) 
model.add(AveragePooling2D(pool_size=(4, 4)))

model.add(Conv2D(conv_size, (5, 5), activation='relu', padding='same'))
model.add(AveragePooling2D(pool_size=(4, 4)))
    
#model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
    
model.add(Flatten())

model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
 

# train the model                    
history1 = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
validation_split=0.1)

out = model.predict(X_test)
# to get the graphs
import matplotlib.pyplot as plt


plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'Validaton'], loc='upper left')
plt.show()