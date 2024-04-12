from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input


def createModel():
    # convolutional layers
    input_shape = (128, 128, 1)
    inputs = Input(shape=input_shape)
    conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxp_1)
    maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
    conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
    maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
    conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
    maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)
    flatten = Flatten()(maxp_4)
    
    # fully connected layers
    dense_1 = Dense(256, activation='relu')(flatten)
    dense_2 = Dense(256, activation='relu')(flatten)
    dropout_1 = Dropout(0.3)(dense_1)
    dropout_2 = Dropout(0.3)(dense_2)
    
    output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)
    output_2 = Dense(1, activation='relu', name='age_out')(dropout_2)
    
    model = Model(inputs=[inputs], outputs=[output_1, output_2])
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])
    
    return model