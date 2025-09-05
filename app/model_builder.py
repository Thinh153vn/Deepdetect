# app/model_builder.py
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, DenseNet121
import config

def build_model(model_name='efficientnet'):
    input_shape = (config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 3)
    if model_name == 'efficientnet':
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    elif model_name == 'densenet':
        base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise ValueError("Unsupported model name.")
    base_model.trainable = False
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model