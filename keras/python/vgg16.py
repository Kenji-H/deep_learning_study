from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np

# load build-in model
model = VGG16(weights='imagenet', include_top=True)

# load test image in VGG16 format
img = image.load_img('pizza.jpg', target_size=(224, 224))
img = np.expand_dims(img, axis=0)
img = preprocess_input(img.astype('float32'))

# predict
preds = model.predict(img)
for _, y, score in decode_predictions(preds, top=5)[0]:
    print ("{}: {}".format(y, score))

# pizza:0.684196829796
# potpie:0.101900361478
# bagel:0.032512024045
# frying_pan:0.0262708757073
# soup_bowl:0.0165681876242
