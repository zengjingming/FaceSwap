from tensorflow.keras.models import load_model
from retinaface.models import *
import cv2
import numpy as np
from utils.utils import (estimate_norm, get_lm, inverse_estimate_norm,
                         norm_crop, transform_landmark_points)


RetinaFace = load_model("./retinaface/RetinaFace-Res50.h5", compile=False,
                            custom_objects={"FPN": FPN,
                                            "SSH": SSH,
                                            "BboxHead": BboxHead,
                                            "LandmarkHead": LandmarkHead,
                                            "ClassHead": ClassHead})
source="./imgs/jk/jk7.jpeg"
source = cv2.imread(source)
source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
source = np.array(source)

source_h, source_w, _ = source.shape #height,width
source_a = RetinaFace(np.expand_dims(source, axis=0)).numpy()[0] #landmarks
print(source_a)
source_lm = get_lm(source_a, source_w, source_h)
print(source_lm)
source_aligned = norm_crop(source, source_lm, image_size=112, shrink_factor=1.0)
source_aligned = cv2.cvtColor(source_aligned, cv2.COLOR_BGR2RGB)
for point in source_lm:
    pt_pos = (int(point[0]), int(point[1]))
    cv2.circle(source_aligned, pt_pos, 1, (255, 0, 0), 2)
cv2.imshow("img",source_aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()