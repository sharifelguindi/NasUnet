import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral
import cv2

def dense_crf(img, output_probs):
    output_probs = output_probs[0, :, :, :]
    num_classes = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]


    d = dcrf.DenseCRF2D(w, h, num_classes) # width, height, nlabels
    output_probs = cv2.normalize(output_probs, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    output_probs = np.clip(output_probs, 0.0, 1.0)

    U = -np.log(output_probs) # U should be negative log-probabilities
    U = U.reshape((num_classes, -1))    # Needs to be flat
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U.astype(np.float32)) # 设置一元势函数

    ## This adds the color-independent term, features are the locations only.
    # d.addPairwiseGaussian(sxy=20, compat=3)
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b). no use for this data !!!
    #d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)#仅支持RGB

    # Create the pairwise bilateral term from the above image.
    # The two `s{dims,chan}` parameters are model hyper-parameters defining
    # the strength of the location and image content bilaterals, respectively.
    # chdim代表channels通道在哪个维度
    # pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01), img=img, chdim=2)

    #d.addPairwiseEnergy(pairwise_energy, compat=10) # 'compat' is the "strength" of this potential

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q
