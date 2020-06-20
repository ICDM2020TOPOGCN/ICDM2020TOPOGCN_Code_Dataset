import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from persim import plot_diagrams, PersImage
from ripser import ripser, lower_star_img
from ripser import Rips
import seaborn as sns
import dionysus as d

# Extract images #
f1 = h5py.File('unwrap_images.h5', 'r')
images = []
for i in range(len(list(f1.keys()))):
    G_temp = f1.get(list(f1.keys())[i])
    image_temp = np.array(G_temp.get('image'))
    for j in range(image_temp.shape[2]):
        image_temp_sec = np.array(G_temp.get('image'))
        images.append(image_temp_sec[:,:,j])

images = np.array(images, dtype=np.float32)
images = images.reshape((-1,9,20,1)) # reshape
images = images/(images.max()/255.0) # normalize

np.save('images_sensor', images)

images = np.load('images_sensor.npy')

# Extract persistence images from images (PIs) #
def homology_persistent_diagrams(dgms):
    h0 = []
    h1 = []
    for i, dgm in enumerate(dgms):
        for p in dgm:
            if i == 0:
                h0.append(np.array((p.birth, p.death)))
            elif i == 1:
                h1.append(np.array((p.birth, p.death)))

    h0 = np.asarray(h0)
    h1 = np.asarray(h1)
    h0 = h0.reshape((-1, 2))
    h1 = h1.reshape((-1, 2))
    return [h0,h1]

PI_null_images = []
PI_one_images = []

for i in range(images.shape[0]):
    print(i)
    f_lower_star = d.fill_freudenthal(images[i,:, :, 0].astype(float))
    f_upper_star = d.fill_freudenthal(images[i,:, :, 0].astype(float), reverse=True)
    p = d.homology_persistence(f_lower_star)
    dgms_temp = d.init_diagrams(p, f_lower_star)
    h0_temp = homology_persistent_diagrams(dgms_temp)[0]
    h1_temp = homology_persistent_diagrams(dgms_temp)[1]
    pim = PersImage(pixels=[20, 20], spread=1)
    PI_0_temp = pim.transform(h0_temp[1:,:])
    PI_1_temp = pim.transform(h1_temp)
    PI_null_images.append(PI_0_temp)
    PI_one_images.append(PI_1_temp)

PI_null_images = np.array(PI_null_images, dtype=np.float32)
PI_null_images = PI_null_images.reshape((-1,20,20,1)) # reshape
PI_null_images = PI_null_images/(PI_null_images.max()/255.0) # normalize

PI_one_images = np.array(PI_one_images, dtype=np.float32)
PI_one_images = PI_one_images.reshape((-1,20,20,1)) # reshape
PI_one_images = PI_one_images/(PI_one_images.max()/255.0) # normalize

np.save('PI_null_images', PI_null_images)
np.save('PI_one_images', PI_one_images)


# Labels #
input_sensor_images = images

labels = np.zeros((input_sensor_images.shape[0],2),dtype= np.float32)
for i in range(len(list(f1.keys()))):
    G_temp = f1.get(list(f1.keys())[i])
    label_temp = np.array(G_temp.get('label'))
    if i == 0:
        final_labels = label_temp
    else:
        final_labels = np.hstack((final_labels, label_temp))

for i in range(input_sensor_images.shape[0]):
    print(i)
    if final_labels[i] == 0.:
        labels[i,0] = 1.
    elif final_labels[i] == 1.:
        labels[i,1] = 1.

np.save("labels",labels)


# Extract persistence diagrams from images (PDs) #
PDs_null = []
PDs_one = []

for i in range(images.shape[0]):
    print(i)
    f_lower_star = d.fill_freudenthal(images[i,:, :, 0].astype(float))
    f_upper_star = d.fill_freudenthal(images[i,:, :, 0].astype(float), reverse=True)
    p = d.homology_persistence(f_lower_star)
    dgms_temp = d.init_diagrams(p, f_lower_star)
    h0_temp = homology_persistent_diagrams(dgms_temp)[0]
    h1_temp = homology_persistent_diagrams(dgms_temp)[1]

    # minmax procedure #
    # for both dg0 and dg1, limit x-axis and y-axis into (0,1)
    h_comb = np.vstack((h0_temp, h1_temp))
    dg_min = np.min(h_comb[:,:])
    dg_max = np.max(h_comb[1:,:])

    h0_temp = np.hstack((np.array([(h0_temp[1:, 0] - dg_min) / (dg_max - dg_min)]).transpose(), np.array([(h0_temp[1:, 1] - dg_min) / (dg_max - dg_min)]).transpose()))
    h1_temp = np.hstack((np.array([(h1_temp[:, 0] - dg_min) / (dg_max - dg_min)]).transpose(),
                         np.array([(h1_temp[:, 1] - dg_min) / (dg_max - dg_min)]).transpose()))

    PDs_null.append(h0_temp)
    PDs_one.append(h1_temp)


# Extract rotated persistence diagrams from images (PDs) #
T_PDs_null = []
T_PDs_one = []

for i in range(images.shape[0]):
    print(i)
    f_lower_star = d.fill_freudenthal(images[i,:, :, 0].astype(float))
    f_upper_star = d.fill_freudenthal(images[i,:, :, 0].astype(float), reverse=True)
    p = d.homology_persistence(f_lower_star)
    dgms_temp = d.init_diagrams(p, f_lower_star)
    h0_temp = homology_persistent_diagrams(dgms_temp)[0]
    h1_temp = homology_persistent_diagrams(dgms_temp)[1]

    # minmax procedure #
    # for both dg0 and dg1, limit x-axis and y-axis into (0,1)
    h_comb = np.vstack((h0_temp, h1_temp))
    dg_min = np.min(h_comb[:,:])
    dg_max = np.max(h_comb[1:,:])

    h0_temp = np.hstack((np.array([(h0_temp[1:, 0] - dg_min) / (dg_max - dg_min)]).transpose(), np.array([(h0_temp[1:, 1] - dg_min) / (dg_max - dg_min)]).transpose()))
    h1_temp = np.hstack((np.array([(h1_temp[:, 0] - dg_min) / (dg_max - dg_min)]).transpose(),
                         np.array([(h1_temp[:, 1] - dg_min) / (dg_max - dg_min)]).transpose()))

    h0_temp[:, 1] -= h0_temp[:, 0]
    h1_temp[:, 1] -= h1_temp[:, 0]

    T_PDs_null.append(h0_temp)
    T_PDs_one.append(h1_temp)

np.save('T_PDs_null_collection', T_PDs_null)
np.save('T_PDs_one_collection', T_PDs_one)


Rotation_PDs_null = []
Rotation_PDs_one = []
Longest_Rotation_PDs_null = []


# Extract rotated persistence diagrams with 5 persistent points which have the longest life span (PDs) #
def homology_persistent_diagrams(dgms):
    h0 = []
    h1 = []
    for i, dgm in enumerate(dgms):
        for p in dgm:
            if i == 0:
                h0.append(np.array((p.birth, p.death)))
            elif i == 1:
                h1.append(np.array((p.birth, p.death)))

    h0 = np.asarray(h0)
    h1 = np.asarray(h1)
    h0 = h0.reshape((-1, 2))
    h1 = h1.reshape((-1, 2))
    return [h0,h1]

for i in range(images.shape[0]):
    print(i)
    f_lower_star = d.fill_freudenthal(images[i,:, :, 0].astype(float))
    f_upper_star = d.fill_freudenthal(images[i,:, :, 0].astype(float), reverse=True)
    p = d.homology_persistence(f_lower_star)
    dgms_temp = d.init_diagrams(p, f_lower_star)
    h0_temp = homology_persistent_diagrams(dgms_temp)[0]
    h1_temp = homology_persistent_diagrams(dgms_temp)[1]

    # minmax procedure #
    # for both dg0 and dg1, limit x-axis and y-axis into (0,1)
    h_comb = np.vstack((h0_temp, h1_temp))
    dg_min = np.min(h_comb[:,:])
    dg_max = np.max(h_comb[1:,:])

    h0_temp = np.hstack((np.array([(h0_temp[1:, 0] - dg_min) / (dg_max - dg_min)]).transpose(), np.array([(h0_temp[1:, 1] - dg_min) / (dg_max - dg_min)]).transpose()))
    h1_temp = np.hstack((np.array([(h1_temp[:, 0] - dg_min) / (dg_max - dg_min)]).transpose(),
                         np.array([(h1_temp[:, 1] - dg_min) / (dg_max - dg_min)]).transpose()))

    h0_temp[:, 1] -= h0_temp[:, 0]
    h1_temp[:, 1] -= h1_temp[:, 0]

    h1_temp[:, 0] = h1_temp[:, 0] + h1_temp[:, 1] + h1_temp[:, 0]
    h0_temp[:, 0] = h0_temp[:, 0] + h0_temp[:, 1] + h0_temp[:, 0]

    Rotation_PDs_null.append(h0_temp)
    Rotation_PDs_one.append(h1_temp)

    five_longest_h0_temp = h0_temp[h0_temp[:, 1].argsort()[-5:], :] # 5 persistent points which have the longest lifespan #
    Longest_Rotation_PDs_null.append(five_longest_h0_temp)


np.save('Longest_Rotation_PDs_null_collection', Longest_Rotation_PDs_null)
