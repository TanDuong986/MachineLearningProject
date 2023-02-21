import numpy as np
import matplotlib.pyplot as plt
import cv2

def count_color(img):
    flat_five = img.copy().reshape(-1,3)
    return np.unique(flat_five,axis=0)
def visualize(label,cluser):
    draw = np.array(label)
    hist = np.bincount(draw.ravel(), minlength=cluser)
    print("shape of histogram is ",hist.shape)
    plt.plot(hist)
    return hist
    

anh = cv2.imread('D:/221_Semester/Image_Processing/Practice_code/data/y-nghia-hoa-huong-duong.jpg')
anh = np.array(anh)
nc = count_color(anh)

cluser = 64
entropy = np.random.permutation(nc.shape[0])

nc = nc.astype(np.float64)
center = nc[entropy[0:64]]
prev_center =  np.zeros()
group = np.full(nc.shape[0],65,np.uint8)




for i in range(nc.shape[0]):
    clus = np.argsort(np.sqrt(np.sum((center-nc[i])**2,axis=1)))[0] # ket luan ve clus
    group[i] = clus #gan nhan cho tung gia tri de tinh lai tb
hist = np.bincount(group.ravel(), minlength=cluser)

new_center = np.zeros(center.shape)
for i in range(cluser):
    indices = [j for j in range(len(group)) if group[j] == i]
    new_center[i] = np.sum(nc[indices],axis=0)/hist[i]
a = center-new_center
print(a)

