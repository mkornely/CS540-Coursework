from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x=np.load(filename)
    mean=np.mean(x, axis=0)
    
    return x-mean

def get_covariance(dataset):
    return np.dot(np.transpose(dataset), dataset)/(len(dataset)-1)

    

def get_eig(S, m):
    Lambda, U=eigh(S,subset_by_index=[len(S)-m,len(S)-1])
    Lambda=np.diag(np.sort(Lambda)[::-1])
    U=np.fliplr(U)
    
    return Lambda,U

def get_eig_prop(S, prop):
    Lambda, U=eigh(S)
    i=sum(Lambda)*prop

    Lambda, U= eigh(S,subset_by_value=[i,np.inf])
    
    Lambda=np.diag(np.sort(Lambda)[::-1])
    U=np.fliplr(U)
    
    return Lambda,U


def project_image(image, U):
    projection=np.zeros(len(image))
    
    for index in range(0,len(U[0])):
        projection+=np.inner(U[:,index],image)*(U[:,index])
        
    return projection
    

def display_image(orig, proj):
    orig=orig.reshape(32,32).T
    proj=proj.reshape(32,32).T
    
    fig,(ax1,ax2) = plt.subplots(1, 2)
    colorbar_1=ax1.imshow(orig,aspect='equal')
    ax1.set_title('Original')
    fig.colorbar(colorbar_1, ax=ax1,shrink=0.55)
    colorbar_2=ax2.imshow(proj,aspect='equal')
    ax2.set_title('Projection')
    fig.colorbar(colorbar_2, ax=ax2,shrink=0.55)
   
    plt.show()
    
