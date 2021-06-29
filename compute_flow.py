import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like 
import scipy.ndimage
import cv2
from math import ceil,floor
from scipy.ndimage.filters import convolve as filter2
from scipy.sparse import spdiags
from scipy.signal import medfilt
from scipy.ndimage import median_filter,gaussian_filter
import scipy.sparse as sparse
import math
from scipy.ndimage import correlate
from skimage.transform import resize
from skimage.util import dtype
import flow_operator as fo
import rescale_img as ri
import energies as en
import matplotlib.pyplot as plt
import numpy.matlib
##############################################################
def compute_image_pyram(Im1,Im2,ratio,N_levels,gaussian_sigma):
    P1=[]
    P2=[]
    tmp1=ri.scale_image(Im1,0,255)
    tmp2=ri.scale_image(Im2,0,255)
    P1.append(tmp1)
    P2.append(tmp2)

    for lev in range(1,N_levels):
        tmp1=gaussian_filter(tmp1,gaussian_sigma)
        tmp2=gaussian_filter(tmp2,gaussian_sigma)
        sz=np.round(np.array(tmp1.shape,dtype=np.float32)*ratio)

        tmp1=resize(tmp1,(sz[0],sz[1]),anti_aliasing=False,mode='symmetric')
        tmp2=resize(tmp2,(sz[0],sz[1]),anti_aliasing=False,mode='symmetric')

        P1.append(tmp1)
        P2.append(tmp2)
    return [P1,P2]


def resample_flow_unequal(u,v,sz,ordre_inter):
    osz=u.shape
    ratioU=sz[0]/osz[0]
    ratioV=sz[1]/osz[1]
    u=resize(u,sz,order=ordre_inter)*ratioU
    v=resize(v,sz,order=ordre_inter)*ratioV
    return u,v
############################################################
def compute_flow(Im1,Im2,u,v,iter_gnc,gnc_pyram_levels,gnc_factor,gnc_spacing, pyram_levels,factor,spacing,ordre_inter, alpha,lmbda, size_median_filter,h,coef,S,max_linear_iter,max_iter,lambda2,lambda3,eps,a,sigma_qua):
    param1=1/8; param2=100; param3=0.95; param4=False;
    #param1=1/10; param2=100; param3=0.5; param4=False 
    Im1,Imm1=ri.decompo_texture(Im1, param1, param2, param3, param4)
    Im2,Imm1=ri.decompo_texture(Im2, param1, param2, param3, param4)

    P1,P2=compute_image_pyram(Im1,Im2,1/factor,pyram_levels,math.sqrt(spacing)/math.sqrt(2))
    P1_gnc,P2_gnc=compute_image_pyram(Im1,Im2,1/gnc_factor,gnc_pyram_levels,math.sqrt(gnc_spacing)/math.sqrt(2))
    uhat=u; vhat=v; remplacement=True;
    itersLO=1;
    for i in range(iter_gnc):
        if i==(iter_gnc-1):
            remplacement=False
        else:
            remplacement=True

        if i==0:
            py_lev=pyram_levels
        else:
            py_lev=gnc_pyram_levels
        print('pylev',py_lev)
        for lev in range(py_lev-1,-1,-1):
            if i==0:
                Image1=P1[lev]; Image2=P2[lev]
                sz= Image1.shape
            else:
                
                Image1=P1_gnc[lev]; Image2=P2_gnc[lev]
                sz= Image1.shape
            
            u,v=resample_flow_unequal(u,v,sz,ordre_inter)
            uhat,vhat=resample_flow_unequal(uhat,vhat,sz,ordre_inter)
            print('shapes,',u.shape,uhat.shape)

            if (lev==0) and (i==iter_gnc) :    #&& this.noMFlastlevel
                median_filter_size =0
            else: 
                median_filter_size=size_median_filter
                    
           
            u,v,uhat,vhat=fo.compute_flow_base(Image1,Image2,max_iter,max_linear_iter,u,v,alpha,lmbda,S,median_filter_size,h,coef,uhat,vhat,itersLO,lambda2,lambda3,remplacement,eps,a,sigma_qua)
           

            if iter_gnc > 0:

                new_alpha  = 1 - (i+1)/ (iter_gnc)
                alpha = min(alpha, new_alpha)
                alpha = max(0,alpha)
            print('iteration gnc',i)

    u=uhat
    v=vhat
    return u,v
####################################################
Im1=cv2.imread('Im11.png',0)
Im2=cv2.imread('Im22.png',0)
Im1=np.array(Im1,dtype=np.float32)
Im2=np.array(Im2,dtype=np.float32)
u=np.zeros((Im1.shape)); v=np.zeros((Im1.shape))
'''iter_gnc=3
gnc_pyram_levels=2
gnc_factor=1.25
gnc_spacing=1.25

pyram_levels=1
factor=2
spacing=2
ordre_inter=1
alpha=1
lmbda=3
lambda2=1e-1 ; lambda3=1
size_median_filter=5
h=np.array([[-1 ,8, 0 ,-8 ,1 ]]); h=h/12
coef=0.5
S=[]
S.append(np.array([[1,-1]]))
S.append(np.array([[1],[-1]]))
max_linear_iter=1
max_iter=10'''
#GNC params
iter_gnc=3
gnc_pyram_levels=2
'''gnc_factor=1.25
gnc_spacing=1.25'''
gnc_factor=1.25
gnc_spacing=1.25
#Pyram params 
pyram_levels=1
'''factor=2
spacing=2'''
factor=2
spacing=2
ordre_inter=1
alpha=1
size_median_filter=1
h=np.array([[-1 ,8, 0 ,-8 ,1 ]]); h=h/12
coef=0.5
S=[]
S.append(np.array([[1,-1]]))
S.append(np.array([[1],[-1]]))
#Algo params
#a=0.45
a=1
eps=0.001
max_linear_iter=1
#max_iter=10
max_iter=10
lmbda=10000
#lambda2=1e-1
#lambda2=0.01
#lambda2=50
lambda2=0.01
lambda3=2.5
sigma_qua=50

#pyram_levels=ri.compute_auto_pyramd_levels(Im1,spacing)
pyram_levels=3
u,v=compute_flow(Im1,Im2,u,v,iter_gnc,gnc_pyram_levels,gnc_factor,gnc_spacing, pyram_levels,factor,spacing,ordre_inter, 
alpha,lmbda, size_median_filter,h,coef,S,max_linear_iter,max_iter,lambda2,lambda3,eps,a,sigma_qua)

print('Im1')
print(Im1[0:5,0:5])

print('Im2')
print(Im2[0:5,0:5])

print('u')
print(u[0:5,0:5])

print('v')
print(v[0:5,0:5])
N,M=Im1.shape
y=np.linspace(0,N-1,N)
x=np.linspace(0,M-1,M)
x,y=np.meshgrid(x,y)
x2=x+u; y2=y+v
x2=np.array(x2,dtype=np.float32)
y2=np.array(y2,dtype=np.float32)
I=cv2.remap(np.array(Im1,dtype=np.float32),x2,y2,cv2.INTER_LINEAR)
norme=np.linalg.norm(I-Im2)/np.linalg.norm(Im2)
print(norme)
cv2.imwrite('I_3.png',I)
print(I[0:5,0:5])
step=20
Exy,Exx=np.gradient(u)
plt.figure(); plt.imshow(Exx);plt.clim(-0.1,0.1);plt.colorbar();'''plt.imshow(Im1);''' ;plt.title("mf.size=1 lmbda=1e+04,Lambda2=1e-02 Lambda3=2.5"); plt.show(block=False) ; plt.savefig('Quadratique_lmbda_10000_lambda2_001_lambda3_2p5_mf1')
np.savetxt('u_lmbda_10000_lambda2_001_lmbda3_2p5_mf1.txt',u,fmt='%.2f') 
np.savetxt('v_lmbda_10000_lambda2_001_lmbda3_2p5_mf1.txt',v,fmt='%.2f')                                                                                                                                                                                                        
print("Energie Image:",en.energie_image(Im1,Im2,u,v))
print("Energie Grad déplacement:",en.energie_grad_dep(u,v,lmbda))                                      
