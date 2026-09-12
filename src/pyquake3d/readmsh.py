import numpy as np
import matplotlib.pyplot as plt
from math import *
import os
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 16})

pi=3.1415926

"""Read Gmsh meshes through the meshio library."""

import numpy as np
import meshio


def read_gmsh(filename,):
    
    mesh = meshio.read(filename)


    nodes = mesh.points
    elements = mesh.cells_dict["triangle"]

    elements = mesh.cells_dict["triangle"] + 1
    scale=1000
    nodes = mesh.points * scale
    return nodes, elements





def read_mshV2(fname,tyK=-1,seprate=False):  #read msh file
    f=open(fname,'r')
    start_node=0
    start_ele=0
    K1=0
    node=[]
    ele=[]
    for line in f:
        line=line[:-1]
        
        if(start_node==1):
            K1=K1+1
            sliptdata=line.split()
            #print(sliptdata)
            if(K1>1):
                try:    
                    nodeline=[]
                    tem=float(line.split()[1])
                    nodeline.append(tem)
                    tem=float(line.split()[2])
                    nodeline.append(tem)
                    tem=float(line.split()[3])
                    nodeline.append(tem)
                    node.append(nodeline)
                    #print(nodeline)
                except:
                    start_node=0
                    K1=0
        if(start_ele==1):
            K1=K1+1
            sliptdata=line.split()
            if(K1>1):
                try:
                    if(seprate==True):
                        if(len(line.split())>=8 and int(sliptdata[-4])==tyK): 
                            eleline=[]
                            tem=int(sliptdata[-3])
                            eleline.append(tem)
                            tem=int(sliptdata[-2])
                            eleline.append(tem)
                            tem=int(sliptdata[-1])
                            eleline.append(tem)
                            ele.append(eleline)
                            #print(eleline)
                    else:
                        if(len(line.split())>=8): 
                            eleline=[]
                            tem=int(sliptdata[-3])
                            eleline.append(tem)
                            tem=int(sliptdata[-2])
                            eleline.append(tem)
                            tem=int(sliptdata[-1])
                            eleline.append(tem)
                            ele.append(eleline)


                except:
                    start_ele=0

        if(line=='$Nodes'):
            start_node=1
            
        if(line=='$Elements'):
            start_ele=1
            #print(start_ele)
        
        #boundary_edges,boundary_nodes=find_boundary_edges_and_nodes(ele)
    
    return np.array(node)*1e3,np.array(ele)
    



# fname='1.msh'
# nodelst,elelst=read_mshV2(fname)
# print(nodelst.shape,elelst.shape)

import numpy as np




def get_eleVec(nodelst,elelst,jud_ele_order):
    eleVec=[]
    xg=[]
    for i in range(len(elelst)):
        xa=nodelst[elelst[i,0]-1]
        xb=nodelst[elelst[i,1]-1]
        xc=nodelst[elelst[i,2]-1]

        # if(i==12):
        #     plt.scatter(xa[0],xa[1],color='r')  #Check whether the nodes are ordered counterclockwise or clockwise
        #     plt.scatter(xb[0],xb[1],color='b')
        #     plt.scatter(xc[0],xc[1],color='y')
        #     plt.show()

        xg.append([np.mean([xa[0],xb[0],xc[0]]),np.mean([xa[1],xb[1],xc[1]]),np.mean([xa[2],xb[2],xc[2]])])

        vba=xb-xa
        vca=xc-xa

       
        
        #jud_ele_order=True
        if(jud_ele_order==True):
            #Nodes are ordered clockwise ac*ab
            ev31 = vca[1]*vba[2]-vca[2]*vba[1]
            ev32 = vca[2]*vba[0]-vca[0]*vba[2]
            ev33 = vca[0]*vba[1]-vca[1]*vba[0]
        else:
            #Nodes are ordered counterclockwise ab*ac
            ev31 = vba[1]*vca[2]-vba[2]*vca[1]
            ev32 = vba[2]*vca[0]-vba[0]*vca[2]
            ev33 = vba[0]*vca[1]-vba[1]*vca[0]
        rr = sqrt(ev31*ev31+ev32*ev32+ev33*ev33)
        # unit vectors for local coordinates of elements
        ev31 /=rr
        ev32 /=rr
        ev33 /= rr

        

        if abs(ev33) < 1.0:
            # ez x normal
            ev11 = -ev32
            ev12 =  ev31
            ev13 =  0.0

            rr = sqrt(ev11*ev11 + ev12*ev12)
            ev11 /= rr
            ev12 /= rr

        else:
            # Horizontal triangle
            ev11 = 1.0
            ev12 = 0.0
            ev13 = 0.0
        

        ev21 = ev32*ev13-ev33*ev12
        ev22 = ev33*ev11-ev31*ev13
        ev23 = ev31*ev12-ev32*ev11
        eleVec.append([ev11,ev12,ev13,ev21,ev22,ev23,ev31,ev32,ev33])
    eleVec=np.array(eleVec)
    xg=np.array(xg)
    #print(eleVec)
    return eleVec,xg



def read_node(fname):
    f=open(fname,'r')
    data0=[]
    data1=[]
    kline=0
    for line in f:
        kline=kline+1
        N1=len(line.split())
        if(N1>0):
            datatem=[]
            for k in range(N1):
                tem=float(line.split()[k])
                datatem.append(tem)
            if(N1==4):
                data0.append(datatem)
            elif(N1==3):
                data1.append(datatem)
    f.close()
    return np.array(data0),np.array(data1)

def read_elenum():
    Fele=[]
    Aele=[]
    Sele=[]
    f=open('indats/in_fgeom.dat')
    for line in f:
        Fele.append(int(line.split()[0]))
    f.close()
    f=open('indats/in_sgeom.dat')
    for line in f:
        Sele.append(int(line.split()[0]))
    f.close()
    # f=open('indats/in_ageom.dat')
    # for line in f:
    #     Aele.append(int(line.split()[0]))
    # f.close()
    return np.array(Fele),np.array(Sele),np.array(Aele)

def read_data(fname,K,N1):
    f=open(fname,'r')
    data1=[]
    kline=0
    for line in f:
        #if(len(line.split())>0 and kline>=K*N1 and kline<(K+1)*N1):
        tem=line.split()[0]
        data1.append(float(tem))
        kline=kline+1
    f.close()
    return np.array(data1)

def read_dats(K,teminfo):
    fv1=read_data('dats/fsnapd1.dat',K,int(teminfo[0]))
    fv2=read_data('dats/fsnapd2.dat',K,int(teminfo[0]))
    fs1=read_data('dats/fsnaps1.dat',K,int(teminfo[0]))
    fs2=read_data('dats/fsnaps2.dat',K,int(teminfo[0]))
    ft1=read_data('dats/fsnapt1.dat',K,int(teminfo[0]))
    ft2=read_data('dats/fsnapt2.dat',K,int(teminfo[0]))
    ft3=read_data('dats/fsnapt3.dat',K,int(teminfo[0]))
    arriT=read_data('dats/arriveT.dat',K,int(teminfo[0]))
    
    sv1=read_data('dats/ssnapd1.dat',K,int(teminfo[1]+teminfo[2]))
    sv2=read_data('dats/ssnapd2.dat',K,int(teminfo[1]+teminfo[2]))
    sv3=read_data('dats/ssnapd3.dat',K,int(teminfo[1]+teminfo[2]))
    ss1=read_data('dats/ssnaps1.dat',K,int(teminfo[1]+teminfo[2]))
    ss2=read_data('dats/ssnaps2.dat',K,int(teminfo[1]+teminfo[2]))
    ss3=read_data('dats/ssnaps3.dat',K,int(teminfo[1]+teminfo[2]))
    return fv1,fv2,fs1,fs2,ft1,ft2,ft3,sv1,sv2,sv3,ss1,ss2,ss3,arriT

def trans(Fele,Sele,Aele,fv,sv):
    N=len(Fele)+len(Sele)+len(Aele)
    data1=[0]*N

    for i in range(len(fv)):
        data1[Fele[i]]=fv[i]
    for i in range(len(sv)):
        if(i<len(Sele)):
            data1[Sele[i]]=sv[i]
        else:
            data1[Aele[i-len(Sele)]]=sv[i]
    return np.array(data1)
