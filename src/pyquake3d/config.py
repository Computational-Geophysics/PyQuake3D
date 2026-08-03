import numpy as np
import struct
import matplotlib.pyplot as plt
from math import *
import pyquake3d.SH_greenfunction as SH_greenfunction
import pyquake3d.DH_greenfunction as DH_greenfunction
import os
import sys
from mpi4py import MPI
from mpi4py.util import pkl5

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm = pkl5.Intracomm(MPI.COMM_WORLD)
# global Para
# Para={}

_cart_comm = None

def init_cart(dims, periods=None, reorder=True):
    global _cart_comm
    if _cart_comm is not None:
        return _cart_comm  # 已初始化

    comm = MPI.COMM_WORLD
    _cart_comm = comm.Create_cart(dims=dims, periods=periods, reorder=reorder)
    return _cart_comm

def get_cart():
    if _cart_comm is None:
        raise RuntimeError("cart_comm not initialized. Please call init_cart() first.")
    return _cart_comm

def readPara0(fname):
    Para0={}
    f=open(fname,'r')
    for line in f:
        #print(line)
        if ':' in line:
            tem=line.split(':')
            #if(tem[0]==)
            Para0[tem[0].strip()]=tem[1].strip()
    return Para0
    


def readPara(data_dir):
    Para={}
    Para0=readPara0(data_dir)
    Para['parameter directory']=data_dir
    Para['Corefunc directory']=Para0['Corefunc directory']

    Para['save Corefunc']=Para0['save Corefunc']=='True'
    Para['Node_order']=Para0['Node_order']=='True'
    Para['Scale_km']=Para0['Scale_km']=='True'

    Para['Hmatrix_mpi_plot']=Para0['Hmatrix_mpi_plot']=='True'
    Para['Using C++ green function']=Para0['Using C++ green function']=='True'
    Para['Lattice Matrice']=Para0['Lattice Matrice']=='True'
    Para['Lattice Partitioning depth']=int(Para0['Lattice Partitioning depth'])

    Para['Batch_size']=int(Para0['Batch_size'])
    Para['GPU']=Para0['GPU']=='True'
    if(Para0['GPU']=='True'):
        Para['GPU_cores']=int(Para0['GPU_cores'])

    Para['Lame constants']=float(Para0['Lame constants'])
    Para['Shear modulus']=float(Para0['Shear modulus'])
    Para['Rock density']=float(Para0['Rock density'])
    Para['InputHetoparamter']=Para0['InputHetoparamter']=='True'
    Para['Inputparamter file']=Para0['Inputparamter file']
    Para['Shear zone width']=float(Para0['Shear zone width'])

    try:
        Para['If Coupledthermal']=Para0['If Coupledthermal']=='True'
        Para['If Dilatancy']=Para0['If Dilatancy']=='True'
        Para['Dilatancy coefficient']=float(Para0['Dilatancy coefficient'])
        Para['Hydraulic diffusivity']=float(Para0['Hydraulic diffusivity'])
        #self.hw=float(self.Para0['Low permeability zone thickness'])
        #Para['Actively shearing zone thickness']=float(Para0['Actively shearing zone thickness'])
        Para['Effective compressibility']=float(Para0['Effective compressibility'])
        Para['Constant porepressure']=float(Para0['Constant porepressure'])
        Para['Initial porepressure']=float(Para0['Initial porepressure'])
    except:
        print('No Dilatancy parameters.')

    try:
        Para['If thermal']=Para0['If thermal']=='True'
        Para['Thermal diffusivity']=float(Para0['Thermal diffusivity'])
        #Para['Hydraulic diffusivity']=float(Para0['Hydraulic diffusivity'])
        #self.hw=float(self.Para0['Low permeability zone thickness'])
        Para['Ratio of thermal expansivity to compressibility']=float(Para0['Ratio of thermal expansivity to compressibility'])
        Para['Heat capacity']=float(Para0['Heat capacity'])
        #Para['Half width']=float(Para0['Half width'])
        Para['Initial temperature']=float(Para0['Initial temperature'])
        Para['Background temperature']=float(Para0['Background temperature'])
        
    except:
        print('No thermal parameters.')
    
    Para['Half space']=Para0['Half space']=='True'
    Para['Fix_Tn']=Para0['Fix_Tn']=='True'
    Para['ssv_scale']=float(Para0['ssv_scale'])
    Para['ssh1_scale']=float(Para0['ssh1_scale'])
    Para['ssh2_scale']=float(Para0['ssh2_scale'])
    Para['Angle between ssh1 and X-axis']=float(Para0['Angle between ssh1 and X-axis'])
    Para['Vertical principal stress value']=float(Para0['Vertical principal stress value'])
    Para['Vertical principal stress value varies with depth']=Para0['Vertical principal stress value varies with depth']=='True'
    Para['Turnning depth']=float(Para0['Turnning depth'])
    Para['Shear traction solved from stress tensor']=Para0['Shear traction solved from stress tensor']=='True'
    Para['Normal traction solved from stress tensor']=Para0['Normal traction solved from stress tensor']=='True'
    Para['Rake solved from stress tensor']=Para0['Rake solved from stress tensor']=='True'
    Para['Fix_rake']=float(Para0['Fix_rake'])
    Para['Widths of VS region']=float(Para0['Widths of VS region'])
    Para['Widths of surface VS region']=float(Para0['Widths of surface VS region'])
    Para['Transition region from VS to VW region']=float(Para0['Transition region from VS to VW region'])

    Para['Reference slip rate']=float(Para0['Reference slip rate'])
    Para['Reference friction coefficient']=float(Para0['Reference friction coefficient'])
    Para['Rate-and-state parameters a in VS region']=float(Para0['Rate-and-state parameters a in VS region'])
    Para['Rate-and-state parameters b in VS region']=float(Para0['Rate-and-state parameters b in VS region'])
    Para['Characteristic slip distance in VS region']=float(Para0['Characteristic slip distance in VS region'])
    Para['Rate-and-state parameters a in VW region']=float(Para0['Rate-and-state parameters a in VW region'])
    Para['Rate-and-state parameters b in VW region']=float(Para0['Rate-and-state parameters b in VW region'])
    Para['Characteristic slip distance in VW region']=float(Para0['Characteristic slip distance in VW region'])
    Para['Rate-and-state parameters a in nucleation region']=float(Para0['Rate-and-state parameters a in nucleation region'])
    Para['Rate-and-state parameters b in nucleation region']=float(Para0['Rate-and-state parameters b in nucleation region'])
    Para['Characteristic slip distance in nucleation region']=float(Para0['Characteristic slip distance in nucleation region'])
    Para['Initial slip rate in nucleation region']=float(Para0['Initial slip rate in nucleation region'])
    Para['Plate loading rate']=float(Para0['Plate loading rate'])
    Para['ChangefriA']=Para0['ChangefriA']=='True'
    Para['Initlab']=Para0['Initlab']=='True'


    Para['Set_nucleation']=Para0['Set_nucleation']=='True'
    Para['Nuclea_posx']=float(Para0['Nuclea_posx'])
    Para['Nuclea_posy']=float(Para0['Nuclea_posy'])
    Para['Nuclea_posz']=float(Para0['Nuclea_posz'])
    Para['Radius of nucleation']=float(Para0['Radius of nucleation'])

    Para['outputvtu']=Para0['outputvtu']=='True'
    Para['outputSLIPV']=Para0['outputSLIPV']=='True'
    #Para['outputTt']=Para0['outputTt']=='True'
    Para['totaloutputsteps']=int(Para0['totaloutputsteps'])
    Para['outsteps']=int(Para0['outsteps'])
    

    return Para
    
    

