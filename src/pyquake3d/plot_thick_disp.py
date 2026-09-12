import numpy as np

import greenfunctions_lib.laplace_lib as laplace_lib

import greenfunctions_lib.kelvin_stress_lib as kelvin_stress_lib

import greenfunctions_lib.kelvin_stress_potential_lib  as kelvin_stress_potential_lib

import greenfunctions_lib.dislocation_lib as dislocation_lib

from pyquake3d import QDsim, config, readmsh
import pyquake3d.SH_greenfunction as SH_greenfunction



fnamegeo='examples/BP5-QD/bp5t.msh'
fnamePara='examples/BP5-QD/parameter.txt'



print('Input msh geometry file:',fnamegeo, flush=True)
print('Input parameter file:',fnamePara, flush=True)   

nodelst,elelst=readmsh.read_mshV2(fnamegeo)
print('elelst',elelst)
Para=config.readPara(fnamePara)
sim0=QDsim.QDsim(elelst,nodelst,Para)


x = np.linspace(-50000, 50000, 200)
y = np.linspace(-10000, 10000, 100)
xv, yv = np.meshgrid(x, y)



obs_points = np.vstack([xv.ravel(), yv.ravel(), np.ones_like(xv.ravel())*(-15000)]).T
obs_points = np.ascontiguousarray(obs_points, dtype=np.float64)


triangle_verticesV0=[]
for i in range(len(elelst)):
    tri_tem=[]

    P1=np.copy(nodelst[elelst[i,0]-1])
    P2=np.copy(nodelst[elelst[i,1]-1])
    P3=np.copy(nodelst[elelst[i,2]-1])
    tri_tem.append(P1)
    #P2[1]=P2[1]+0.001
    tri_tem.append(P2)
    tri_tem.append(P3)
    
    #f_list.append(fvector)
    tri_tem=np.ascontiguousarray(tri_tem)
    triangle_verticesV0.append(tri_tem.reshape([-1]))

triangle_verticesV=np.array(triangle_verticesV0,dtype=np.float64)

f_list = np.array([1.0, 0.0, 0], dtype=np.float64)


lam = 32038120320 # 示例拉梅常数
mu = 32038120320   # 示例剪切模量
nu=lam/(2.0*(lam+mu))
mode=0
# 调用 C++ 模块
# 返回形状通常为 (N_obs, M_src, 6)



# stress_core = dislocation_lib.compute_dislocation_stresses(
#     obs_points, 
#     triangle_verticesV[:],
#     f_list,
#     nu, 
#     mu,
#     eps_ratio=0.1
# )

# np.save('bpq5_slip1_stress',stress_core)


# dispcore = dislocation_lib.compute_dislocation_displacements(
#     obs_points, 
#     triangle_verticesV[:],
#     f_list,
#     nu, 
#     mu,
#     eps_ratio=0.1
# )

# np.save('bpq5_slip1_dips',dispcore)
dispcore=np.load('bpq5_slip1_dips.npy')

# np.save('surfcore/sigmaxx.npy',stress_core[:,:,0])
# np.save('surfcore/sigmayy.npy',stress_core[:,:,1])
# np.save('surfcore/sigmazz.npy',stress_core[:,:,2])
# np.save('surfcore/sigmaxy.npy',stress_core[:,:,3])
# np.save('surfcore/sigmaxz.npy',stress_core[:,:,4])
# np.save('surfcore/sigmayz.npy',stress_core[:,:,5])

# sigmaxx=np.load('surfcore/sigmaxx.npy')
# sigmayy=np.load('surfcore/sigmayy.npy')
# sigmazz=np.load('surfcore/sigmazz.npy')
# sigmaxy=np.load('surfcore/sigmaxy.npy')
# sigmaxz=np.load('surfcore/sigmaxz.npy')
# sigmayz=np.load('surfcore/sigmayz.npy')
# sigmacore=[sigmaxx,sigmayy,sigmazz,sigmaxy,sigmaxz,sigmayz]

import pyvista as pv
filename = "case1/out_vtu/step2000.vtu"
mesh = pv.read(filename)
centers_polydata = mesh.cell_centers()

# 3. 提取中心点的坐标 (N_cells, 3)
cell_centers_coords = centers_polydata.points
meshpoints=mesh.points

slip=mesh.cell_data['slip1[m]']

# #print(stress_core.shape)



# # print(slipsurface.shape)





import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Publication-style settings
mpl.rcParams.update({
    "font.size": 14,
})

dir1='near_disp_surf'
path = Path(dir1)

path.mkdir(parents=True, exist_ok=True)

sigma_list=['u_x','u_y','u_z']

for i in range(len(sigma_list)):
    # Stress fields
    stress_xx = np.dot(dispcore[:, :, i].transpose(), slip)
    stress_xx = stress_xx.reshape(xv.shape)

    fig, ax = plt.subplots(
        figsize=(12, 6),
        constrained_layout=True
    )
    vmax=np.max(stress_xx)
    pcm = ax.pcolormesh(
        xv, yv, stress_xx,
        cmap='Spectral_r',
        vmin=-vmax,
        vmax=vmax,
    )

    # Optional contour lines
    # ax.contour(
    #     xv, yv, stress_xx,
    #     levels=10,
    #     colors='k',
    #     linewidths=0.4,
    #     alpha=0.4
    # )

    ax.set_xlabel(r"$X (m)$")
    ax.set_ylabel(r"$Y (m)$")

    y_scale = 1.0 / 2
    ax.set_aspect(1 / y_scale)

    cbar = fig.colorbar(
        pcm,
        ax=ax,
        shrink=0.9,
        pad=0.02
    )
    cbar.set_label(rf"Displacement $u_{{{sigma_list[i]}}}$ (m)")

    plt.savefig(f'{dir1}/near_field_disp_{sigma_list[i]}.png', dpi=300)
    plt.show()

