from matplotlib import cm
import numpy as np

import greenfunctions_lib.laplace_lib as laplace_lib

import greenfunctions_lib.kelvin_stress_lib as kelvin_stress_lib

import greenfunctions_lib.kelvin_stress_potential_lib  as kelvin_stress_potential_lib

import greenfunctions_lib.dislocation_lib as dislocation_lib

from pyquake3d import DH_greenfunction, QDsim, config, readmsh
import pyquake3d.SH_greenfunction as SH_greenfunction

import json
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
# 读取 ParaView / VTK 风格的 colorbar JSON
with open("surfcore/PuOr.json", "r") as f:
    data = json.load(f)

rgb_points = data[0]["RGBPoints"]
name = data[0]["Name"]

# 每 4 个数是一组: value, R, G, B
points = np.array(rgb_points).reshape(-1, 4)

values = points[:, 0]
colors = points[:, 1:4]

# Matplotlib 的 colormap 位置需要归一化到 0~1
values_norm = (values - values.min()) / (values.max() - values.min())

# 构造 colormap
cmap = LinearSegmentedColormap.from_list(
    name,
    list(zip(values_norm, colors))
)



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



import pyvista as pv
filename = "case1/out_vtu3/step2000.vtu"
mesh = pv.read(filename)
centers_polydata = mesh.cell_centers()
 
# 3. 提取中心点的坐标 (N_cells, 3)
cell_centers_coords = centers_polydata.points
meshpoints=mesh.points

slip=mesh.cell_data['slip1[m]']-1.2

# #print(stress_core.shape)



# # print(slipsurface.shape)

def phi(r, eps):
    return (15 * eps**4) / (8 * np.pi * (r**2 + eps**2)**(7/2))



import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


Y = np.linspace(-2000, 2000, 400)
eps1=0.05


obs_points = np.vstack([np.ones_like(Y.ravel())*0, Y.ravel(), np.ones_like(Y.ravel())*(-15000)]).T
obs_points = np.ascontiguousarray(obs_points, dtype=np.float64)



stress_core = dislocation_lib.compute_dislocation_stresses(
    obs_points, 
    triangle_verticesV[:],
    f_list,
    nu, 
    mu,
    eps_ratio=0.2
)


dispcore = dislocation_lib.compute_dislocation_displacements(
    obs_points, 
    triangle_verticesV[:],
    f_list,
    nu, 
    mu,
    eps_ratio=0.2
)

stress_xy2 = np.dot(stress_core[:,:,3].transpose(), slip)
ux2 = np.dot(dispcore[:,:,0].transpose(), slip)


fig, ax1 = plt.subplots(figsize=(12, 5))

# 左侧：位移
line1 = ax1.plot(Y, ux2, c='black', linewidth=2, label=rf'$u_x$')
ax1.set_xlabel('Y (m)')
ax1.set_ylabel(rf'$u_x$ (m)', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# 右侧：应力
ax2 = ax1.twinx()

line2 = ax2.plot(Y, stress_xy2 * 1e-6, c='brown', linewidth=2, label=rf'$\sigma_{{xy}}$')
ax2.set_ylabel(rf'$\sigma_{{xy}}$ (MPa)', color='darkred')
ax2.tick_params(axis='y', labelcolor='darkred')

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best')

plt.tight_layout()
plt.savefig('near_stress_surf/near_field_disp_stress2.png', dpi=300)
plt.show()


Stres_=[]
Vx_=[]
timeindex=[]
for i in range(400):
    k=i*100
    timeindex.append(k)
    filename = f"case1/out_vtu3/step{k}.vtu"
    mesh = pv.read(filename)
    centers_polydata = mesh.cell_centers()
    
    # 3. 提取中心点的坐标 (N_cells, 3)
    cell_centers_coords = centers_polydata.points
    meshpoints=mesh.points

    slipv=-mesh.cell_data['Slipv1[m/s]']+1e-9
    stress_xy = np.dot(stress_core[:,:,3].transpose(), slipv)
    vx = np.dot(dispcore[:,:,0].transpose(), slipv)

    Stres_.append(stress_xy)
    Vx_.append(vx)


Stres_=np.array(Stres_)
Vx_=np.array(Vx_)



timeindex=np.array(timeindex)
# plt.pcolor(Y, timeindex, Vx_,cmap='Spectral_r')
# plt.xlabel('Y (m)')
# plt.ylabel('Time step')
# plt.colorbar()
# plt.show()


import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 16,          # 比14更适合单栏论文
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


fig, ax = plt.subplots(figsize=(7, 7))

pcm = ax.pcolormesh(
    Y,
    timeindex,
    Vx_,
    cmap="Spectral_r",
    shading="auto"
)

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(r'$V_x$ (m/s)')

# 坐标轴细节
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Time steps")
plt.savefig('near_stress_surf/disp_ve_elvo.png', dpi=300)
plt.show()




fig, ax = plt.subplots(figsize=(7, 7))

pcm = ax.pcolormesh(
    Y,
    timeindex,
    Stres_*1e-6,
    cmap=cmap,
    shading="auto",
    vmin=-np.max(np.abs(Stres_*1e-6)),
    vmax=np.max(np.abs(Stres_*1e-6))
)

cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(r'$\dot{\sigma}_{xy}$ (MPa/s)')

# 坐标轴细节
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Time steps")
plt.savefig('near_stress_surf/disp_stress_elvo.png', dpi=300)
plt.show()








norm = plt.Normalize(vmin=20, vmax=60)
cmap = cm.get_cmap('viridis')

plt.figure(figsize=(7, 6))

k=2000
filename = f"case1/out_vtu3/step{k}.vtu"
mesh = pv.read(filename)
centers_polydata = mesh.cell_centers()

# 3. 提取中心点的坐标 (N_cells, 3)
cell_centers_coords = centers_polydata.points
meshpoints=mesh.points

slipv0=mesh.cell_data['slip[m]']

# 2. 循环绘图
for i in range(20, 60):
    k=i*100
    filename = f"case1/out_vtu3/step{k}.vtu"
    mesh = pv.read(filename)
    centers_polydata = mesh.cell_centers()
    
    # 3. 提取中心点的坐标 (N_cells, 3)
    cell_centers_coords = centers_polydata.points
    meshpoints=mesh.points

    slipv=mesh.cell_data['slip[m]']-slipv0

    stress_xy = np.dot(stress_core[:,:,3].transpose(), slipv)
    color = cmap(norm(i))
    
    plt.plot(Y, stress_xy * 1e-6, color=color, alpha=0.7, lw=1.5)

# 3. 添加色标 (Colorbar)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Step Index')

# 4. 样式美化
plt.xlabel('Y (m)')
plt.ylabel(r'$\sigma_{xy}$ (MPa)')
plt.grid(True, linestyle='--', alpha=0.5) # 添加网格更利于阅读数据

plt.tight_layout()
plt.show()






















stressesN=[]
Ux=[]
for i in range(len(elelst)):
    print(i)
    P1=np.copy(nodelst[elelst[i,0]-1])
    P2=np.copy(nodelst[elelst[i,1]-1])
    P3=np.copy(nodelst[elelst[i,2]-1])

    Ss,Ds,Ts=1,0,0
    StsMS,StrMS=SH_greenfunction.TDstressFS(obs_points[:,0],obs_points[:,1],obs_points[:,2],P1,P2,P3,Ss,Ds,Ts,mu,lam)
    stressesN.append(StsMS)

    ue,un,uv=DH_greenfunction.TDdispFS(obs_points[:,0],obs_points[:,1],obs_points[:,2],P1,P2,P3,Ss,Ds,Ts,nu)
    Ux.append(ue)

Ux=np.array(Ux)
stressesN=np.array(stressesN)

stresses=stressesN.transpose([0,2,1])
# stresses=np.sum(stresses,axis=0)

# Ux=np.sum(Ux,axis=0)



plt.figure(figsize=(7, 6))

k=2000
filename = f"case1/out_vtu3/step{k}.vtu"
mesh = pv.read(filename)
centers_polydata = mesh.cell_centers()

# 3. 提取中心点的坐标 (N_cells, 3)
cell_centers_coords = centers_polydata.points
meshpoints=mesh.points

slipv0=mesh.cell_data['slip[m]']

# 2. 循环绘图
for i in range(20, 60):
    k=i*100
    filename = f"case1/out_vtu3/step{k}.vtu"
    mesh = pv.read(filename)
    centers_polydata = mesh.cell_centers()
    
    # 3. 提取中心点的坐标 (N_cells, 3)
    cell_centers_coords = centers_polydata.points
    meshpoints=mesh.points

    slipv=mesh.cell_data['slip[m]']-slipv0

    stress_xy = np.dot(stresses[:,:,3], slipv)
    color = cmap(norm(i))
    
    plt.plot(Y, stress_xy * 1e-6, color=color, alpha=0.7, lw=1.5)

# 3. 添加色标 (Colorbar)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Step Index')

# 4. 样式美化
plt.xlabel('Y (m)')
plt.ylabel(r'$\sigma_{xy}$ (MPa)')
plt.grid(True, linestyle='--', alpha=0.5) # 添加网格更利于阅读数据

plt.tight_layout()
plt.show()
