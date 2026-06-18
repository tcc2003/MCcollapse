import numpy as np
import h5py
import math

### 1. Parameters
BoxSize = 3.0    
CloudRadius = 1.0
CloudMass = 100.0

u_cloud = 0.39         # 10K
u_bg = 39.0            # 1000K

center = np.array([BoxSize/2, BoxSize/2, BoxSize/2])

### 2. Molecular cloud
N1d_cloud = 100
spacing_c = (CloudRadius * 2) / N1d_cloud #resolution
xc = np.linspace(center[0] - CloudRadius + spacing_c/2, center[0] + CloudRadius - spacing_c/2, N1d_cloud)
Xc, Yc, Zc = np.meshgrid(xc, xc, xc, indexing='ij')
pos_c_box = np.vstack([Xc.ravel(), Yc.ravel(), Zc.ravel()]).T

# particles inside 1pc
r_c = np.linalg.norm(pos_c_box - center, axis=1)
in_cloud = r_c <= CloudRadius
pos_cloud = pos_c_box[in_cloud]
N_cloud = len(pos_cloud)

### 3. background
N1d_bg = 64  
spacing_bg = BoxSize / N1d_bg
xb = np.linspace(spacing_bg/2, BoxSize - spacing_bg/2, N1d_bg)
Xb, Yb, Zb = np.meshgrid(xb, xb, xb, indexing='ij')
pos_b_box = np.vstack([Xb.ravel(), Yb.ravel(), Zb.ravel()]).T

# particles outside 1pc
r_b = np.linalg.norm(pos_b_box - center, axis=1)
in_bg = r_b > CloudRadius
pos_bg = pos_b_box[in_bg]
N_bg = len(pos_bg)


### 4. Combine cloud and background gas
pos_total = np.vstack([pos_cloud, pos_bg])
N_total = N_cloud + N_bg
ids = np.arange(1, N_total + 1)

masses = np.zeros(N_total)
u_energy = np.zeros(N_total)

# mass of molecular cloud
m_cloud_part = CloudMass / N_cloud
masses[:N_cloud] = m_cloud_part
u_energy[:N_cloud] = u_cloud

# mass of background gas
vol_cloud = (4.0/3.0) * math.pi * (CloudRadius**3)
rho_cloud = CloudMass / vol_cloud
rho_bg = rho_cloud * 0.01

vol_bg = (BoxSize**3) - vol_cloud
bg_mass_total = rho_bg * vol_bg
m_bg_part = bg_mass_total / N_bg

masses[N_cloud:] = m_bg_part
u_energy[N_cloud:] = u_bg

vel = np.zeros((N_total, 3))

### 5. Write the ic file
filename = "ic_r01.hdf5"
with h5py.File(filename, 'w') as f:
    header = f.create_group("Header")
    numpart = np.array([N_total, 0, 0, 0, 0, 0], dtype=np.int32)

    header.attrs["NumPart_ThisFile"] = numpart
    header.attrs["NumPart_Total"] = numpart

    header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.int32)

    header.attrs["MassTable"] = np.zeros(6, dtype=np.float64) 
    header.attrs["BoxSize"] = BoxSize
    header.attrs["NumFilesPerSnapshot"] = 1
    header.attrs["Time"] = 0.0

    header.attrs["Omega0"] = 0.0
    header.attrs["OmegaLambda"] = 0.0
    header.attrs["HubbleParam"] = 1.0
    
    header.attrs["Redshift"] = 0.0
    header.attrs["Flag_Sfr"] = 0
    header.attrs["Flag_Cooling"] = 0
    header.attrs["Flag_Feedback"] = 0
    header.attrs["Flag_StellarAge"] = 0
    header.attrs["Flag_Metals"] = 0
    header.attrs["Flag_DoublePrecision"] = 1 

    p0 = f.create_group("PartType0")
    p0.create_dataset("Coordinates", data=pos_total, dtype=np.float64)
    p0.create_dataset("Velocities", data=vel, dtype=np.float32)
    p0.create_dataset("Masses", data=masses, dtype=np.float32)
    p0.create_dataset("InternalEnergy", data=u_energy, dtype=np.float32)
    p0.create_dataset("ParticleIDs", data=ids, dtype=np.int32)

print("Complete!")
#print(f"Cloud: {N_cloud} particles, backgorund: {N_bg} particles")
#print(f"Cloud mass: {m_cloud_part:.6e}, background mass: {m_bg_part:.6e} ")