import numpy as np
from pySD.sdmout_src import pyzarr, pysetuptxt, pygbxsdat

np.set_printoptions(threshold=np.inf)

dataset = "/work/ka1298/k202167/remotes/cleo/mpi/CLEO/build/bin/fromfile_sol.zarr"
setuptxt = "/work/ka1298/k202167/remotes/cleo/mpi/CLEO/build/bin/fromfile_setup.txt"
gridfile = "/work/ka1298/k202167/remotes/cleo/mpi/CLEO/build/share/fromfile_dimlessGBxboundaries.dat"

config = pysetuptxt.get_config(setuptxt, nattrs=3, isprint=False)
consts = pysetuptxt.get_consts(setuptxt, isprint=False)
gbxs = pygbxsdat.get_gridboxes(gridfile, consts["COORD0"], isprint=False)
ds = pyzarr.get_rawdataset(dataset)

print(ds)

print("\n########## COORD1 ##########\n")
print(ds["coord1"].values)

print("\n########## COORD2 ##########\n")
print(ds["coord2"].values)

print("\n########## COORD3 ##########\n")
print(ds["coord3"].values)

print("\n########## PRESS ##########\n")
print(ds["press"].values)

print("\n########## QCOND ##########\n")
print(ds["qcond"].values)

print("\n########## QVAP ##########\n")
print(ds["qvap"].values)

print("\n########## RAGGEDCOUNT ##########\n")
print(ds["raggedcount"].values)

print("\n########## SDID ##########\n")
print(ds["sdId"].values)

print("\n########## TEMP ##########\n")
print(ds["temp"].values)

print("\n########## UVEL ##########\n")
print(ds["uvel"].values)

print("\n########## VVEL ##########\n")
print(ds["vvel"].values)

print("\n########## WVEL ##########\n")
print(ds["wvel"].values)
