import numpy as np
from os.path import isfile
from .. import cxx2py, writebinary
from ..gbxboundariesbinary_src.read_gbxboundaries import read_dimless_gbxboundaries_binary

def thermoinputsdict(configfile, constsfile):
  ''' create values from constants file & config file
  required as inputs to create initial 
  superdroplet conditions '''

  consts = cxx2py.read_cpp_into_floats(constsfile)[0]
  moreconsts = cxx2py.derive_more_floats(consts)
  config = cxx2py.read_configtxt_into_floats(configfile)[0]

  inputs = {
    # for creating thermodynamic profiles
    "G": consts["G"],
    "CP_DRY": consts["CP_DRY"],
    "RHO_DRY": consts["RHO_DRY"],               # dry air density [Kg/m^3]
    "RGAS_DRY": moreconsts["RGAS_DRY"],
    "Mr_ratio": moreconsts["Mr_ratio"],
    "COUPLTSTEP": config["COUPLTSTEP"],
    "T_END": config["T_END"],

    # for de-dimensionalising attributes
    "W0": consts["W0"],
    "P0": consts["P0"],
    "TEMP0": consts["TEMP0"],
    "RHO0": moreconsts["RHO0"],               # characteristic density scale [Kg/m^3]
    "CP0": moreconsts["CP0"],
    "COORD0": moreconsts["COORD0"]            # z coordinate lengthscale [m]
  }

  inputs["ntime"] = round(inputs["T_END"]/inputs["COUPLTSTEP"])+1

  return inputs

class DimlessThermodynamics:

  def __init__(self, inputs=False, configfile="", constsfile=""):

    if not inputs:
      inputs = thermoinputsdict(configfile, constsfile)

    # scale_factors to de-dimensionalise data
    self.PRESS0 = inputs["P0"]
    self.TEMP0 = inputs["TEMP0"]
    self.qvap0 = 1.0
    self.qcond0 = 1.0
    self.VEL0 = inputs["W0"]
    
  def makedimless(self, THERMO):

    thermodata = {
        "press": THERMO["PRESS"] / self.PRESS0,
        "temp":THERMO["TEMP"] / self.TEMP0,
        "qvap":THERMO["qvap"],
        "qcond": THERMO["qcond"],
        "wvel":THERMO["WVEL"] / self.VEL0,
        "uvel":THERMO["UVEL"] / self.VEL0,
        "vvel":THERMO["VVEL"] / self.VEL0
      }
 
    sfs = [self.PRESS0, self.TEMP0, 1.0, 1.0]
    sfs += [self.VEL0]*3
    
    return thermodata, sfs

  def redimensionalise(self, thermo):
    
    THERMODATA = {
        "press": thermo["press"] * self.PRESS0,
        "temp":thermo["temp"] * self.TEMP0,
        "qvap":thermo["qvap"],
        "qcond": thermo["qcond"],
        "wvel":thermo["wvel"] * self.VEL0,
        "uvel":thermo["uvel"] * self.VEL0,
        "vvel":thermo["vvel"] * self.VEL0
      } 
    
    return THERMODATA

def set_arraydtype(arr, dtype):
   
  og = type(arr[0])
  if og != dtype: 
    arr = np.array(arr, dtype=dtype)

    warning = "WARNING! dtype of attributes is being changed!"+\
                " from "+str(og)+" to "+str(dtype)
    raise ValueError(warning) 

  return list(arr)

def ctype_compatible_thermodynamics(thermodata):
  ''' check type of gridbox boundaries data is compatible
  with c type double. If not, change type and raise error '''

  datatypes = [np.double]*7

  for k, key in enumerate(thermodata.keys()):

    thermodata[key] = set_arraydtype(thermodata[key], datatypes[k])
  
  return thermodata, datatypes


def check_datashape(thermodata, ndata, ngridboxes, ntime):
  ''' make sure each superdroplet attribute in data has length stated
  in ndata and that this length is compatible with the nummber of
  attributes and superdroplets expected given ndata'''
  
  err=''
  if any([n != ndata[0] for n in ndata]):
    
    err += "\n------ ERROR! -----\n"+\
          "not all variables in thermodynamics data are the"+\
            " same length, ndata = "+str(ndata)+\
              "\n---------------------\n"
  
  if any([len(d) != ngridboxes*ntime for d in thermodata.values()]): 
    err += "inconsistent dimensions of thermodynamic data: "+\
            str(ndata)+". Lengths should all = "+str(ngridboxes*ntime)+\
            " since data should be list of [ntimesteps]*ngridboxes"   

  if err: 
    raise ValueError(err)
  
def write_thermodynamics_binary(thermofile, thermogen, configfile,
                                constsfile, gridfile):
  
  if not isfile(gridfile):
    errmsg = "gridfile not found, but must be"+\
              " created before initSDsfile can be"
    raise ValueError(errmsg)

  inputs = thermoinputsdict(configfile, constsfile)
  gbxbounds = read_dimless_gbxboundaries_binary(gridfile,
                                                COORD0=inputs["COORD0"])
  

  ngridboxes = len(gbxbounds.keys())
  thermodata = thermogen.generate_thermo(ngridboxes, inputs["ntime"])

  dth = DimlessThermodynamics(inputs=inputs)
  thermodata, scale_factors = dth.makedimless(thermodata)

  ndata = [len(dt) for dt in thermodata.values()]
  
  data, datatypes = ctype_compatible_thermodynamics(thermodata) 
  check_datashape(data, ndata, ngridboxes, inputs["ntime"])

  units = [b'P', b'K', b' ', b' ']
  units += [b'm']*3 # velocity units

  scale_factors = np.asarray(scale_factors, dtype=np.double)

  filestem, filetype = thermofile.split(".")
  ng, nt = str(ngridboxes), str(inputs["ntime"])
  vars = ["press", "temp", "qvap", "qcond", "wvel", "vvel", "uvel"]
  for v, var in enumerate(vars):

    metastr = 'Variable in this file is flattened array of '+var+\
              ' with original dimensions [ngridboxes, time] = ['+\
              ng+', '+nt+'] (ie. file contains '+str(ndata[v])+\
              ' datapoints corresponding to '+ng+' gridboxes over '+\
              nt+' time steps)'
    
    filename = filestem+"_"+var+"."+filetype
    writebinary.writebinary(filename, thermodata[var],
                            [ndata[v]], [datatypes[v]],
                            [units[v]], [scale_factors[v]],
                            metastr)