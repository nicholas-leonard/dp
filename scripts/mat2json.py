import scipy.io
import optparse
import json
import numpy

parser = optparse.OptionParser()
# Input/Output data information
parser.add_option("--srcPath", action="store", dest="srcPath",
   type="string", help="path to file containing matlab data",
   default='meta_clsloc.mat')
parser.add_option("--dstPath", action="store", dest="dstPath",
   type="string", help="path to file which will contain the json data",
   default='meta_clsloc.json')
(opt, args) = parser.parse_args()
print(opt)

mat = scipy.io.loadmat(opt.srcPath)

# Convert arrays to lists, etc

def tolist(mat):
   if type(mat) is numpy.ndarray:
      mat = mat.tolist()
   elif type(mat) is list or type(mat) is tuple:
      mat = list(mat)
      
   if type(mat) is list:
      mat = [tolist(item) for item in mat]
   elif type(mat) is dict:
      for k, v in mat.items():
         mat[k] = tolist(v)
   
   return mat
   
mat = tolist(mat)
   
with open(opt.dstPath, 'w') as outfile:
    json.dump(mat, outfile)
