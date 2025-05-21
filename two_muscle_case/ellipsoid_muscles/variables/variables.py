import sys
import os
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ellipsoid_mesh_generation

if sys.argv[-1] != "BayesOpt.py":
    n_ranks = int(sys.argv[-1])
    rank_no = int(sys.argv[-2])
else:
    n_ranks = 1

# Time stepping
dt_3D = 1            # time step of 3D mechanics
dt_splitting = 2e-3     # time step of strang splitting
dt_1D = 2e-3            # time step of 1D fiber diffusion
dt_0D = 1e-3            # time step of 0D cellml problem
end_time = 100.0         # end time of the simulation 
output_interval = dt_3D # time interval between outputs

# Material parameters
pmax = 7.3                                                  # maximum active stress
rho = 10                                                    # density of the muscle
material_parameters = [3.176e-10, 1.813, 1.075e-2, 1.0]     # [c1, c2, b, d]
diffusion_prefactor = 3.828 / (500.0 * 0.58)                # Conductivity / (Am * Cm)

force = 10.0
scenario_name = "incompressible_mooney_rivlin"

# 3D Meshes
el_x, el_y, el_z = 2, 2, 6                     # number of elements
bs_x, bs_y, bs_z = 2*el_x+1, 2*el_y+1, 2*el_z+1 # quadratic basis functions

c = 4.5 # perpendicular to fiber direction z
a = 8.8 # fiber direction z
zmin = -6.6
zmax = 7.1

physical_offset = [0, 0, 16.8]

nodes_left = ellipsoid_mesh_generation.mesh_nodes(el_x*2,el_y*2, el_z*2, c, a, zmin, zmax)
nodes_right = ellipsoid_mesh_generation.apply_offset(nodes_left,physical_offset)
print(nodes_left[0])
print(nodes_right[0])

meshes = { # create 3D mechanics mesh
    "3Dmesh_quadratic_1": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [el_x, el_y, el_z],               # number of quadratic elements in x, y and z direction
      "nodePositions":              nodes_left,      
    },
    "3Dmesh_quadratic_2": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [el_x, el_y, el_z],               # number of quadratic elements in x, y and z direction
      "nodePositions":              nodes_right,    
    }
}

# n 1D meshes left

with open("../mesh_left.json","r") as f:
	fdata_left = json.load(f)
      
fb_points_left = 99           # number of points per fiber

fiber_idx = 0
for fiber in fdata_left:
	fdict = fdata_left[fiber]
	npos = [[fdict[ii]['x'],fdict[ii]['y'],fdict[ii]['z']] for ii in range(len(fdict)-1) ]
	meshName = "fiber{}_1".format(fiber_idx)
	meshes[meshName] = {
			"nElements":		    [len(npos)-1],
			"nodePositions":	    npos,
			"inputMeshIsGlobal":	True,
			"nRanks":				n_ranks
	}
	fiber_idx += 1
     
n_fibers_left = fiber_idx

# n 1D meshes right

with open("../mesh_right.json","r") as f:
	fdata_right = json.load(f)
      
fb_points_right = 99           # number of points per fiber

fiber_idx = 0
for fiber in fdata_right:
	fdict = fdata_right[fiber]
	npos = [[fdict[ii]['x'],fdict[ii]['y'],fdict[ii]['z']] for ii in range(len(fdict)-1) ]
	meshName = "fiber{}_2".format(fiber_idx)
	meshes[meshName] = {
			"nElements":		    [len(npos)-1],
			"nodePositions":	    npos,
			"inputMeshIsGlobal":	True,
			"nRanks":				n_ranks
	}
	fiber_idx += 1
     
n_fibers_right = fiber_idx


# Define directory for cellml files
input_dir = os.path.join(os.environ.get('OPENDIHU_HOME', '../../../../../'), "examples/electrophysiology/input/")

# Fiber activation
fiber_distribution_file = input_dir + "MU_fibre_distribution_3780.txt"
firing_times_file = input_dir + "MU_firing_times_always.txt"
specific_states_call_enable_begin_2 = 1.0                     # time of first fiber activation
specific_states_call_enable_begin_1 = end_time#end_time
specific_states_call_frequency = 1e-5                       # frequency of fiber activation

tendon_damping_constant = 0.8
tendon_spring_constant = 10
tendon_spring_simulation = False