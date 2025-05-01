# This settings file can be used for two different equations:
# - Isotropic hyperelastic material
# - Linear elasticity
#
# arguments: <scenario_name> <force> <individuality_parameter>


import numpy as np
import sys, os
import time

script_path = os.path.dirname(os.path.abspath(__file__))
var_path = os.path.join(script_path, "variables")
sys.path.insert(0, var_path)

import variables

n_ranks = (int)(sys.argv[-1])

# parameters
force = 10.0                       # [N] load on top
material_parameters = [3.176e-10, 1.813, 1.075e-2, 1.0]     # [c1, c2, b, d]

constant_body_force = None                                                                      
scenario_name = "tensile_test"
dirichlet_bc_mode = "fix_floating"

if len(sys.argv) > 3:                                                                           
  scenario_name = sys.argv[0]
  force = float(sys.argv[1])
  print("scenario_name: {}".format(scenario_name))
  print("force: {}".format(force))
    
  # set material parameters depending on scenario name
  if scenario_name == "compressible_mooney_rivlin":
    material_parameters = [3.176e-10, 1.813, 10]      # c1, c2, c
    
  elif scenario_name == "compressible_mooney_rivlin_decoupled":
    material_parameters = [3.176e-10, 1.813, 10.0]      # c1, c2, kappa
    
  elif scenario_name == "incompressible_mooney_rivlin":
    material_parameters = [3.176e-10, 1.813]      # c1, c2
    
  elif scenario_name == "nearly_incompressible_mooney_rivlin":
    material_parameters = [3.176e-10, 1.813, 1e3]      # c1, c2, kappa

  elif scenario_name == "nearly_incompressible_mooney_rivlin_decoupled":
    material_parameters = [3.176e-10, 1.813, 1e3]      # c1, c2, kappa

  elif scenario_name == "linear":
    pass

  elif scenario_name == "nearly_incompressible_mooney_rivlin_febio":
    material_parameters = [3.176e-10, 1.813, 1e3]      # c1, c2, kappa

  else:
    print("Error! Please specify the correct scenario, see settings.py for allowed values.\n")
    quit()

if len(sys.argv) > 4:
  individuality_parameter = sys.argv[2] 
else:
  individuality_parameter = str(time.time())

nx, ny, nz = 3, 3, 12                     # number of elements
mx, my, mz = 2*nx+1, 2*ny+1, 2*nz+1 # quadratic basis functions

fb_x, fb_y = 10, 10         # number of fibers
fb_points = 100             # number of points per fiber
fiber_direction_1 = [0, 0, 1] # direction of fiber in element
fiber_direction_2 = [0, 0, 1]

def get_fiber_no(fiber_x, fiber_y):
    return fiber_x + fiber_y*fb_x

physical_extent = [3.0, 3.0, 12.0]

physical_offset_1 = [0, 0, 0]
physical_offset_2 = [0, 0, 14.0]

meshes = { # create 3D mechanics mesh
    "3Dmesh_quadratic_1": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [nx, ny, nz],               # number of quadratic elements in x, y and z direction
      "physicalExtent":             physical_extent,            # physical size of the box
      "physicalOffset":             physical_offset_1,          # offset/translation where the whole mesh begins
    },
    "3Dmesh_quadratic_2": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [nx, ny, nz],               # number of quadratic elements in x, y and z direction
      "physicalExtent":             physical_extent,            # physical size of the box
      "physicalOffset":             physical_offset_2,                  # offset/translation where the whole mesh begins
    },
    "3Dmesh_1": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [mx, my, mz],               # number of quadratic elements in x, y and z direction
      "physicalExtent":             physical_extent,            # physical size of the box
      "physicalOffset":             physical_offset_1,          # offset/translation where the whole mesh begins
    },
    "3Dmesh_2": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [mx, my, mz],               # number of quadratic elements in x, y and z direction
      "physicalExtent":             physical_extent,            # physical size of the box
      "physicalOffset":             physical_offset_2,                  # offset/translation where the whole mesh begins
    }
}

for fiber_x in range(fb_x):
    for fiber_y in range(fb_y):
        fiber_no = get_fiber_no(fiber_x, fiber_y)
        x = physical_extent[0] * fiber_x / (fb_x - 1)
        y = physical_extent[1] * fiber_y / (fb_y - 1)
        nodePositions_1 = [[x, y, physical_extent[2] * i / (fb_points - 1)] for i in range(fb_points)]
        nodePositions_2 = [[x, y, physical_offset_2[2]+physical_extent[2] * i / (fb_points - 1)] for i in range(fb_points)]
        meshName_1 = "fiber{}_1".format(fiber_no)
        meshes[meshName_1] = { # create fiber meshes
            "nElements":            [fb_points - 1],
            "nodePositions":        nodePositions_1,
            "inputMeshIsGlobal":    True,
            "nRanks":               n_ranks
        }
        meshName_2 = "fiber{}_2".format(fiber_no)
        meshes[meshName_2] = { # create fiber meshes
            "nElements":            [fb_points - 1],
            "nodePositions":        nodePositions_2,
            "inputMeshIsGlobal":    True,
            "nRanks":               n_ranks
        }

# set Dirichlet BC, fix bottom
elasticity_dirichlet_bc_1 = {}
elasticity_dirichlet_bc_2 = {}
k1 = 0
k2=mz-1
# fix z value on the whole x-y-plane
for j in range(my):
  for i in range(mx):
    elasticity_dirichlet_bc_1[k1*mx*my + j*mx + i] = [None,None,0.0,None,None,None]
    elasticity_dirichlet_bc_2[k2*mx*my + j*mx + i] = [None,None,0.0,None,None,None]

# fix left edge 
for j in range(my):
  elasticity_dirichlet_bc_1[k1*mx*my + j*mx + 0][0] = 0.0
  elasticity_dirichlet_bc_2[k2*mx*my + j*mx + 0][0] = 0.0
  
# fix front edge 
for i in range(mx):
  elasticity_dirichlet_bc_1[k1*mx*my + 0*mx + i][1] = 0.0
  elasticity_dirichlet_bc_2[k2*mx*my + 0*mx + i][1] = 0.0
       
# set Neumann BC, set traction at the top
k = nz-1
traction_vector = [0, 0, force]     # the traction force in specified in the reference configuration

#elasticity_neumann_bc = [{"element": k*nx*ny + j*nx + i, "constantVector": traction_vector, "face": "2+"} for j in range(ny) for i in range(nx)]
elasticity_neumann_bc_1 = {}
elasticity_neumann_bc_2 = {}

# callback for result
def handle_result_prestretch(result):
  data = result[0]

  number_of_nodes = mx * my
  average_z_start = 0
  average_z_end = 0

  z_data = data["data"][0]["components"][2]["values"]

  for i in range(number_of_nodes):
    average_z_start += z_data[i]
    average_z_end += z_data[number_of_nodes*(mz -1) + i]

  average_z_start /= number_of_nodes
  average_z_end /= number_of_nodes

  length_of_muscle = np.abs(average_z_end - average_z_start)
  print("length of muscle (prestretch): ", length_of_muscle)

  if data["timeStepNo"] == 0:
    f = open("muscle_length_prestretch"+individuality_parameter+".csv", "w")
    f.write(str(length_of_muscle))
    f.write(",")
    f.close()
  else:
    f = open("muscle_length_prestretch"+individuality_parameter+".csv", "a")
    f.write(str(length_of_muscle))
    f.write(",")
    f.close()
  
  
  if data["timeStepNo"] == 1:
    field_variables = data["data"]
    
    strain = max(field_variables[1]["components"][2]["values"])
    stress = max(field_variables[5]["components"][2]["values"])
    
    print("strain: {}, stress: {}".format(strain, stress))
    
    with open("result.csv","a") as f:
      f.write("{},{},{}\n".format(scenario_name,strain,stress))


def callback_function_contraction_1(raw_data):
  global elasticity_neumann_bc_2
  t = raw_data[0]["currentTime"]
  number_of_nodes = variables.bs_x * variables.bs_y
  average_z_start = 0
  average_z_end = 0

  z_data = raw_data[0]["data"][0]["components"][2]["values"]

  for i in range(number_of_nodes):
    average_z_start += z_data[i]
    average_z_end += z_data[number_of_nodes*(variables.bs_z -1) + i]

  average_z_start /= number_of_nodes
  average_z_end /= number_of_nodes

  length_of_muscle = np.abs(average_z_end - average_z_start)
  print("length of muscle 1 (contraction): ", length_of_muscle)

  if t == variables.dt_3D:
    f = open("muscle_length_contraction"+individuality_parameter+"_1.csv", "w")
    f.write(str(length_of_muscle))
    f.write(",")
    f.close()
  else:
    f = open("muscle_length_contraction"+individuality_parameter+"_1.csv", "a")
    f.write(str(length_of_muscle))
    f.write(",")
    f.close()

  force_data = raw_data[0]["data"][6]["components"][2]["values"]
  for i in range(variables.bs_x):
    for j in range(variables.bs_y):
      force = force_data[(variables.bs_z-1)*variables.bs_x*variables.bs_y + j*variables.bs_x + i]
      traction_vector = [0,0,-force]
      elasticity_neumann_bc_2 = {}#[{"element": (variables.bs_z-1)*variables.bs_x*variables.bs_y + j*variables.bs_x + i, "constantVector": traction_vector, "face": "2-"}]

def callback_function_contraction_2(raw_data):
  global elasticity_neumann_bc_1
  t = raw_data[0]["currentTime"]
  number_of_nodes = variables.bs_x * variables.bs_y
  average_z_start = 0
  average_z_end = 0

  z_data = raw_data[0]["data"][0]["components"][2]["values"]

  for i in range(number_of_nodes):
    average_z_start += z_data[i]
    average_z_end += z_data[number_of_nodes*(variables.bs_z -1) + i]

  average_z_start /= number_of_nodes
  average_z_end /= number_of_nodes

  length_of_muscle = np.abs(average_z_end - average_z_start)
  print("length of muscle 2 (contraction): ", length_of_muscle)

  if t == variables.dt_3D:
    f = open("muscle_length_contraction"+individuality_parameter+"_2.csv", "w")
    f.write(str(length_of_muscle))
    f.write(",")
    f.close()
  else:
    f = open("muscle_length_contraction"+individuality_parameter+"_2.csv", "a")
    f.write(str(length_of_muscle))
    f.write(",")
    f.close()

  force_data = raw_data[0]["data"][6]["components"][2]["values"]
  for i in range(variables.bs_x):
    for j in range(variables.bs_y):
      force = force_data[j*variables.bs_x + i]
      traction_vector = [0,0,-force]
      elasticity_neumann_bc_1 = {}#[{"element": j*variables.bs_x + i, "constantVector": traction_vector, "face": "2+"}]


config = {
  "scenarioName":                 scenario_name,                # scenario name to identify the simulation runs in the log file
  "logFormat":                    "csv",                        # "csv" or "json", format of the lines in the log file, csv gives smaller files
  "solverStructureDiagramFile":   "solver_structure.txt",       # output file of a diagram that shows data connection between solvers
  "mappingsBetweenMeshesLogFile": "mappings_between_meshes_log.txt",    # log file for mappings 

  "Meshes": meshes,
  #   "MappingsBetweenMeshes": { 
  #     "3Dmesh_1" : ["fiber{}_1".format(variables.get_fiber_no(fiber_x, fiber_y)) for fiber_x in range(variables.fb_x) for fiber_y in range(variables.fb_y)],
  #     "3Dmesh_2" : ["fiber{}_2".format(variables.get_fiber_no(fiber_x, fiber_y)) for fiber_x in range(variables.fb_x) for fiber_y in range(variables.fb_y)]

  # },
  # "MappingsBetweenMeshes": 
  #   {"fiber{}_{}".format(f,m) : ["3Dmesh_quadratic_{}".format(m)] for f in range(variables.fb_x*variables.fb_y) for m in [1,2]},

  "Solvers": {
    "linearElasticitySolver": {           # solver for linear elasticity
      "relativeTolerance":  1e-10,
      "absoluteTolerance":  1e-10,         # 1e-10 absolute tolerance of the residual    ,
      "maxIterations":      1e4,
      "solverType":         "gmres",
      "preconditionerType": "none",
      "dumpFilename":       "",
      "dumpFormat":         "matlab",
    }, 
    "diffusionSolver": {
      "solverType":                     "cg",
      "preconditionerType":             "none",
      "relativeTolerance":              1e-10,
      "absoluteTolerance":              1e-10,
      "maxIterations":                  1e4,
      "dumpFilename":                   "",
      "dumpFormat":                     "matlab"
    },
    "mechanicsSolver": {
      "solverType":                     "preonly",
      "preconditionerType":             "lu",
      "relativeTolerance":              1e-10,
      "absoluteTolerance":              1e-10,
      "maxIterations":                  1e4,
      "snesLineSearchType":             "l2",
      "snesRelativeTolerance":          1e-5,
      "snesAbsoluteTolerance":          1e-5,
      "snesMaxIterations":              10,
      "snesMaxFunctionEvaluations":     1e8,
      "snesRebuildJacobianFrequency":   5,
      "dumpFilename":                   "",
      "dumpFormat":                     "matlab"
    }
  },

  "Coupling": {
    "timeStepWidth": variables.dt_3D,
    "endTime": variables.end_time,
    "connectedSlotsTerm1To2": None,
    "connectedSlotsTerm2To1": None,
    
    "Term1": {
      "Coupling": {
        "timeStepWidth":            variables.dt_3D,
        "logTimeStepWidthAsKey":    "dt_3D",
        "durationLogKey":           "duration_3D",
        "connectedSlotsTerm1To2":   {1:2},  # transfer stress to MuscleContractionSolver gamma
        "connectedSlotsTerm2To1":   None,   # transfer nothing back

        "Term1": { # fibers (FastMonodomainSolver)
          "MultipleInstances": { 
            "ranksAllComputedInstances":    list(range(n_ranks)),
            "nInstances":                   1,

            "instances": [{
              "ranks": [0],

              "StrangSplitting": {
                "timeStepWidth":            variables.dt_splitting,
                "logTimeStepWidthAsKey":    "dt_splitting",
                "durationLogKey":           "duration_splitting",
                "timeStepOutputInterval":   100,
                "connectedSlotsTerm1To2":   None, #{0:0,1:1,2:2,3:3,4:4},
                "connectedSlotsTerm2To1":   None, #{0:0,1:1,2:2,3:3,4:4},

                "Term1": { # reaction term
                  "MultipleInstances": {
                    "nInstances":   variables.fb_x * variables.fb_y,

                    "instances": [{
                      "ranks": [0],

                      "Heun": {
                        "timeStepWidth":            variables.dt_0D,
                        "logTimeStepWidthAsKey":    "dt_0D",
                        "durationLogKey":           "duration_0D",
                        "timeStepOutputInterval":   100,

                        "initialValues":                [],
                        "dirichletBoundaryConditions":  {},
                        "dirichletOutputFilename":      None,
                        "inputMeshIsGlobal":            True,
                        "checkForNanInf":               False,
                        "nAdditionalFieldVariables":    0,
                        "additionalSlotNames":          [],
                        "OutputWriter":                 [],

                        "CellML": {
                          "modelFilename":          variables.input_dir + "hodgkin_huxley-razumova.cellml",
                          "meshName":               "fiber{}_1".format(variables.get_fiber_no(fiber_x, fiber_y)), 
                          "stimulationLogFilename": "out/" + scenario_name + "stimulation.log",

                          "statesInitialValues":                        [],
                          "initializeStatesToEquilibrium":              False,
                          "initializeStatesToEquilibriumTimeStepWidth": 1e-4,
                          "optimizationType":                           "vc",
                          "approximateExponentialFunction":             True,
                          "compilerFlags":                              "-fPIC -O3 -march=native -Wno-deprecated_declarations -shared",
                          "maximumNumberOfThreads":                     0,

                          "setSpecificStatesCallEnableBegin":       variables.specific_states_call_enable_begin,
                          "setSpecificStatesCallFrequency":         variables.specific_states_call_frequency,
                          "setSpecificStatesRepeatAfterFirstCall":  0.01,
                          "setSpecificStatesFrequencyJitter":       [0] ,
                          "setSpecificStatesCallInterval":          0,
                          "setSpecificStatesFunction":              None,
                          "additionalArgument":                     None, 

                          "mappings": {
                            ("parameter", 0):               "membrane/i_Stim",
                            ("parameter", 1):               "Razumova/l_hs",
                            ("parameter", 2):               ("constant", "Razumova/rel_velo"),
                            ("connectorSlot", "vm1"):        "membrane/V",
                            ("connectorSlot", "stress1"):    "Razumova/activestress",
                            ("connectorSlot", "alpha1"):     "Razumova/activation",
                            ("connectorSlot", "lambda1"):    "Razumova/l_hs",
                            ("connectorSlot", "ldot1"):      "Razumova/rel_velo"
                          },
                          "parametersInitialValues": [0.0, 1.0, 0.0],
                        },
                      }
                    } for fiber_x in range(variables.fb_x) for fiber_y in range(variables.fb_y)] 
                  }
                },

                "Term2": { # diffusion term
                  "MultipleInstances": {
                    "nInstances": variables.fb_x * variables.fb_y, 

                    "OutputWriter": [
                      {
                        "format":             "Paraview",
                        "outputInterval":     int(1.0 / variables.dt_3D * variables.output_interval),
                        "filename":           "out/" + scenario_name + "/fibers_1",
                        "fileNumbering":      "incremental",
                        "binary":             True,
                        "fixedFormat":        False,
                        "onlyNodalValues":    True,
                        "combineFiles":       True
                      }
                    ],

                    "instances": [{
                      "ranks": [0],

                      "ImplicitEuler": {
                        "timeStepWidth":            variables.dt_1D,
                        "logTimeStepWidthAsKey":    "dt_1D",
                        "durationLogKey":           "duration_1D",
                        "timeStepOutputInterval":   100,

                        "nAdditionalFieldVariables":    4,
                        "additionalSlotNames":          ["stress1", "alpha1", "lambda1", "ldot1"],

                        "solverName":                       "diffusionSolver",
                        "timeStepWidthRelativeTolerance":   1e-10,

                        "dirichletBoundaryConditions":      {},
                        "dirichletOutputFilename":          None,
                        "inputMeshIsGlobal":                True,
                        "checkForNanInf":                   False,
                        "OutputWriter":                     [],

                        "FiniteElementMethod": {
                          "meshName":           "fiber{}_1".format(variables.get_fiber_no(fiber_x, fiber_y)),
                          "inputMeshIsGlobal":  True,
                          "solverName":         "diffusionSolver",
                          "prefactor":          variables.diffusion_prefactor,
                          "slotName":           "vm1"
                        }
                      }
                    } for fiber_x in range(variables.fb_x) for fiber_y in range(variables.fb_y)]
                  }
                }
              }
            }]
          },

          "fiberDistributionFile":                              variables.fiber_distribution_file,
          "firingTimesFile":                                    variables.firing_times_file,
          "valueForStimulatedPoint":                            20.0,
          "onlyComputeIfHasBeenStimulated":                     True,
          "disableComputationWhenStatesAreCloseToEquilibrium":  True,
          "neuromuscularJunctionRelativeSize":                  0.0,################################change for no randomness
          "generateGPUSource":                                  True,
          "useSinglePrecision":                                 False
        },

        "Term2": { # solid mechanics (MuscleContractionSolver)
          "MuscleContractionSolver": {
            "Pmax":                         variables.pmax,
            # "slotNames":                    ["lambdaContraction", "ldotContraction", "gammaContraction", "TContraction"],
            "slotNames":                    ["lambdaContraction1", "ldotContraction1", "gammaContraction1", "TContraction1"],
            "dynamic":                      True,

            "numberTimeSteps":              1,
            "timeStepOutputInterval":       100,
            "lambdaDotScalingFactor":       1,
            "enableForceLengthRelation":    True,
            "mapGeometryToMeshes":          ["fiber{}_1".format(variables.get_fiber_no(fiber_x, fiber_y)) for fiber_x in range(fb_x) for fiber_y in range(fb_y)],

            "OutputWriter": [
              {
                "format":             "Paraview",
                "outputInterval":     int(1.0 / variables.dt_3D * variables.output_interval),
                "filename":           "out/" + scenario_name + "/mechanics_1",
                "fileNumbering":      "incremental",
                "binary":             True,
                "fixedFormat":        False,
                "onlyNodalValues":    True,
                "combineFiles":       True
              }
            ],

            "DynamicHyperelasticitySolver": {
              "durationLogKey":         "duration_3D",
              "logTimeStepWidthAsKey":  "dt_3D",
              "numberTimeSteps":        1,
              "materialParameters":     variables.material_parameters,
              "density":                variables.rho,
              "timeStepOutputInterval": 1,

              "meshName":                   "3Dmesh_quadratic_1",
              "inputMeshIsGlobal":          True,
              "fiberMeshNames":             "fiber{}_1".format(variables.get_fiber_no(fiber_x, fiber_y)),

              "solverName":                 "mechanicsSolver",
              "displacementsScalingFactor":  1.0,
              "useAnalyticJacobian":        True,
              "useNumericJacobian":         False,
              "dumpDenseMatlabVariables":   False,
              "loadFactorGiveUpThreshold":  1,
              "loadFactors":                [],
              "scaleInitialGuess":          False,
              "extrapolateInitialGuess":    True,
              "nNonlinearSolveCalls":       1,

              "dirichletBoundaryConditions":                            elasticity_dirichlet_bc_1, #variables.dirichlet_bc,
              "neumannBoundaryConditions":                              elasticity_neumann_bc_1, #elasticity_neumann_bc, #variables.neumann_bc,
              "updateDirichletBoundaryConditionsFunction":              None,
              "updateDirichletBoundaryConditionsFunctionCallInterval":  1,
              "divideNeumannBoundaryConditionValuesByTotalArea":        True,

              "initialValuesDisplacements": [[0, 0, 0] for _ in range(variables.bs_x * variables.bs_y * variables.bs_z)],
              "initialValuesVelocities":    [[0, 0, 0] for _ in range(variables.bs_x * variables.bs_y * variables.bs_z)],
              "constantBodyForce":          (0, 0, 0),

              "dirichletOutputFilename":    "out/" + scenario_name + "/dirichlet_output_1",
              "residualNormLogFilename":    "out/" + scenario_name + "/residual_norm_log_1.txt",
              "totalForceLogFilename":      "out/" + scenario_name + "/total_force_log_1.txt",

              "OutputWriter": [
                {
                  "format": "PythonCallback",
                  "callback": callback_function_contraction_1,
                  "outputInterval": 1,
                }
              ],
              "pressure":       { "OutputWriter": [] },
              "dynamic":        { "OutputWriter": [] },
              "LoadIncrements": { "OutputWriter": [] }
            }
          }
        }
      }
    },
    
    "Term2": {
      "Coupling": {
        "timeStepWidth":            variables.dt_3D,
        "logTimeStepWidthAsKey":    "dt_3D",
        "durationLogKey":           "duration_3D",
        "connectedSlotsTerm1To2":   {1:2},  # transfer stress to MuscleContractionSolver gamma
        "connectedSlotsTerm2To1":   None,   # transfer nothing back

        "Term1": { # fibers (FastMonodomainSolver)
          "MultipleInstances": { 
            "ranksAllComputedInstances":    list(range(n_ranks)),
            "nInstances":                   1,

            "instances": [{
              "ranks": [0],

              "StrangSplitting": {
                "timeStepWidth":            variables.dt_splitting,
                "logTimeStepWidthAsKey":    "dt_splitting",
                "durationLogKey":           "duration_splitting",
                "timeStepOutputInterval":   100,
                "connectedSlotsTerm1To2":   None, #{0:0,1:1,2:2,3:3,4:4},
                "connectedSlotsTerm2To1":   None, #{0:0,1:1,2:2,3:3,4:4},

                "Term1": { # reaction term
                  "MultipleInstances": {
                    "nInstances":   variables.fb_x * variables.fb_y,

                    "instances": [{
                      "ranks": [0],

                      "Heun": {
                        "timeStepWidth":            variables.dt_0D,
                        "logTimeStepWidthAsKey":    "dt_0D",
                        "durationLogKey":           "duration_0D",
                        "timeStepOutputInterval":   100,

                        "initialValues":                [],
                        "dirichletBoundaryConditions":  {},
                        "dirichletOutputFilename":      None,
                        "inputMeshIsGlobal":            True,
                        "checkForNanInf":               False,
                        "nAdditionalFieldVariables":    0,
                        "additionalSlotNames":          [],
                        "OutputWriter":                 [],

                        "CellML": {
                          "modelFilename":          variables.input_dir + "hodgkin_huxley-razumova.cellml",
                          "meshName":               "fiber{}_2".format(variables.get_fiber_no(fiber_x, fiber_y)), 
                          "stimulationLogFilename": "out/" + scenario_name + "stimulation_2.log",

                          "statesInitialValues":                        [],
                          "initializeStatesToEquilibrium":              False,
                          "initializeStatesToEquilibriumTimeStepWidth": 1e-4,
                          "optimizationType":                           "vc",
                          "approximateExponentialFunction":             True,
                          "compilerFlags":                              "-fPIC -O3 -march=native -Wno-deprecated_declarations -shared",
                          "maximumNumberOfThreads":                     0,

                          "setSpecificStatesCallEnableBegin":       variables.specific_states_call_enable_begin,
                          "setSpecificStatesCallFrequency":         variables.specific_states_call_frequency,
                          "setSpecificStatesRepeatAfterFirstCall":  0.01,
                          "setSpecificStatesFrequencyJitter":       [0] ,
                          "setSpecificStatesCallInterval":          0,
                          "setSpecificStatesFunction":              None,
                          "additionalArgument":                     None, 

                          "mappings": {
                            ("parameter", 0):               "membrane/i_Stim",
                            ("parameter", 1):               "Razumova/l_hs",
                            ("parameter", 2):               ("constant", "Razumova/rel_velo"),
                            ("connectorSlot", "vm2"):        "membrane/V",
                            ("connectorSlot", "stress2"):    "Razumova/activestress",
                            ("connectorSlot", "alpha2"):     "Razumova/activation",
                            ("connectorSlot", "lambda2"):    "Razumova/l_hs",
                            ("connectorSlot", "ldot2"):      "Razumova/rel_velo"
                          },
                          "parametersInitialValues": [0.0, 1.0, 0.0],
                        },
                      }
                    } for fiber_x in range(variables.fb_x) for fiber_y in range(variables.fb_y)] 
                  }
                },

                "Term2": { # diffusion term
                  "MultipleInstances": {
                    "nInstances": variables.fb_x * variables.fb_y, 

                    "OutputWriter": [
                      {
                        "format":             "Paraview",
                        "outputInterval":     int(1.0 / variables.dt_3D * variables.output_interval),
                        "filename":           "out/" + scenario_name + "/fibers_2",
                        "fileNumbering":      "incremental",
                        "binary":             True,
                        "fixedFormat":        False,
                        "onlyNodalValues":    True,
                        "combineFiles":       True
                      }
                    ],

                    "instances": [{
                      "ranks": [0],

                      "ImplicitEuler": {
                        "timeStepWidth":            variables.dt_1D,
                        "logTimeStepWidthAsKey":    "dt_1D",
                        "durationLogKey":           "duration_1D",
                        "timeStepOutputInterval":   100,

                        "nAdditionalFieldVariables":    4,
                        "additionalSlotNames":          ["stress2", "alpha2", "lambda2", "ldot2"],

                        "solverName":                       "diffusionSolver",
                        "timeStepWidthRelativeTolerance":   1e-10,

                        "dirichletBoundaryConditions":      {},
                        "dirichletOutputFilename":          None,
                        "inputMeshIsGlobal":                True,
                        "checkForNanInf":                   False,
                        "OutputWriter":                     [],

                        "FiniteElementMethod": {
                          "meshName":           "fiber{}_2".format(variables.get_fiber_no(fiber_x, fiber_y)),
                          "inputMeshIsGlobal":  True,
                          "solverName":         "diffusionSolver",
                          "prefactor":          variables.diffusion_prefactor,
                          "slotName":           "vm2"
                        }
                      }
                    } for fiber_x in range(variables.fb_x) for fiber_y in range(variables.fb_y)]
                  }
                }
              }
            }]
          },

          "fiberDistributionFile":                              variables.fiber_distribution_file,
          "firingTimesFile":                                    variables.firing_times_file,
          "valueForStimulatedPoint":                            20.0,
          "onlyComputeIfHasBeenStimulated":                     True,
          "disableComputationWhenStatesAreCloseToEquilibrium":  True,
          "neuromuscularJunctionRelativeSize":                  0.0,################################change for no randomness
          "generateGPUSource":                                  True,
          "useSinglePrecision":                                 False
        },

        "Term2": { # solid mechanics (MuscleContractionSolver)
          "MuscleContractionSolver": {
            "Pmax":                         variables.pmax,
            "slotNames":                    ["lambdaContraction2", "ldotContraction2", "gammaContraction2", "TContraction2"],
            #"slotNames":                    ["lambda", "ldot", "gamma", "T"],
            "dynamic":                      True,

            "numberTimeSteps":              1,
            "timeStepOutputInterval":       100,
            "lambdaDotScalingFactor":       1,
            "enableForceLengthRelation":    True,
            "mapGeometryToMeshes":          ["fiber{}_2".format(variables.get_fiber_no(fiber_x, fiber_y)) for fiber_x in range(fb_x) for fiber_y in range(fb_y)],

            "OutputWriter": [
              {
                "format":             "Paraview",
                "outputInterval":     int(1.0 / variables.dt_3D * variables.output_interval),
                "filename":           "out/" + scenario_name + "/mechanics_2",
                "fileNumbering":      "incremental",
                "binary":             True,
                "fixedFormat":        False,
                "onlyNodalValues":    True,
                "combineFiles":       True
              }
            ],

            "DynamicHyperelasticitySolver": {
              "durationLogKey":         "duration_3D",
              "logTimeStepWidthAsKey":  "dt_3D",
              "numberTimeSteps":        1,
              "materialParameters":     variables.material_parameters,
              "density":                variables.rho,
              "timeStepOutputInterval": 1,

              "meshName":                   "3Dmesh_quadratic_2",
              "inputMeshIsGlobal":          True,
              "fiberMeshNames":             "fiber{}_2".format(variables.get_fiber_no(fiber_x, fiber_y)),
              "solverName":                 "mechanicsSolver",
              "displacementsScalingFactor":  1.0,
              "useAnalyticJacobian":        True,
              "useNumericJacobian":         False,
              "dumpDenseMatlabVariables":   False,
              "loadFactorGiveUpThreshold":  1,
              "loadFactors":                [],
              "scaleInitialGuess":          False,
              "extrapolateInitialGuess":    True,
              "nNonlinearSolveCalls":       1,

              "dirichletBoundaryConditions":                            elasticity_dirichlet_bc_2, #variables.dirichlet_bc,
              "neumannBoundaryConditions":                              elasticity_neumann_bc_2, #elasticity_neumann_bc, #variables.neumann_bc,
              "updateDirichletBoundaryConditionsFunction":              None,
              "updateDirichletBoundaryConditionsFunctionCallInterval":  1,
              "divideNeumannBoundaryConditionValuesByTotalArea":        True,

              "initialValuesDisplacements": [[0, 0, 0] for _ in range(variables.bs_x * variables.bs_y * variables.bs_z)],
              "initialValuesVelocities":    [[0, 0, 0] for _ in range(variables.bs_x * variables.bs_y * variables.bs_z)],
              "constantBodyForce":          (0, 0, 0),

              "dirichletOutputFilename":    "out/" + scenario_name + "/dirichlet_output_2",
              "residualNormLogFilename":    "out/" + scenario_name + "/residual_norm_log_2.txt",
              "totalForceLogFilename":      "out/" + scenario_name + "/total_force_log_2.txt",

              "OutputWriter": [
                {
                  "format": "PythonCallback",
                  "callback": callback_function_contraction_2,
                  "outputInterval": 1,
                }
              ],
              "pressure":       { "OutputWriter": [] },
              "dynamic":        { "OutputWriter": [] },
              "LoadIncrements": { "OutputWriter": [] }
            }
          }
        }
      }
    }
  },
}
