import numpy as np
import sys, os
import time

script_path = os.path.dirname(os.path.abspath(__file__))
var_path = os.path.join(script_path, "variables")
sys.path.insert(0, var_path)

import variables

n_ranks = (int)(sys.argv[-1])

# parameters
force = variables.force
scenario_name = variables.scenario_name  

if len(sys.argv) > 3:                                                                           
  scenario_name = sys.argv[0]
  force = float(sys.argv[1])
  print("scenario_name: {}".format(scenario_name))
  print("force: {}".format(force))

if len(sys.argv) > 4:
  individuality_parameter = sys.argv[2] 
else:
  individuality_parameter = str(time.time())

tendon_length_0 = variables.physical_offset_2[2] - variables.physical_extent[2]
tendon_length_t = tendon_length_0
tendon_start_t = variables.physical_extent[2]
tendon_end_t = variables.physical_offset_2[2]

force_data_muscle_1 = []
force_data_muscle_2 = []

meshes = { # create 3D mechanics mesh
    "3Dmesh_quadratic_1": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [variables.el_x, variables.el_y, variables.el_z],               # number of quadratic elements in x, y and z direction
      "physicalExtent":             variables.physical_extent,            # physical size of the box
      "physicalOffset":             variables.physical_offset_1,          # offset/translation where the whole mesh begins
    },
    "3Dmesh_quadratic_2": { 
      "inputMeshIsGlobal":          True,                       # boundary conditions are specified in global numberings, whereas the mesh is given in local numberings
      "nElements":                  [variables.el_x, variables.el_y, variables.el_z],               # number of quadratic elements in x, y and z direction
      "physicalExtent":             variables.physical_extent,            # physical size of the box
      "physicalOffset":             variables.physical_offset_2,                  # offset/translation where the whole mesh begins
    }
}

for fiber_x in range(variables.fb_x):
    for fiber_y in range(variables.fb_y):
        fiber_no = variables.get_fiber_no(fiber_x, fiber_y)
        x = variables.physical_extent[0] * fiber_x / (variables.fb_x - 1)
        y = variables.physical_extent[1] * fiber_y / (variables.fb_y - 1)
        nodePositions_1 = [[x, y, variables.physical_extent[2] * i / (variables.fb_points - 1)] for i in range(variables.fb_points)]
        nodePositions_2 = [[x, y, variables.physical_offset_2[2]+variables.physical_extent[2] * i / (variables.fb_points - 1)] for i in range(variables.fb_points)]
        meshName_1 = "fiber{}_1".format(fiber_no)
        meshes[meshName_1] = { # create fiber meshes
            "nElements":            [variables.fb_points - 1],
            "nodePositions":        nodePositions_1,
            "inputMeshIsGlobal":    True,
            "nRanks":               n_ranks
        }
        meshName_2 = "fiber{}_2".format(fiber_no)
        meshes[meshName_2] = { # create fiber meshes
            "nElements":            [variables.fb_points - 1],
            "nodePositions":        nodePositions_2,
            "inputMeshIsGlobal":    True,
            "nRanks":               n_ranks
        }

# set Dirichlet BC, fix bottom
elasticity_dirichlet_bc_1 = {}
elasticity_dirichlet_bc_2 = {}

k1 = 0
k2=variables.bs_z-1
# fix z value on the whole x-y-plane
for j in range(variables.bs_y):
  for i in range(variables.bs_x):
    elasticity_dirichlet_bc_1[k1*variables.bs_x*variables.bs_y + j*variables.bs_x + i] = [None,None,0.0,None,None,None]
    elasticity_dirichlet_bc_2[k2*variables.bs_x*variables.bs_y + j*variables.bs_x + i] = [None,None,0.0,None,None,None]

# fix left edge 
for j in range(variables.bs_y):
  elasticity_dirichlet_bc_1[k1*variables.bs_x*variables.bs_y + j*variables.bs_x + 0][0] = 0.0
  elasticity_dirichlet_bc_2[k2*variables.bs_x*variables.bs_y + j*variables.bs_x + 0][0] = 0.0
  
# fix front edge 
for i in range(variables.bs_x):
  elasticity_dirichlet_bc_1[k1*variables.bs_x*variables.bs_y + 0*variables.bs_x + i][1] = 0.0
  elasticity_dirichlet_bc_2[k2*variables.bs_x*variables.bs_y + 0*variables.bs_x + i][1] = 0.0

traction_vector = [0, 0, 0]

elasticity_neumann_bc_1 = [{"element": (variables.el_z-1)*variables.el_x*variables.el_y + i*variables.el_y + j, "constantVector": traction_vector, "face": "2+", "isInReferenceConfiguration": True} for i in range(variables.el_x) for j in range(variables.el_y)]
elasticity_neumann_bc_2 = [{"element": j*variables.el_x + i, "constantVector": traction_vector, "face": "2-", "isInReferenceConfiguration": True} for i in range(variables.el_x) for j in range(variables.el_y)]



# callback for result
def handle_result_prestretch(result):
  data = result[0]

  number_of_nodes = variables.bs_x * variables.bs_y
  average_z_start = 0
  average_z_end = 0

  z_data = data["data"][0]["components"][2]["values"]

  for i in range(number_of_nodes):
    average_z_start += z_data[i]
    average_z_end += z_data[number_of_nodes*(variables.bs_z -1) + i]

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
  global elasticity_neumann_bc_2, tendon_start_t, force_data_muscle_1
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
  tendon_start_t = average_z_end

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

  force_data_muscle_1 = raw_data[0]["data"][6]["components"][2]["values"]

      
def callback_function_contraction_2(raw_data):
  global elasticity_neumann_bc_1, tendon_end_t, force_data_muscle_2
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
  tendon_end_t = average_z_start

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

  force_data_muscle_2 = raw_data[0]["data"][6]["components"][2]["values"]    


def updateNeumannContraction_1(t):
  if not variables.tendon_spring_simulation:
    for i in range(variables.el_x):
      for j in range(variables.el_y):
        force = force_data_muscle_2[i*variables.el_y + j]
        traction_vector = [0,0,variables.tendon_damping_constant*force]
        elasticity_neumann_bc_1[i*variables.el_y+j]["constantVector"] = traction_vector

  if variables.tendon_spring_simulation:
    tendon_length_t = tendon_end_t - tendon_start_t
    force = variables.tendon_spring_constant * (tendon_length_t-tendon_length_0)
    traction_vector = [0,0,force]
    for i in range(variables.el_x):
      for j in range(variables.el_y):
        elasticity_neumann_bc_1[i*variables.el_y+j]["constantVector"] = traction_vector

  config = {
    "InputMeshIsGlobal":  True,
    "divideNeumannBoundaryConditionValuesByTotalArea": variables.tendon_spring_simulation,
    "neumannBoundaryConditions":  elasticity_neumann_bc_1
  }
  return config



def updateNeumannContraction_2(t):
  if not variables.tendon_spring_simulation:
    for i in range(variables.el_x):
      for j in range(variables.el_y):
        force = force_data_muscle_1[i*variables.el_y + j]
        traction_vector = [0,0,variables.tendon_damping_constant*force]
        elasticity_neumann_bc_2[i*variables.el_y+j]["constantVector"] = traction_vector

  if variables.tendon_spring_simulation:
    tendon_length_t = tendon_end_t - tendon_start_t
    force = variables.tendon_spring_constant * (tendon_length_t-tendon_length_0)
    traction_vector = [0,0,force]
    for i in range(variables.el_x):
      for j in range(variables.el_y):
        elasticity_neumann_bc_2[i*variables.el_y+j]["constantVector"] = traction_vector

  config = {
    "InputMeshIsGlobal":  True,
    "divideNeumannBoundaryConditionValuesByTotalArea": variables.tendon_spring_simulation,
    "neumannBoundaryConditions":  elasticity_neumann_bc_2
  }
  return config



config = {
  "scenarioName":                 scenario_name,                # scenario name to identify the simulation runs in the log file
  "logFormat":                    "csv",                        # "csv" or "json", format of the lines in the log file, csv gives smaller files
  "solverStructureDiagramFile":   "solver_structure.txt",       # output file of a diagram that shows data connection between solvers
  "mappingsBetweenMeshesLogFile": "mappings_between_meshes_log.txt",    # log file for mappings 

  "Meshes": meshes,

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

                          "setSpecificStatesCallEnableBegin":       variables.specific_states_call_enable_begin_1,
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
            "mapGeometryToMeshes":          ["fiber{}_1".format(variables.get_fiber_no(fiber_x, fiber_y)) for fiber_x in range(variables.fb_x) for fiber_y in range(variables.fb_y)],

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

              "dirichletBoundaryConditions":                            elasticity_dirichlet_bc_1,
              "neumannBoundaryConditions":                              elasticity_neumann_bc_1,
              "updateDirichletBoundaryConditionsFunction":              None,
              "updateNeumannBoundaryConditionsFunction":                updateNeumannContraction_1,
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

                          "setSpecificStatesCallEnableBegin":       variables.specific_states_call_enable_begin_2,
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
            "mapGeometryToMeshes":          ["fiber{}_2".format(variables.get_fiber_no(fiber_x, fiber_y)) for fiber_x in range(variables.fb_x) for fiber_y in range(variables.fb_y)],

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

              "dirichletBoundaryConditions":                            elasticity_dirichlet_bc_2,
              "neumannBoundaryConditions":                              elasticity_neumann_bc_2,
              "updateDirichletBoundaryConditionsFunction":              None,
              "updateNeumannBoundaryConditionsFunction":                updateNeumannContraction_2,
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
