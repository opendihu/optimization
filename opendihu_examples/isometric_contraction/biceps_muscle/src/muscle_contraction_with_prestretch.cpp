#include <Python.h>
#include <iostream>
#include <cstdlib>

#include <iostream>
#include "easylogging++.h"

#include "opendihu.h"


int main(int argc, char *argv[]) {
  // Solves nonlinear hyperelasticity (Mooney-Rivlin) using the built-in solver

  // initialize everything, handle arguments and parse settings from input file
  DihuContext settings(argc, argv);

  // define problem
  Control::Coupling<

    Control::Coupling< // Couple fibers (FastMonodomain) with solid mechanics
                        // (MuscleContraction)
        FastMonodomainSolver<Control::MultipleInstances< // subdomains in xy-plane
            OperatorSplitting::Strang<
                Control::MultipleInstances< // fiber reaction term
                    TimeSteppingScheme::Heun<CellmlAdapter<
                        57, 71, // nStates, nAlgebraics
                        FunctionSpace::FunctionSpace<
                            Mesh::StructuredDeformableOfDimension<1>,
                            BasisFunction::LagrangeOfOrder<1>>>>>,
                Control::MultipleInstances<            // fiber diffusion
                    TimeSteppingScheme::CrankNicolson< // note that implicit euler
                                                        // gives lower error in
                                                        // this case than crank
                                                        // nicolson
                        SpatialDiscretization::FiniteElementMethod<
                            Mesh::StructuredDeformableOfDimension<1>,
                            BasisFunction::LagrangeOfOrder<1>,
                            Quadrature::Gauss<2>,
                            Equation::Dynamic::IsotropicDiffusion>>>>>>,
        MuscleContractionSolver<>>,

    Control::Coupling< // Couple fibers (FastMonodomain) with solid mechanics
                     // (MuscleContraction)
      FastMonodomainSolver<Control::MultipleInstances< // subdomains in xy-plane
          OperatorSplitting::Strang<
              Control::MultipleInstances< // fiber reaction term
                  TimeSteppingScheme::Heun<CellmlAdapter<
                      57, 71, // nStates, nAlgebraics
                      FunctionSpace::FunctionSpace<
                          Mesh::StructuredDeformableOfDimension<1>,
                          BasisFunction::LagrangeOfOrder<1>>>>>,
              Control::MultipleInstances<            // fiber diffusion
                  TimeSteppingScheme::CrankNicolson< // note that implicit euler
                                                     // gives lower error in
                                                     // this case than crank
                                                     // nicolson
                      SpatialDiscretization::FiniteElementMethod<
                          Mesh::StructuredDeformableOfDimension<1>,
                          BasisFunction::LagrangeOfOrder<1>,
                          Quadrature::Gauss<2>,
                          Equation::Dynamic::IsotropicDiffusion>>>>>>,
      MuscleContractionSolver<>>
    > problem(settings);

  // run problem
  problem.run();

  return EXIT_SUCCESS;
}
