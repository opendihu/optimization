#include <Python.h>
#include <iostream>
#include <cstdlib>

#include <iostream>
#include "easylogging++.h"

#include "opendihu.h"

// define material
struct Material : Equation::SolidMechanics::HyperelasticityBase {
  static constexpr bool isIncompressible =
      true; //< if the formulation is incompressible, then,
            // strainEnergyDensityFunctionVolumetric will not be considered
  static constexpr bool usesFiberDirection =
      false; //< if the decoupled form uses the 4th or 5th invariants, Ibar4,
             // Ibar2, this means it is an anisotropic material
  static constexpr bool usesActiveStress =
      false; //< if the value of an active stress term will be added to the
             // stress

  // material parameters
  static constexpr auto c1 = PARAM(0); //< material parameter
  static constexpr auto c2 = PARAM(1); //< material parameter

  static constexpr int nMaterialParameters =
      2; //< number of material parameters

  //! the isochoric part of the decoupled strain energy function,
  //! Ψ_iso(Ibar1,Ibar2,Ibar4,Ibar5), in terms of the reduced invariants
  static const auto constexpr strainEnergyDensityFunctionIsochoric =
      c1 * (Ibar1 - INT(3)) + c2 * (Ibar2 - INT(3));

  //! the volumetric part of the decoupled strain energy function, Ψ_vol(J),
  //! only used for compressible formulation (isIncompressible == false)
  static const auto constexpr strainEnergyDensityFunctionVolumetric = INT(0);

  //! coupled form of the strain energy function, Ψ(I1,I2,I3), as alternative to
  //! the two decoupled functions
  static const auto constexpr strainEnergyDensityFunctionCoupled = INT(0);

  //! another coupled form of the strain energy function, Ψ(C), dependent on
  //! right Cauchy Green tensor, C. it must only depend on variables C11, C12,
  //! C13, C22, C23, C33.
  static const auto constexpr strainEnergyDensityFunctionCoupledDependentOnC =
      INT(0);
};

int main(int argc, char *argv[]) {

  DihuContext settings(argc, argv);

  // define problem
  Control::Coupling<
    Control::Coupling< 
        FastMonodomainSolver<Control::MultipleInstances< // electrophysiology solver for muscle 1
            OperatorSplitting::Strang<
                Control::MultipleInstances< 
                    TimeSteppingScheme::Heun<CellmlAdapter<
                        9, 19, // nStates, nAlgebraics
                        FunctionSpace::FunctionSpace<Mesh::StructuredDeformableOfDimension<1>,
                        BasisFunction::LagrangeOfOrder<1>>>>>,
                Control::MultipleInstances<            
                    TimeSteppingScheme::ImplicitEuler< 
                        SpatialDiscretization::FiniteElementMethod<
                            Mesh::StructuredDeformableOfDimension<1>,
                            BasisFunction::LagrangeOfOrder<1>,
                            Quadrature::Gauss<2>,
                            Equation::Dynamic::IsotropicDiffusion>>>>>>,

        FastMonodomainSolver<Control::MultipleInstances< // electrophysiology solver for muscle 2
            OperatorSplitting::Strang<
                Control::MultipleInstances< // fiber reaction term
                    TimeSteppingScheme::Heun<CellmlAdapter<
                        9, 19, // nStates, nAlgebraics
                        FunctionSpace::FunctionSpace<
                            Mesh::StructuredDeformableOfDimension<1>,
                            BasisFunction::LagrangeOfOrder<1>>>>>,
                Control::MultipleInstances<            
                    TimeSteppingScheme::ImplicitEuler< 
                        SpatialDiscretization::FiniteElementMethod<
                            Mesh::StructuredDeformableOfDimension<1>,
                            BasisFunction::LagrangeOfOrder<1>,
                            Quadrature::Gauss<2>,
                            Equation::Dynamic::IsotropicDiffusion>>>>>>>,

    Control::Coupling< 
        MuscleContractionSolver< // 3D solid mechanics equation for muscle 1
            Mesh::StructuredDeformableOfDimension<3>,
            Equation::SolidMechanics::
            TransverselyIsotropicMooneyRivlinIncompressibleActive3D>,
        MuscleContractionSolver< // 3D solid mechanics equation for muscle 2
            Mesh::StructuredDeformableOfDimension<3>,
            Equation::SolidMechanics::
            TransverselyIsotropicMooneyRivlinIncompressibleActive3D>>>
    problem(settings);

  // run problem
  problem.run();

  return EXIT_SUCCESS;
}
