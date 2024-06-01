#include <math.h>
#include <vc_or_std_simd.h>  // this includes <Vc/Vc> or a Vc-emulating wrapper of <experimental/simd> if available
#include <iostream> 
/*
   There are a total of 19 entries in the algebraic variable array.
   There are a total of 9 entries in each of the rate and state variable arrays.
   There are a total of 39 entries in the constant variable array.
 */
/*
 * VOI is time in component environment (millisecond).
 * STATES[0] is V in component membrane (millivolt).
 * CONSTANTS[0] is E_R in component membrane (millivolt).
 * CONSTANTS[1] is Cm in component membrane (microF_per_cm2).
 * ALGEBRAIC[10] is i_Na in component sodium_channel (microA_per_cm2).
 * ALGEBRAIC[15] is i_K in component potassium_channel (microA_per_cm2).
 * ALGEBRAIC[17] is i_L in component leakage_current (microA_per_cm2).
 * ALGEBRAIC[0] is i_Stim in component membrane (microA_per_cm2).
 * CONSTANTS[2] is g_Na in component sodium_channel (milliS_per_cm2).
 * CONSTANTS[28] is E_Na in component sodium_channel (millivolt).
 * STATES[1] is m in component sodium_channel_m_gate (dimensionless).
 * STATES[2] is h in component sodium_channel_h_gate (dimensionless).
 * ALGEBRAIC[1] is alpha_m in component sodium_channel_m_gate (per_millisecond).
 * ALGEBRAIC[11] is beta_m in component sodium_channel_m_gate (per_millisecond).
 * ALGEBRAIC[2] is alpha_h in component sodium_channel_h_gate (per_millisecond).
 * ALGEBRAIC[12] is beta_h in component sodium_channel_h_gate (per_millisecond).
 * CONSTANTS[3] is g_K in component potassium_channel (milliS_per_cm2).
 * CONSTANTS[29] is E_K in component potassium_channel (millivolt).
 * STATES[3] is n in component potassium_channel_n_gate (dimensionless).
 * ALGEBRAIC[3] is alpha_n in component potassium_channel_n_gate (per_millisecond).
 * ALGEBRAIC[13] is beta_n in component potassium_channel_n_gate (per_millisecond).
 * CONSTANTS[4] is g_L in component leakage_current (milliS_per_cm2).
 * CONSTANTS[30] is E_L in component leakage_current (millivolt).
 * STATES[4] is D in component Razumova (dimensionless).
 * STATES[5] is A_1 in component Razumova (dimensionless).
 * STATES[6] is A_2 in component Razumova (dimensionless).
 * STATES[7] is x_1 in component Razumova (micrometer).
 * STATES[8] is x_2 in component Razumova (micrometer).
 * CONSTANTS[5] is x_0 in component Razumova (micrometer).
 * CONSTANTS[6] is R_T in component Razumova (dimensionless).
 * ALGEBRAIC[18] is R_off in component Razumova (dimensionless).
 * CONSTANTS[7] is A_2_0 in component Razumova (dimensionless).
 * ALGEBRAIC[4] is r in component Razumova (dimensionless).
 * CONSTANTS[8] is r_0 in component Razumova (dimensionless).
 * CONSTANTS[9] is l_hs in component Razumova (micrometer).
 * CONSTANTS[10] is rel_velo in component Razumova (micrometer_per_millisecond).
 * CONSTANTS[31] is velo_scaled in component Razumova (micrometer_per_millisecond).
 * CONSTANTS[11] is velo_max in component Razumova (dimensionless).
 * ALGEBRAIC[14] is k_on in component Razumova (per_millisecond).
 * ALGEBRAIC[16] is k_off in component Razumova (per_millisecond).
 * CONSTANTS[32] is f in component Razumova (per_millisecond).
 * CONSTANTS[33] is f_prime in component Razumova (per_millisecond).
 * CONSTANTS[34] is h in component Razumova (per_millisecond).
 * CONSTANTS[35] is h_prime in component Razumova (per_millisecond).
 * CONSTANTS[36] is g in component Razumova (per_millisecond).
 * CONSTANTS[37] is g_prime in component Razumova (per_millisecond).
 * CONSTANTS[12] is k_0_on in component Razumova (per_millisecond).
 * CONSTANTS[13] is k_0_off in component Razumova (per_millisecond).
 * CONSTANTS[14] is k_Ca_on in component Razumova (per_millisecond).
 * CONSTANTS[15] is k_Ca_off in component Razumova (per_millisecond).
 * CONSTANTS[16] is g_0 in component Razumova (per_millisecond).
 * CONSTANTS[17] is f_0 in component Razumova (per_millisecond).
 * CONSTANTS[18] is h_0 in component Razumova (per_millisecond).
 * CONSTANTS[19] is f_prime0 in component Razumova (per_millisecond).
 * CONSTANTS[20] is h_prime0 in component Razumova (per_millisecond).
 * CONSTANTS[21] is g_prime0 in component Razumova (per_millisecond).
 * ALGEBRAIC[5] is sigma in component Razumova (dimensionless).
 * ALGEBRAIC[6] is lambda_A1 in component Razumova (dimensionless).
 * ALGEBRAIC[7] is lambda_A2 in component Razumova (dimensionless).
 * CONSTANTS[22] is Ca_50 in component Razumova (dimensionless).
 * CONSTANTS[23] is nu in component Razumova (dimensionless).
 * CONSTANTS[24] is E_ATP in component Razumova (joule).
 * CONSTANTS[25] is kappa in component Razumova (joule_per_kelvin).
 * CONSTANTS[26] is T in component Razumova (kelvin).
 * ALGEBRAIC[8] is activestress in component Razumova (dimensionless).
 * ALGEBRAIC[9] is activation in component Razumova (dimensionless).
 * CONSTANTS[38] is f_l in component Razumova (dimensionless).
 * CONSTANTS[27] is A_2_max in component Razumova (dimensionless).
 * RATES[0] is d/dt V in component membrane (millivolt).
 * RATES[1] is d/dt m in component sodium_channel_m_gate (dimensionless).
 * RATES[2] is d/dt h in component sodium_channel_h_gate (dimensionless).
 * RATES[3] is d/dt n in component potassium_channel_n_gate (dimensionless).
 * RATES[4] is d/dt D in component Razumova (dimensionless).
 * RATES[5] is d/dt A_1 in component Razumova (dimensionless).
 * RATES[6] is d/dt A_2 in component Razumova (dimensionless).
 * RATES[7] is d/dt x_1 in component Razumova (micrometer).
 * RATES[8] is d/dt x_2 in component Razumova (micrometer).
 */

using Vc::double_v; 

/* This file was created by opendihu at 2024/6/1 22:36:37.
 * It is designed for the FastMonodomainSolver.
  */

// helper functions
Vc::double_v exponential(Vc::double_v x);
Vc::double_v pow2(Vc::double_v x);
Vc::double_v pow3(Vc::double_v x);
Vc::double_v pow4(Vc::double_v x);

Vc::double_v exponential(Vc::double_v x)
{
  //return Vc::exp(x);
  // it was determined the x is always in the range [-12,+12] for the Hodgkin-Huxley model

  // exp(x) = lim n→∞ (1 + x/n)^n, we set n=1024
  x = 1.0 + x / 1024.;
  for (int i = 0; i < 10; i++)
  {
    x *= x;
  }
  return x;

  // relative error of this implementation:
  // x    rel error
  // 0    0
  // 1    0.00048784455634225593
  // 3    0.0043763626896140342
  // 5    0.012093715791500804
  // 9    0.038557535762274039
  // 12   0.067389808619653505
}

Vc::double_v pow2(Vc::double_v x)
{
  return x*x;
}
Vc::double_v pow3(Vc::double_v x)
{
  return x*(pow2(x));
}

Vc::double_v pow4(Vc::double_v x)
{
  return pow2(pow2(x));
}

// set initial values for all states
#ifdef __cplusplus
extern "C"
#endif

void initializeStates(Vc::double_v states[]) 
{
  states[0] = -75;
  states[1] = 0.05;
  states[2] = 0.6;
  states[3] = 0.325;
  states[4] = 3.8e-14;
  states[5] = 1e-14;
  states[6] = 3.4e-13;
  states[7] = 1e-16;
  states[8] = 0.008;
}

// compute one Heun step
#ifdef __cplusplus
extern "C"
#endif

void compute0DInstance(Vc::double_v states[], std::vector<Vc::double_v> &parameters, double currentTime, double timeStepWidth, bool stimulate,
                       bool storeAlgebraicsForTransfer, std::vector<Vc::double_v> &algebraicsForTransfer, const std::vector<int> &algebraicsForTransferIndices, double valueForStimulatedPoint) 
{
  // assert that Vc::double_v::size() is the same as in opendihu, otherwise there will be problems
  if (Vc::double_v::size() != 4)
  {
    std::cout << "Fatal error in compiled library of source file \"src/hodgkin_huxley-razumova_fast_monodomain.c\", size of SIMD register in compiled code (" << Vc::double_v::size() << ") does not match opendihu code (4)." << std::endl;
    std::cout << "Delete library such that it will be regenerated with the correct compile options!" << std::endl;
    exit(1);
  }

  // define constants
  const double constant0 = -75;
  const double constant1 = 1;
  const double constant2 = 120;
  const double constant3 = 36;
  const double constant4 = 0.3;
  const double constant5 = 8e-3;
  const double constant6 = 1;
  const double constant7 = 3.4e-13;
  const double constant8 = 10;
  const double constant9 = 1;
  const double constant10 = 0;
  const double constant11 = 7.815e-5;
  const double constant12 = 0;
  const double constant13 = 100e-3;
  const double constant14 = 120e-3;
  const double constant15 = 50e-3;
  const double constant16 = 4e-3;
  const double constant17 = 50e-3;
  const double constant18 = 8e-3;
  const double constant19 = 400e-3;
  const double constant20 = 6e-3;
  const double constant21 = 3.5400e-13;
  const double constant22 = 1;
  const double constant23 = 3.2;
  const double constant24 = 9.1362e-20;
  const double constant25 = 1.38e-23;
  const double constant26 = 310;
  const double constant27 = 0.015;
  const double constant28 = constant0+115.000;
  const double constant29 = constant0 - 12.0000;
  const double constant30 = constant0+10.6130;
  const double constant31 =  constant10*constant11;
  const double constant32 = constant17;
  const double constant33 = constant19;
  const double constant34 = constant18;
  const double constant35 = constant20;
  const double constant36 = constant16;
  const double constant37 = 0.00000;
  const double constant38 = (constant9<0.635000 ? 0.00000 : constant9<0.835000 ?  4.20000*(constant9 - 0.635000) : constant9<1.00000 ? 0.840000+ 0.969700*(constant9 - 0.835000) : constant9<1.12500 ? 1.00000 : constant9<1.82500 ? 1.00000 -  1.42860*(constant9 - 1.12500) : 0.00000);

  // compute new rates, rhs(y_n)
  const double_v rate5 = ( constant32*states[4]+ constant35*states[6]) -  (constant33+constant34)*states[5];
  const double_v rate6 = ( constant34*states[5] -  (constant35+constant36)*states[6])+ constant37*states[4];
  const double_v rate7 = ( (( - constant32*states[4])/states[5])*states[7] -  (( constant35*states[6])/states[5])*states[7])+constant31;
  const double_v rate8 =  (( - constant34*states[5])/states[6])*(states[8] - constant5)+constant31;
  const double_v algebraic1 = ( - 0.100000*(states[0]+50.0000))/(exponential(- (states[0]+50.0000)/10.0000) - 1.00000);
  const double_v algebraic11 =  4.00000*exponential(- (states[0]+75.0000)/18.0000);
  const double_v rate1 =  algebraic1*(1.00000 - states[1]) -  algebraic11*states[1];
  const double_v algebraic2 =  0.0700000*exponential(- (states[0]+75.0000)/20.0000);
  const double_v algebraic12 = 1.00000/(exponential(- (states[0]+45.0000)/10.0000)+1.00000);
  const double_v rate2 =  algebraic2*(1.00000 - states[2]) -  algebraic12*states[2];
  const double_v algebraic3 = ( - 0.0100000*(states[0]+65.0000))/(exponential(- (states[0]+65.0000)/10.0000) - 1.00000);
  const double_v algebraic13 =  0.125000*exponential((states[0]+75.0000)/80.0000);
  const double_v rate3 =  algebraic3*(1.00000 - states[3]) -  algebraic13*states[3];
  const double_v algebraic10 =  constant2*pow3(states[1])*states[2]*(states[0] - constant28);
  const double_v algebraic15 =  constant3*pow4(states[3])*(states[0] - constant29);
  const double_v algebraic17 =  constant4*(states[0] - constant30);
    // (not assigning to a parameter) algebraics = (VOI>=10.0000&&VOI<=10.5000 Vc::iif(VOI>=10.0000&&VOI<=10.5000 , (Vc::double_v(Vc::One)*( 20.0000 )), VOI>=50.0000&&VOI<=50.5000 Vc::iif( VOI>=50.0000&&VOI<=50.5000 , (Vc::double_v(Vc::One)*( 20.0000 )), VOI>=90.0000&&VOI<=90.5000 Vc::iif( VOI>=90.0000&&VOI<=90.5000 , (Vc::double_v(Vc::One)*( 20.0000 )), VOI>=130.000&&VOI<=130.500 Vc::iif( VOI>=130.000&&VOI<=130.500 , (Vc::double_v(Vc::One)*( 20.0000 )), Vc::double_v(Vc::Zero))))));
  const double_v rate0 = - (- parameters[0]+algebraic10+algebraic15+algebraic17)/constant1;
  const double_v algebraic18 = ((constant6 - states[5]) - states[6]) - states[4];
  const double_v algebraic4 = (Vc::iif(states[0]>- 60.0000 , Vc::double_v(Vc::One), Vc::double_v(Vc::Zero)));
  const double_v algebraic14 = constant12+( (constant14 - constant12)*algebraic4*constant8)/( algebraic4*constant8+constant22);
  const double_v algebraic16 = constant13+( (constant15 - constant13)*algebraic4*constant8)/( algebraic4*constant8+constant22);
  const double_v rate4 = ( algebraic14*algebraic18+ constant33*states[5]+ constant36*states[6]) -  (algebraic16+constant32+constant37)*states[4];
  const double_v algebraic5 = (Vc::iif(states[8]>constant5 , Vc::double_v(Vc::One),Vc::iif( states[8]<(Vc::double_v(Vc::One)*constant5) , (Vc::double_v(Vc::One)*( 8.00000 )), Vc::double_v(Vc::Zero))));
  const double_v algebraic6 = states[5]/constant6;
  const double_v algebraic7 = states[6]/constant6;
  const double_v algebraic8 =  ((( states[6]*states[8]+ states[5]*states[7]) -  constant7*constant5)/( constant27*constant5))*constant38;
  const double_v algebraic9 = (states[6] - constant7)/constant27;

  // algebraic step
  // compute y* = y_n + dt*rhs(y_n), y_n = state, rhs(y_n) = rate, y* = algebraicState
  double_v algebraicState0 = states[0] + timeStepWidth*rate0;
  const double_v algebraicState1 = states[1] + timeStepWidth*rate1;
  const double_v algebraicState2 = states[2] + timeStepWidth*rate2;
  const double_v algebraicState3 = states[3] + timeStepWidth*rate3;
  const double_v algebraicState4 = states[4] + timeStepWidth*rate4;
  const double_v algebraicState5 = states[5] + timeStepWidth*rate5;
  const double_v algebraicState6 = states[6] + timeStepWidth*rate6;
  const double_v algebraicState7 = states[7] + timeStepWidth*rate7;
  const double_v algebraicState8 = states[8] + timeStepWidth*rate8;



  // if stimulation, set value of Vm (state0)
  if (stimulate)
  {
    for (int i = 0; i < std::min(3,(int)Vc::double_v::size()); i++)
    {
      algebraicState0[i] = valueForStimulatedPoint;
    }
  }
  // compute new rates, rhs(y*)
  const double_v algebraicRate5 = ( constant32*algebraicState4+ constant35*algebraicState6) -  (constant33+constant34)*algebraicState5;
  const double_v algebraicRate6 = ( constant34*algebraicState5 -  (constant35+constant36)*algebraicState6)+ constant37*algebraicState4;
  const double_v algebraicRate7 = ( (( - constant32*algebraicState4)/algebraicState5)*algebraicState7 -  (( constant35*algebraicState6)/algebraicState5)*algebraicState7)+constant31;
  const double_v algebraicRate8 =  (( - constant34*algebraicState5)/algebraicState6)*(algebraicState8 - constant5)+constant31;
  const double_v algebraicAlgebraic1 = ( - 0.100000*(algebraicState0+50.0000))/(exponential(- (algebraicState0+50.0000)/10.0000) - 1.00000);
  const double_v algebraicAlgebraic11 =  4.00000*exponential(- (algebraicState0+75.0000)/18.0000);
  const double_v algebraicRate1 =  algebraicAlgebraic1*(1.00000 - algebraicState1) -  algebraicAlgebraic11*algebraicState1;
  const double_v algebraicAlgebraic2 =  0.0700000*exponential(- (algebraicState0+75.0000)/20.0000);
  const double_v algebraicAlgebraic12 = 1.00000/(exponential(- (algebraicState0+45.0000)/10.0000)+1.00000);
  const double_v algebraicRate2 =  algebraicAlgebraic2*(1.00000 - algebraicState2) -  algebraicAlgebraic12*algebraicState2;
  const double_v algebraicAlgebraic3 = ( - 0.0100000*(algebraicState0+65.0000))/(exponential(- (algebraicState0+65.0000)/10.0000) - 1.00000);
  const double_v algebraicAlgebraic13 =  0.125000*exponential((algebraicState0+75.0000)/80.0000);
  const double_v algebraicRate3 =  algebraicAlgebraic3*(1.00000 - algebraicState3) -  algebraicAlgebraic13*algebraicState3;
  const double_v algebraicAlgebraic10 =  constant2*pow3(algebraicState1)*algebraicState2*(algebraicState0 - constant28);
  const double_v algebraicAlgebraic15 =  constant3*pow4(algebraicState3)*(algebraicState0 - constant29);
  const double_v algebraicAlgebraic17 =  constant4*(algebraicState0 - constant30);
    // (not assigning to a parameter) algebraics = (VOI>=10.0000&&VOI<=10.5000 Vc::iif(VOI>=10.0000&&VOI<=10.5000 , (Vc::double_v(Vc::One)*( 20.0000 )), VOI>=50.0000&&VOI<=50.5000 Vc::iif( VOI>=50.0000&&VOI<=50.5000 , (Vc::double_v(Vc::One)*( 20.0000 )), VOI>=90.0000&&VOI<=90.5000 Vc::iif( VOI>=90.0000&&VOI<=90.5000 , (Vc::double_v(Vc::One)*( 20.0000 )), VOI>=130.000&&VOI<=130.500 Vc::iif( VOI>=130.000&&VOI<=130.500 , (Vc::double_v(Vc::One)*( 20.0000 )), Vc::double_v(Vc::Zero))))));
  const double_v algebraicRate0 = - (- parameters[0]+algebraicAlgebraic10+algebraicAlgebraic15+algebraicAlgebraic17)/constant1;
  const double_v algebraicAlgebraic18 = ((constant6 - algebraicState5) - algebraicState6) - algebraicState4;
  const double_v algebraicAlgebraic4 = (Vc::iif(algebraicState0>- 60.0000 , Vc::double_v(Vc::One), Vc::double_v(Vc::Zero)));
  const double_v algebraicAlgebraic14 = constant12+( (constant14 - constant12)*algebraicAlgebraic4*constant8)/( algebraicAlgebraic4*constant8+constant22);
  const double_v algebraicAlgebraic16 = constant13+( (constant15 - constant13)*algebraicAlgebraic4*constant8)/( algebraicAlgebraic4*constant8+constant22);
  const double_v algebraicRate4 = ( algebraicAlgebraic14*algebraicAlgebraic18+ constant33*algebraicState5+ constant36*algebraicState6) -  (algebraicAlgebraic16+constant32+constant37)*algebraicState4;
  const double_v algebraicAlgebraic5 = (Vc::iif(algebraicState8>constant5 , Vc::double_v(Vc::One),Vc::iif( algebraicState8<(Vc::double_v(Vc::One)*constant5) , (Vc::double_v(Vc::One)*( 8.00000 )), Vc::double_v(Vc::Zero))));
  const double_v algebraicAlgebraic6 = algebraicState5/constant6;
  const double_v algebraicAlgebraic7 = algebraicState6/constant6;
  const double_v algebraicAlgebraic8 =  ((( algebraicState6*algebraicState8+ algebraicState5*algebraicState7) -  constant7*constant5)/( constant27*constant5))*constant38;
  const double_v algebraicAlgebraic9 = (algebraicState6 - constant7)/constant27;


  // final step
  // y_n+1 = y_n + 0.5*[rhs(y_n) + rhs(y*)]
  states[0] += 0.5*timeStepWidth*(rate0 + algebraicRate0);
  states[1] += 0.5*timeStepWidth*(rate1 + algebraicRate1);
  states[2] += 0.5*timeStepWidth*(rate2 + algebraicRate2);
  states[3] += 0.5*timeStepWidth*(rate3 + algebraicRate3);
  states[4] += 0.5*timeStepWidth*(rate4 + algebraicRate4);
  states[5] += 0.5*timeStepWidth*(rate5 + algebraicRate5);
  states[6] += 0.5*timeStepWidth*(rate6 + algebraicRate6);
  states[7] += 0.5*timeStepWidth*(rate7 + algebraicRate7);
  states[8] += 0.5*timeStepWidth*(rate8 + algebraicRate8);

  if (stimulate)
  {
    for (int i = 0; i < std::min(3,(int)Vc::double_v::size()); i++)
    {
      states[0][i] = valueForStimulatedPoint;
    }
  }
  // store algebraics for transfer
  if (storeAlgebraicsForTransfer)
  {
    for (int i = 0; i < algebraicsForTransferIndices.size(); i++)
    {
      const int algebraic = algebraicsForTransferIndices[i];
      switch (algebraic)
      {
        // case 0: is a parameter
        case 1:
          algebraicsForTransfer[i] = algebraicAlgebraic1;
          break;
        case 2:
          algebraicsForTransfer[i] = algebraicAlgebraic2;
          break;
        case 3:
          algebraicsForTransfer[i] = algebraicAlgebraic3;
          break;
        case 4:
          algebraicsForTransfer[i] = algebraicAlgebraic4;
          break;
        case 5:
          algebraicsForTransfer[i] = algebraicAlgebraic5;
          break;
        case 6:
          algebraicsForTransfer[i] = algebraicAlgebraic6;
          break;
        case 7:
          algebraicsForTransfer[i] = algebraicAlgebraic7;
          break;
        case 8:
          algebraicsForTransfer[i] = algebraicAlgebraic8;
          break;
        case 9:
          algebraicsForTransfer[i] = algebraicAlgebraic9;
          break;
        case 10:
          algebraicsForTransfer[i] = algebraicAlgebraic10;
          break;
        case 11:
          algebraicsForTransfer[i] = algebraicAlgebraic11;
          break;
        case 12:
          algebraicsForTransfer[i] = algebraicAlgebraic12;
          break;
        case 13:
          algebraicsForTransfer[i] = algebraicAlgebraic13;
          break;
        case 14:
          algebraicsForTransfer[i] = algebraicAlgebraic14;
          break;
        case 15:
          algebraicsForTransfer[i] = algebraicAlgebraic15;
          break;
        case 16:
          algebraicsForTransfer[i] = algebraicAlgebraic16;
          break;
        case 17:
          algebraicsForTransfer[i] = algebraicAlgebraic17;
          break;
        case 18:
          algebraicsForTransfer[i] = algebraicAlgebraic18;
          break;

      }
    }
  }
}

// compute the rhs for a single instance, this can be used for computation of the equilibrium values of the states
#ifdef __cplusplus
extern "C"
#endif
void computeCellMLRightHandSideSingleInstance(void *context, double t, double *states, double *rates, double *algebraics, double *parameters)
{
  double VOI = t;   /* current simulation time */

  /* define constants */
  double CONSTANTS[39];
  CONSTANTS[0] = -75;
  CONSTANTS[1] = 1;
  CONSTANTS[2] = 120;
  CONSTANTS[3] = 36;
  CONSTANTS[4] = 0.3;
  CONSTANTS[5] = 8e-3;
  CONSTANTS[6] = 1;
  CONSTANTS[7] = 3.4e-13;
  CONSTANTS[8] = 10;
  CONSTANTS[9] = 1;
  CONSTANTS[10] = 0;
  CONSTANTS[11] = 7.815e-5;
  CONSTANTS[12] = 0;
  CONSTANTS[13] = 100e-3;
  CONSTANTS[14] = 120e-3;
  CONSTANTS[15] = 50e-3;
  CONSTANTS[16] = 4e-3;
  CONSTANTS[17] = 50e-3;
  CONSTANTS[18] = 8e-3;
  CONSTANTS[19] = 400e-3;
  CONSTANTS[20] = 6e-3;
  CONSTANTS[21] = 3.5400e-13;
  CONSTANTS[22] = 1;
  CONSTANTS[23] = 3.2;
  CONSTANTS[24] = 9.1362e-20;
  CONSTANTS[25] = 1.38e-23;
  CONSTANTS[26] = 310;
  CONSTANTS[27] = 0.015;
  CONSTANTS[28] = CONSTANTS[0]+115.000;
  CONSTANTS[29] = CONSTANTS[0] - 12.0000;
  CONSTANTS[30] = CONSTANTS[0]+10.6130;
  CONSTANTS[31] =  CONSTANTS[10]*CONSTANTS[11];
  CONSTANTS[32] = CONSTANTS[17];
  CONSTANTS[33] = CONSTANTS[19];
  CONSTANTS[34] = CONSTANTS[18];
  CONSTANTS[35] = CONSTANTS[20];
  CONSTANTS[36] = CONSTANTS[16];
  CONSTANTS[37] = 0.00000;
  CONSTANTS[38] = (CONSTANTS[9]<0.635000 ? 0.00000 : CONSTANTS[9]<0.835000 ?  4.20000*(CONSTANTS[9] - 0.635000) : CONSTANTS[9]<1.00000 ? 0.840000+ 0.969700*(CONSTANTS[9] - 0.835000) : CONSTANTS[9]<1.12500 ? 1.00000 : CONSTANTS[9]<1.82500 ? 1.00000 -  1.42860*(CONSTANTS[9] - 1.12500) : 0.00000);

  rates[5] = ( CONSTANTS[32]*states[4]+ CONSTANTS[35]*states[6]) -  (CONSTANTS[33]+CONSTANTS[34])*states[5];
  rates[6] = ( CONSTANTS[34]*states[5] -  (CONSTANTS[35]+CONSTANTS[36])*states[6])+ CONSTANTS[37]*states[4];
  rates[7] = ( (( - CONSTANTS[32]*states[4])/states[5])*states[7] -  (( CONSTANTS[35]*states[6])/states[5])*states[7])+CONSTANTS[31];
  rates[8] =  (( - CONSTANTS[34]*states[5])/states[6])*(states[8] - CONSTANTS[5])+CONSTANTS[31];
  algebraics[1] = ( - 0.100000*(states[0]+50.0000))/(exp(- (states[0]+50.0000)/10.0000) - 1.00000);
  algebraics[11] =  4.00000*exp(- (states[0]+75.0000)/18.0000);
  rates[1] =  algebraics[1]*(1.00000 - states[1]) -  algebraics[11]*states[1];
  algebraics[2] =  0.0700000*exp(- (states[0]+75.0000)/20.0000);
  algebraics[12] = 1.00000/(exp(- (states[0]+45.0000)/10.0000)+1.00000);
  rates[2] =  algebraics[2]*(1.00000 - states[2]) -  algebraics[12]*states[2];
  algebraics[3] = ( - 0.0100000*(states[0]+65.0000))/(exp(- (states[0]+65.0000)/10.0000) - 1.00000);
  algebraics[13] =  0.125000*exp((states[0]+75.0000)/80.0000);
  rates[3] =  algebraics[3]*(1.00000 - states[3]) -  algebraics[13]*states[3];
  algebraics[10] =  CONSTANTS[2]*pow(states[1], 3.00000)*states[2]*(states[0] - CONSTANTS[28]);
  algebraics[15] =  CONSTANTS[3]*pow(states[3], 4.00000)*(states[0] - CONSTANTS[29]);
  algebraics[17] =  CONSTANTS[4]*(states[0] - CONSTANTS[30]);
    // (not assigning to a parameter) algebraics = (VOI>=10.0000&&VOI<=10.5000 VOI>=10.0000&&VOI<=10.5000 ? 20.0000 : VOI>=50.0000&&VOI<=50.5000  VOI>=50.0000&&VOI<=50.5000 ? 20.0000 : VOI>=90.0000&&VOI<=90.5000  VOI>=90.0000&&VOI<=90.5000 ? 20.0000 : VOI>=130.000&&VOI<=130.500  VOI>=130.000&&VOI<=130.500 ? 20.0000 : 0.00000);
  rates[0] = - (- parameters[0]+algebraics[10]+algebraics[15]+algebraics[17])/CONSTANTS[1];
  algebraics[18] = ((CONSTANTS[6] - states[5]) - states[6]) - states[4];
  algebraics[4] = (states[0]>- 60.0000 ? 1.00000 : 0.00000);
  algebraics[14] = CONSTANTS[12]+( (CONSTANTS[14] - CONSTANTS[12])*algebraics[4]*CONSTANTS[8])/( algebraics[4]*CONSTANTS[8]+CONSTANTS[22]);
  algebraics[16] = CONSTANTS[13]+( (CONSTANTS[15] - CONSTANTS[13])*algebraics[4]*CONSTANTS[8])/( algebraics[4]*CONSTANTS[8]+CONSTANTS[22]);
  rates[4] = ( algebraics[14]*algebraics[18]+ CONSTANTS[33]*states[5]+ CONSTANTS[36]*states[6]) -  (algebraics[16]+CONSTANTS[32]+CONSTANTS[37])*states[4];
  algebraics[5] = (states[8]>CONSTANTS[5] ? 1.00000 : states[8]<CONSTANTS[5] ? 8.00000 : 0.00000);
  algebraics[6] = states[5]/CONSTANTS[6];
  algebraics[7] = states[6]/CONSTANTS[6];
  algebraics[8] =  ((( states[6]*states[8]+ states[5]*states[7]) -  CONSTANTS[7]*CONSTANTS[5])/( CONSTANTS[27]*CONSTANTS[5]))*CONSTANTS[38];
  algebraics[9] = (states[6] - CONSTANTS[7])/CONSTANTS[27];
}
