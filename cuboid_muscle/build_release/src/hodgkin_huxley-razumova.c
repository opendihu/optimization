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
void
initConsts(double* CONSTANTS, double* RATES, double *STATES)
{
STATES[0] = -75;
CONSTANTS[0] = -75;
CONSTANTS[1] = 1;
CONSTANTS[2] = 120;
STATES[1] = 0.05;
STATES[2] = 0.6;
CONSTANTS[3] = 36;
STATES[3] = 0.325;
CONSTANTS[4] = 0.3;
STATES[4] = 3.8e-14;
STATES[5] = 1e-14;
STATES[6] = 3.4e-13;
STATES[7] = 1e-16;
STATES[8] = 8e-3;
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
}
void
computeRates(double VOI, double* CONSTANTS, double* RATES, double* STATES, double* ALGEBRAIC)
{
RATES[5] = ( CONSTANTS[32]*STATES[4]+ CONSTANTS[35]*STATES[6]) -  (CONSTANTS[33]+CONSTANTS[34])*STATES[5];
RATES[6] = ( CONSTANTS[34]*STATES[5] -  (CONSTANTS[35]+CONSTANTS[36])*STATES[6])+ CONSTANTS[37]*STATES[4];
RATES[7] = ( (( - CONSTANTS[32]*STATES[4])/STATES[5])*STATES[7] -  (( CONSTANTS[35]*STATES[6])/STATES[5])*STATES[7])+CONSTANTS[31];
RATES[8] =  (( - CONSTANTS[34]*STATES[5])/STATES[6])*(STATES[8] - CONSTANTS[5])+CONSTANTS[31];
ALGEBRAIC[1] = ( - 0.100000*(STATES[0]+50.0000))/(exp(- (STATES[0]+50.0000)/10.0000) - 1.00000);
ALGEBRAIC[11] =  4.00000*exp(- (STATES[0]+75.0000)/18.0000);
RATES[1] =  ALGEBRAIC[1]*(1.00000 - STATES[1]) -  ALGEBRAIC[11]*STATES[1];
ALGEBRAIC[2] =  0.0700000*exp(- (STATES[0]+75.0000)/20.0000);
ALGEBRAIC[12] = 1.00000/(exp(- (STATES[0]+45.0000)/10.0000)+1.00000);
RATES[2] =  ALGEBRAIC[2]*(1.00000 - STATES[2]) -  ALGEBRAIC[12]*STATES[2];
ALGEBRAIC[3] = ( - 0.0100000*(STATES[0]+65.0000))/(exp(- (STATES[0]+65.0000)/10.0000) - 1.00000);
ALGEBRAIC[13] =  0.125000*exp((STATES[0]+75.0000)/80.0000);
RATES[3] =  ALGEBRAIC[3]*(1.00000 - STATES[3]) -  ALGEBRAIC[13]*STATES[3];
ALGEBRAIC[10] =  CONSTANTS[2]*pow(STATES[1], 3.00000)*STATES[2]*(STATES[0] - CONSTANTS[28]);
ALGEBRAIC[15] =  CONSTANTS[3]*pow(STATES[3], 4.00000)*(STATES[0] - CONSTANTS[29]);
ALGEBRAIC[17] =  CONSTANTS[4]*(STATES[0] - CONSTANTS[30]);
ALGEBRAIC[0] = (VOI>=10.0000&&VOI<=10.5000 ? 20.0000 : VOI>=50.0000&&VOI<=50.5000 ? 20.0000 : VOI>=90.0000&&VOI<=90.5000 ? 20.0000 : VOI>=130.000&&VOI<=130.500 ? 20.0000 : 0.00000);
RATES[0] = - (- ALGEBRAIC[0]+ALGEBRAIC[10]+ALGEBRAIC[15]+ALGEBRAIC[17])/CONSTANTS[1];
ALGEBRAIC[18] = ((CONSTANTS[6] - STATES[5]) - STATES[6]) - STATES[4];
ALGEBRAIC[4] = (STATES[0]>- 60.0000 ? 1.00000 : 0.00000);
ALGEBRAIC[14] = CONSTANTS[12]+( (CONSTANTS[14] - CONSTANTS[12])*ALGEBRAIC[4]*CONSTANTS[8])/( ALGEBRAIC[4]*CONSTANTS[8]+CONSTANTS[22]);
ALGEBRAIC[16] = CONSTANTS[13]+( (CONSTANTS[15] - CONSTANTS[13])*ALGEBRAIC[4]*CONSTANTS[8])/( ALGEBRAIC[4]*CONSTANTS[8]+CONSTANTS[22]);
RATES[4] = ( ALGEBRAIC[14]*ALGEBRAIC[18]+ CONSTANTS[33]*STATES[5]+ CONSTANTS[36]*STATES[6]) -  (ALGEBRAIC[16]+CONSTANTS[32]+CONSTANTS[37])*STATES[4];
}
void
computeVariables(double VOI, double* CONSTANTS, double* RATES, double* STATES, double* ALGEBRAIC)
{
ALGEBRAIC[1] = ( - 0.100000*(STATES[0]+50.0000))/(exp(- (STATES[0]+50.0000)/10.0000) - 1.00000);
ALGEBRAIC[11] =  4.00000*exp(- (STATES[0]+75.0000)/18.0000);
ALGEBRAIC[2] =  0.0700000*exp(- (STATES[0]+75.0000)/20.0000);
ALGEBRAIC[12] = 1.00000/(exp(- (STATES[0]+45.0000)/10.0000)+1.00000);
ALGEBRAIC[3] = ( - 0.0100000*(STATES[0]+65.0000))/(exp(- (STATES[0]+65.0000)/10.0000) - 1.00000);
ALGEBRAIC[13] =  0.125000*exp((STATES[0]+75.0000)/80.0000);
ALGEBRAIC[10] =  CONSTANTS[2]*pow(STATES[1], 3.00000)*STATES[2]*(STATES[0] - CONSTANTS[28]);
ALGEBRAIC[15] =  CONSTANTS[3]*pow(STATES[3], 4.00000)*(STATES[0] - CONSTANTS[29]);
ALGEBRAIC[17] =  CONSTANTS[4]*(STATES[0] - CONSTANTS[30]);
ALGEBRAIC[0] = (VOI>=10.0000&&VOI<=10.5000 ? 20.0000 : VOI>=50.0000&&VOI<=50.5000 ? 20.0000 : VOI>=90.0000&&VOI<=90.5000 ? 20.0000 : VOI>=130.000&&VOI<=130.500 ? 20.0000 : 0.00000);
ALGEBRAIC[18] = ((CONSTANTS[6] - STATES[5]) - STATES[6]) - STATES[4];
ALGEBRAIC[4] = (STATES[0]>- 60.0000 ? 1.00000 : 0.00000);
ALGEBRAIC[14] = CONSTANTS[12]+( (CONSTANTS[14] - CONSTANTS[12])*ALGEBRAIC[4]*CONSTANTS[8])/( ALGEBRAIC[4]*CONSTANTS[8]+CONSTANTS[22]);
ALGEBRAIC[16] = CONSTANTS[13]+( (CONSTANTS[15] - CONSTANTS[13])*ALGEBRAIC[4]*CONSTANTS[8])/( ALGEBRAIC[4]*CONSTANTS[8]+CONSTANTS[22]);
ALGEBRAIC[5] = (STATES[8]>CONSTANTS[5] ? 1.00000 : STATES[8]<CONSTANTS[5] ? 8.00000 : 0.00000);
ALGEBRAIC[6] = STATES[5]/CONSTANTS[6];
ALGEBRAIC[7] = STATES[6]/CONSTANTS[6];
ALGEBRAIC[8] =  ((( STATES[6]*STATES[8]+ STATES[5]*STATES[7]) -  CONSTANTS[7]*CONSTANTS[5])/( CONSTANTS[27]*CONSTANTS[5]))*CONSTANTS[38];
ALGEBRAIC[9] = (STATES[6] - CONSTANTS[7])/CONSTANTS[27];
}
