/// Econ 810: Advanced Macroeconomic Theory
/// Professor: Carter Braxton
/// Problem Set 1: Variance of persistent and transitory shocks (Simulated panel)
/// Authors: Fernando de Lima Lopes, Stefano Lord-Medrano, Yeonggyu Yun
/// Date: 01/28/2023

///////////////////////////////////////////////////////////////////////////////
* Housekeeping
* Clear workspace
clear all

* Set directory
cd "/Users/smlm/Desktop/Datasets - Metrics/PSID data"
use Simulated_Panel.dta

* Run panel regression for log_y on age
xtset id year, yearly
xtreg y i.year, fe vce(robust)
* Compute variance of persistent (\zeta) and transitory (\varepsilon)
* shocks
* Compute residuals of panel regression and generate pseudo differences given by
* \Tilde{\Delta}y_{t}=y_{t}-\rho y_{t-1}
predict res_y, res
gen pseudo_dif_t0 = res_y - 0.97*L1.res_y
gen pseudo_dif_aux = (0.97^2)*L1.pseudo_dif_t0+0.97*pseudo_dif_t0+F1.pseudo_dif_t0
gen dif_con_t0 = D1.c
* Pass through coefficients
* Transitory
correlate dif_con_t0 F1.pseudo_dif_t0, cov
correlate pseudo_dif_t0 F1.pseudo_dif_t0, cov
* Permanent
correlate dif_con_t0 pseudo_dif_aux, cov
correlate pseudo_dif_t0 pseudo_dif_aux, cov

* True part
gen y_true = z+e
gen pseudo_dif_t0_true = y_true - 0.97*L1.y_true
gen pseudo_dif_aux_true = (0.97^2)*L1.pseudo_dif_t0_true+0.97*pseudo_dif_t0_true+F1.pseudo_dif_t0_true
* Pass through coefficients
* Transitory
correlate dif_con_t0 F1.pseudo_dif_t0_true, cov
correlate pseudo_dif_t0_true F1.pseudo_dif_t0_true, cov
* Permanent
correlate dif_con_t0 pseudo_dif_aux_true, cov
correlate pseudo_dif_t0_true pseudo_dif_aux_true, cov
