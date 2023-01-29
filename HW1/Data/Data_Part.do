/// Econ 810: Advanced Macroeconomic Theory
/// Professor: Carter Braxton
/// Problem Set 1: Variance of persistent and transitory shocks
/// Authors: Fernando de Lima Lopes, Stefano Lord-Medrano, Yeonggyu Yun
/// Date: 01/28/2023

///////////////////////////////////////////////////////////////////////////////
* Housekeeping
* Clear workspace
clear all

* Set directory
cd "/Users/smlm/Desktop/Datasets - Metrics/PSID data"
use pequiv_long.dta
* Install coefplot package
ssc install coefplot, replace

///////////////////////////////////////////////////////////////////////////////
* Part 1: Cleaning
* Drop observations after 1997 due to change in format of PSID
keep if inrange(year, 1978, 1997)
summ year

* Drop SEO oversample (see page 372 of Codebook for the Cross-National 
* Equivalent File)
drop if x11104 == 12
summ x11104
///////////////////////////////////////////////////////////////////////////////
* Part 2: Sample selection criteria
* Age between 30 and 65
keep if inrange(d11101, 30, 65)
* Marital status: married
keep if d11104 == 1
* Drop individuals that are in the first and last 10% of HH labor income distr.
keep if inrange(i11103, 11001, 84375)
* Describe data set and compute some statistics
describe
summarize
///////////////////////////////////////////////////////////////////////////////
* Part 3: Remove the life-cycle component of earnings (\kappa_{it})
* Note: y_{it}=\log(Y_{it})-\kappa_{it}
* We choose age as our observable.
* Compute log HH post-government income (TAXSIM)
* See page 218 of Codebook for the Cross-National Equivalent File
* Drop negative values
drop if i11113<0
gen log_y = log(i11113)
* Define panel
xtset x11101LL year, yearly
* Run panel regression for log_y on age
xtreg log_y i.d11101, fe vce(robust)
* Plot coefficients of age on log-income
coefplot, vertical drop(_cons) noci nolabel
///////////////////////////////////////////////////////////////////////////////
* Part 4: Compute variance of persistent (\zeta) and transitory (\varepsilon)
* shocks
* Compute residuals of panel regression and generate pseudo differences given by
* \Tilde{\Delta}y_{t}=y_{t}-\rho y_{t-1}
predict res_y, res
gen pseudo_dif_t0 = res_y - 0.97*L1.res_y
gen pseudo_dif_aux = (0.97^2)*L1.pseudo_dif_t0+0.97*pseudo_dif_t0+F1.pseudo_dif_t0
* Transitory shock variance
corr pseudo_dif_t0 F1.pseudo_dif_t0, cov
display (-1/0.97)*(-0.016923)
* Permanent shock variance
corr pseudo_dif_t0 pseudo_dif_aux, cov
display (1/0.97)*(0.016147)
