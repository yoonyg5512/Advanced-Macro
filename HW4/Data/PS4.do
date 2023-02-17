/// Econ 810: Advanced Macroeconomic Theory
/// Professor: Carter Braxton
/// Problem Set 4: Ben-Porath Models
/// Authors: Fernando de Lima Lopes, Stefano Lord-Medrano, Yeonggyu Yun
/// Date: 02/15/2023

///////////////////////////////////////////////////////////////////////////////
* Housekeeping
* Clear workspace
clear all

* Set directory and install packages
cd "/Users/smlm/Desktop/Datasets - Metrics/PSID data"
use pequiv_long.dta
ssc install rangestat, replace
///////////////////////////////////////////////////////////////////////////////
* Part 1: Filter panel and create main statistics
* Use data from 1978 to 1997
keep if inrange(year, 1978, 1997)
* Drop the SOE oversample
drop if x11104LL == 12
* Redefine some variables
rename x11101LL id
rename d11102LL gender
rename d11101 age
rename i11110 earnings
* Keep ages between 20 and 60 and male
keep if inrange(age, 20, 65)
keep if gender == 1
* Create age bins
gen age_bin = 0
replace age_bin = 1 if inrange(age, 20, 25)
replace age_bin = 2 if inrange(age, 26, 30)
replace age_bin = 3 if inrange(age, 31, 35)
replace age_bin = 4 if inrange(age, 36, 40)
replace age_bin = 5 if inrange(age, 41, 45)
replace age_bin = 6 if inrange(age, 46, 50)
replace age_bin = 7 if inrange(age, 51, 55)
replace age_bin = 8 if inrange(age, 56, 60)
replace age_bin = 9 if inrange(age, 61, 65)
* Compute statistics
gen log_earnings = log(earnings)
gen cohort = year - age
egen avg_earnings = mean(log_earnings), by(age_bin)
egen median_earnings = median(log_earnings), by(age_bin)
egen sd_earnings = sd(log_earnings), by(age_bin year)
gen var_earnings = sd_earnings^2
gen skewness = avg_earnings/median_earnings
egen fourth_moment = mean((log_earnings-avg_earnings)^2), by(age_bin year)
gen kurtosis_log_earnings = fourth_moment/(var_earnings^2)-3
* Run regressions
reg avg_earnings i.age_bin i.cohort, vce(robust)
// estimates store time_fe
// reg avg_earnings i.age_bin i.cohort, vce(robust)
// estimates store cohort_fe
coefplot, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Mean of Log-Earnings") xtitle("Age") color(blue%50) graphregion(fcolor(white))
reg var_earnings i.age_bin i.cohort, vce(robust)
coefplot, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Variance of Log-Earnings") xtitle("Age") color(blue%50) graphregion(fcolor(white))
reg skewness i.age_bin i.cohort, vce(robust)
coefplot, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Log-Earnings skewness") xtitle("Age") color(blue%50) graphregion(fcolor(white))
reg kurtosis_log_earnings i.age_bin i.cohort, vce(robust)
coefplot, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Log-Earnings kurtosis") xtitle("Age") color(blue%50) graphregion(fcolor(white))
