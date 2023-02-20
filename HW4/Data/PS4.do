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
* Part 1: Data assignment
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
egen avg_earnings = mean(log_earnings), by(age_bin year)
egen median_earnings = median(log_earnings), by(age_bin year)
egen sd_earnings = sd(log_earnings), by(age_bin year)
gen var_earnings = sd_earnings^2
gen skewness = avg_earnings/median_earnings
egen obs = count(id), by(age_bin year)
egen third_moment = sum((log_earnings-avg_earnings)^3), by(age_bin year)
gen third_moment_corr = third_moment/obs
gen skewness_2 = third_moment_corr/(var_earnings^1.5)
egen fourth_moment = sum((log_earnings-avg_earnings)^4), by(age_bin year)
gen fourth_moment_corr = fourth_moment/obs
gen kurtosis_log_earnings = (fourth_moment_corr/(var_earnings^2))-3
* Run regressions
* Mean
reg avg_earnings i.age_bin i.year, vce(robust)
estimates store time_fe_m
reg avg_earnings i.age_bin i.cohort, vce(robust)
estimates store cohort_fe_m
coefplot time_fe_m cohort_fe_m, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Mean of Log-Earnings") xtitle("Age") color(blue%50) graphregion(fcolor(white))
* Variance
reg var_earnings i.age_bin i.year, vce(robust)
estimates store time_fe_v
reg var_earnings i.age_bin i.cohort, vce(robust)
estimates store cohort_fe_v
coefplot time_fe_v cohort_fe_v, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Variance of Log-Earnings") xtitle("Age") color(blue%50) graphregion(fcolor(white))
* Skewness
reg skewness i.age_bin i.year, vce(robust)
estimates store time_fe_s
reg skewness i.age_bin i.cohort, vce(robust)
estimates store cohort_fe_s
coefplot time_fe_s cohort_fe_s, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Log-Earnings skewness") xtitle("Age") color(blue%50) graphregion(fcolor(white))

* Different calculation for skewness
reg skewness_2 i.age_bin i.year, vce(robust)
estimates store time_fe_s
reg skewness_2 i.age_bin i.cohort, vce(robust)
estimates store cohort_fe_s
coefplot time_fe_s cohort_fe_s, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Log-Earnings skewness") xtitle("Age") color(blue%50) graphregion(fcolor(white))


* Kurtosis
reg kurtosis_log_earnings i.age_bin i.year, vce(robust)
estimates store time_fe_k
reg kurtosis_log_earnings i.age_bin i.cohort, vce(robust)
estimates store cohort_fe_k
coefplot time_fe_k cohort_fe_k, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Log-Earnings kurtosis") xtitle("Age") color(blue%50) graphregion(fcolor(white))

///////////////////////////////////////////////////////////////////////////////
* Part 2: Model assignment
clear all
cd "/Users/smlm/Desktop/Datasets - Metrics/PSID data/PS4"
import delimited using Simulated_panel
* Create some variables
gen cohort = 1979+year - age
gen age_bin = 0
replace age_bin = 1 if inrange(age, 1, 5)
replace age_bin = 2 if inrange(age, 6, 10)
replace age_bin = 3 if inrange(age, 11, 15)
replace age_bin = 4 if inrange(age, 16, 20)
replace age_bin = 5 if inrange(age, 21, 25)
replace age_bin = 6 if inrange(age, 26, 30)

* Compute statistics
gen log_earnings = log(earnings)
egen avg_earnings = mean(log_earnings), by(age_bin year)
egen median_earnings = median(log_earnings), by(age_bin year)
egen sd_earnings = sd(log_earnings), by(age_bin year)
gen var_earnings = sd_earnings^2
gen skewness = avg_earnings/median_earnings
egen obs = count(id), by(age_bin year)
egen third_moment = sum((log_earnings-avg_earnings)^3), by(age_bin year)
gen third_moment_corr = third_moment/obs
gen skewness_2 = third_moment_corr/(var_earnings^1.5)
egen fourth_moment = sum((log_earnings-avg_earnings)^4), by(age_bin year)
gen fourth_moment_corr = fourth_moment/obs
gen kurtosis_log_earnings = (fourth_moment_corr/(var_earnings^2))-3

* Run regressions
* Mean
reg avg_earnings i.age_bin i.year, vce(robust)
estimates store time_fe_m
reg avg_earnings i.age_bin i.cohort, vce(robust)
estimates store cohort_fe_m
coefplot time_fe_m cohort_fe_m, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Mean of log-earnings") xtitle("Age") color(blue%50) graphregion(fcolor(white))
* Variance
reg var_earnings i.age_bin i.year, vce(robust)
estimates store time_fe_v
reg var_earnings i.age_bin i.cohort, vce(robust)
estimates store cohort_fe_v
coefplot time_fe_v cohort_fe_v, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Variance of log-earnings") xtitle("Age") color(blue%50) graphregion(fcolor(white))
* Skewness
reg skewness i.age_bin i.year, vce(robust)
estimates store time_fe_s
reg skewness i.age_bin i.cohort, vce(robust)
estimates store cohort_fe_s
coefplot time_fe_s cohort_fe_s, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Log-earnings skewness") xtitle("Age") color(blue%50) graphregion(fcolor(white))

* Different calculation for skewness
reg skewness_2 i.age_bin i.year, vce(robust)
estimates store time_fe_s
reg skewness_2 i.age_bin i.cohort, vce(robust)
estimates store cohort_fe_s
coefplot time_fe_s cohort_fe_s, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Log-earnings skewness") xtitle("Age") color(blue%50) graphregion(fcolor(white))


* Kurtosis
reg kurtosis_log_earnings i.age_bin i.year, vce(robust)
estimates store time_fe_k
reg kurtosis_log_earnings i.age_bin i.cohort, vce(robust)
estimates store cohort_fe_k
coefplot time_fe_k cohort_fe_k, vertical keep(2.age_bin 3.age_bin 4.age_bin 5.age_bin 6.age_bin 7.age_bin 8.age_bin) nolabel recast(connected) ytitle("Log-earnings kurtosis") xtitle("Age") color(blue%50) graphregion(fcolor(white))

* Generate measure of life-time earnings following Guvenen et.al. (2022, AEJ: Applied)
xtset id year, yearly
by id: egen num_years = count(year)
gen log_earnings = ln(earnings)
by id: gen first = age[1]
drop if log_earnings >= .
keep if first == 1
keep if num_years >= 20
gen cohort = 1979+year - age
egen life_earn = mean(log_earnings), by(id)
egen median_life_earn = median(life_earn), by(cohort)
twoway (scatter median_life_earn cohort), ytitle("Median life earnings") xtitle("Cohort entry year") graphregion(fcolor(white))
egen sd_life_earn = sd(life_earn), by(year)
twoway (scatter sd_life_earn year), ytitle("Std.Dev. life earnings") xtitle("Year") graphregion(fcolor(white))
