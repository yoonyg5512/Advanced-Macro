/// Econ 810: Advanced Macroeconomic Theory
/// Professor: Carter Braxton
/// Problem Set 2: Earnings and job loss
/// Authors: Fernando de Lima Lopes, Stefano Lord-Medrano, Yeonggyu Yun
/// Date: 02/02/2023

///////////////////////////////////////////////////////////////////////////////
* Housekeeping
* Clear workspace
clear all

* Set directory
cd "/Users/smlm/Desktop/Datasets - Metrics/PSID data"
use pequiv_long.dta

///////////////////////////////////////////////////////////////////////////////
* Part 1: Earnings gains while employed
* Define panel
xtset x11101LL year, yearly
* Drop SEO oversample (see page 372 of Codebook for the Cross-National 
* Equivalent File)
drop if x11104 == 12
* Difference in years
by x11101LL: gen dif_year = year - year[_n-1]
* Hours worked
keep if e11101 >= 2000
* Consecutive years
replace dif_year = 0 if missing(dif_year)
keep if dif_year<=1
* Generate difference in earnings
gen dif_earnings = D1.i11110
summ dif_earnings

///////////////////////////////////////////////////////////////////////////////
* Part 2: Simulation
clear
set obs 1000
gen id = _n
expand 11
bysort id: gen year = _n
xtset id year
gen e = rnormal(0,5000) if year == 1
gen y = .
replace y = 30000+e
gen losers=0
replace losers = 1 if id <= 500
gen u = rnormal(0,5000) if year >= 2
gen loss = 0
bysort id (year): replace loss=-9000 if _n == 6 & id <= 500

gen increment=0
foreach t of num 2/11 {
	bysort id (year): replace increment = 1000 + u + loss
}

gen earnings = y
foreach i of num 2/11 {
	bysort id (year): replace earnings= earnings[`i'-1] + increment[`i'] if _n == `i'
}

drop y e u increment losers loss
///////////////////////////////////////////////////////////////////////////////
* Part 3: Distributed lag regression
by id: gen minus_4 = 0
by id: replace minus_4 = 1 if year - 6 == -4 & id <=500
by id: gen minus_3 = 0
by id: replace minus_3 = 1 if year - 6 == -3 & id <=500
by id: gen minus_2 = 0
by id: replace minus_2 = 1 if year - 6 == -2 & id <=500
by id: gen minus_1 = 0
by id: replace minus_1 = 1 if year - 6 == -1 & id <=500
by id: gen minus_0 = 0
by id: replace minus_0 = 1 if year - 6 == 0 & id <=500
by id: gen plus_1 = 0
by id: replace plus_1 = 1 if year - 6 == 1 & id <=500
by id: gen plus_2 = 0
by id: replace plus_2 = 1 if year - 6 == 2 & id <=500
by id: gen plus_3 = 0
by id: replace plus_3 = 1 if year - 6 == 3 & id <=500
by id: gen plus_4 = 0
by id: replace plus_4 = 1 if year - 6 == 4 & id <=500
by id: gen plus_5 = 0
by id: replace plus_5 = 1 if year - 6 == 5 & id <=500

xtreg earnings i.year minus_4 minus_3 minus_2 minus_1 minus_0 plus_1 plus_2 plus_3 plus_4 plus_5, fe vce(robust)
coefplot, vertical keep(minus_4 minus_3 minus_2 minus_1 minus_0 plus_1 plus_2 plus_3 plus_4 plus_5) nolabel recast(connected)

coefplot, vertical drop(_cons minus_4 minus_3 minus_2 minus_1 minus_0 plus_1 plus_2 plus_3 plus_4 plus_5) nolabel

///////////////////////////////////////////////////////////////////////////////
* Part 4: Working with simulated data
* Clear workspace
clear all
* Use simulated panel from model
cd "/Users/smlm/Desktop/Datasets - Metrics/PSID data/PS2"
use Simulated_Panel.dta
* Add package for winsorizing data
ssc install winsor, replace

* Plotting distribution of human capital among employed and unemployed
kdensity h_cap if status == 1, kernel(epanechnikov) generate(pts_emp den_emp) nograph
kdensity h_cap if status == 0, kernel(epanechnikov) generate(pts_u den_u) nograph
twoway (area den_emp pts_emp) (area den_u pts_u)

* Winsorizing to replace extreme values with 10th and 90th percentile
winsor h_cap, p(.05) gen(h_cap_w05)
kdensity h_cap_w05 if status == 1, kernel(epanechnikov) generate(pts_empw den_empw) nograph
kdensity h_cap_w05 if status == 0, kernel(epanechnikov) generate(pts_uw den_uw) nograph
twoway (area den_empw pts_empw) (area den_uw pts_uw)

* Clean
drop pts_emp den_emp pts_u den_u h_cap_w05 pts_empw den_empw pts_uw den_uw

* Average gain in earnings for individuals who are working for two cons. years
xtset id month
gen earnings = w*h_cap
* Consecutive years
gen consec = 0
by id: replace consec = 1 if (status + status[_n-1])/2 == 1
* Statistic
by id: gen gain = earnings - earnings[_n-1] if consec == 1
summ gain

replace status = -1 if status == 0
bysort id (month): gen statuschg = sum(status~=status[_n-1] & _n~=1)*status
