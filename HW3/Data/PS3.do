/// Econ 810: Advanced Macroeconomic Theory
/// Professor: Carter Braxton
/// Problem Set 3: Directed search and insurance against job loss
/// Authors: Fernando de Lima Lopes, Stefano Lord-Medrano, Yeonggyu Yun
/// Date: 02/12/2023

///////////////////////////////////////////////////////////////////////////////
* Housekeeping
* Clear workspace
clear all

* Set directory
cd "/Users/smlm/Desktop/Datasets - Metrics/PSID data"
use pequiv_long.dta
///////////////////////////////////////////////////////////////////////////////
* Part 1: Definition of treatment and control group
* Define panel
rename x11101LL id
xtset id year, yearly
* Drop SEO oversample (see page 372 of Codebook for the Cross-National 
* Equivalent File)
drop if x11104 == 12
* Create indicator for full time
gen full_time = 0
by id: replace full_time = 1 if e11101 >= 2000
* Drop non responders and identify changes in number of childs in HH
drop if d11107 == .s
by id: gen members = d11107 - d11107[_n-1]
by id: egen change = sum(abs(members))
by id: drop if change != 0
by id: gen work_years = sum(full_time)
by id: gen worked_hours = sum(e11101)
by id: gen num_years = sum(year - year[_n-1])
by id: gen laid_off = 0
by id: replace laid_off = 1 if (e11101 + e11101[_n-1] + e11101[_n-2] + e11101[_n-3] <= 7500) & (full_time[_n-1]+full_time[_n-2]+full_time[_n-3] == 3)
by id: egen treated = sum(laid_off)

* Generate indicators for DL
by id: gen minus_3 = 0
by id: replace minus_3 = 1 if laid_off[_n+3] == 1
by id: gen minus_2 = 0
by id: replace minus_2 = 1 if laid_off[_n+2] == 1
by id: gen minus_1 = 0
by id: replace minus_1 = 1 if laid_off[_n+1] == 1
by id: gen minus_0 = 0
by id: replace minus_0 = 1 if laid_off == 1
by id: gen plus_1 = 0
by id: replace plus_1 = 1 if laid_off[_n-1] == 1
by id: gen plus_2 = 0
by id: replace plus_2 = 1 if laid_off[_n-2] == 1
by id: gen plus_3 = 0
by id: replace plus_3 = 1 if laid_off[_n-3] == 1
by id: gen plus_4 = 0
by id: replace plus_4 = 1 if laid_off[_n-4] == 1
by id: gen plus_5 = 0
by id: replace plus_5 = 1 if laid_off[_n-5] == 1
* Run DL regression
rename i11110 earnings
xtreg earnings i.year minus_3 minus_2 minus_1 minus_0 plus_1 plus_2 plus_3 plus_4 plus_5, fe vce(robust)
coefplot, vertical keep(minus_3 minus_2 minus_1 minus_0 plus_1 plus_2 plus_3 plus_4 plus_5) nolabel recast(connected) xline(4) ytitle("Earnings") xtitle("Year") color(blue%50) graphregion(fcolor(white))
