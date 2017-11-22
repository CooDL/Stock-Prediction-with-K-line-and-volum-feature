import os
from os.path import isfile, join
import sys
import PyGnuplot as gp
import numpy as np
from collections import Counter
import shutil as stl
import time


def stplot(filname):

	fl = open(filname,'r')
	file = filname.split('/')[-1].split('.')[0]
	picformat = 'png'
	tradec = []
	day_flag = ''
	for idx in fl.readlines():
		idx=idx.strip()
		point = idx.split(',')
		day, values = point[0], point[5]
		if day != day_flag:
			day_flag = day
			tradec.append([])
			tradec[-1].append(float(values))
		else:
			tradec[-1].append(float(values))
			assert len(tradec[-1])<=48, 'Error number in line %s in %s'%(idx, file)
	#tradec = np.array(tradec[1])# array.tolist(tradec)
	#================To Array========================#
	tradeshape = [len(tradec),len(tradec[0])]
	tradearray = np.zeros(tradeshape,dtype=np.float)
	for i, dattr in enumerate(tradec):
	#	print dattr
		dattr = np.array(dattr)
		tradearray[i, 0:len(dattr)] = dattr
	#print tradearray[1]

	#dt_day = tradec[i,:]#48
	#dt_wk = tradec[-7:i, 3::4] #from i previous 7, with sencond dim from 2 with step 2,not include i 7*12=84
	#dt_mt = tradec[-30:i, 15::16] #from i previous 30, with second dim from 12 with step 12 30*3=90
	#dt_yr = tradec[-365:i::5, -1] #from i previous 365, with single data one day

	#================Setting Gnuplot==================#
	gp.c('set term %s size 224,224'%picformat)
	gp.c('unset key')
	gp.c('unset border')
	gp.c('unset xtics')
	gp.c('unset ytics')
	#gp.c('set style data lines')
	gp.c('set rmargin 0.1')
	gp.c('set lmargin 0.1')
	gp.c('set tmargin 0.1')
	gp.c('set bmargin 0.1')

	#=============Create Polority Dir=================#
	if not os.path.exists('up'):
		os.mkdir('up')
	if not os.path.exists('down'):
		os.mkdir('down')
	if not os.path.exists('bal'):
		os.mkdir('bal')

	#===============All points start from 366 day===========#
	for i, dat in enumerate(tradearray[365:len(tradearray)-1]):
		X = np.arange(len(dat))
		Y = dat.tolist()
		#=========Plot Day==============
		gp.s([X,Y],'daypoint%d.dat'%i)
		day_filename = file+'_'+str(i)+'_d'+'.%s'%picformat
	#	print day_filename
	#print len(tradearray),len(tradearray[365:])
		gp.c('set output \'%s\''%day_filename)
		gp.c('plot "daypoint%d.dat" using 1:2 with lines linewidth 2 linecolor 8 title "D"'%i)

		#=========Plot Week==============
		dt_wk = tradearray[i+358:i+365, 3::4] #not include i, to include i, make i+357:i+366
		dt_wk = np.reshape(dt_wk, -1)
		dt_wk_x = np.arange(len(dt_wk))
		#print len(dt_wk)
		gp.s([dt_wk_x,dt_wk],'weekpoint%d.dat'%i)
		week_filename = file+'_'+str(i)+'_w'+'.%s'%picformat
		gp.c('set output \'%s\''%week_filename)
		gp.c('plot "weekpoint%d.dat" using 1:2 with lines linewidth 2 linecolor 8 title "W"'%i)
	 
		#=========Plot Month==============
		dt_mt = tradearray[i+335:i+365, 15::16] #not include i, to include i, make i+335:i+366
		dt_mt = np.reshape(dt_mt, -1)
		dt_mt_x = np.arange(len(dt_mt))
		#print len(dt_mt)
		gp.s([dt_mt_x,dt_mt],'monthpoint%d.dat'%i)
		month_filename = file+'_'+str(i)+'_m'+'.%s'%picformat
		gp.c('set output \'%s\''%month_filename)
		gp.c('plot "monthpoint%d.dat" using 1:2 with lines linewidth 2 linecolor 8 title "M"'%i)

		#=========Plot Year==============
		dt_yr = tradearray[i:i+365, -1] #not include i, to include i, make i+335:i+366 Every 120 points
		dt_yr = dt_yr[2::3]
		dt_yr = np.reshape(dt_yr, -1)
		dt_yr_x = np.arange(len(dt_yr))
	#	print len(dt_yr)
		gp.s([dt_yr_x,dt_yr],'yearpoint%d.dat'%i)
		year_filename = file+'_'+str(i)+'_y'+'.%s'%picformat
		gp.c('set output \'%s\''%year_filename)
		gp.c('plot "yearpoint%d.dat" using 1:2 with lines linewidth 2 linecolor 8 title "Y"'%i)

		#==========Notice==============
		if i%1000==0:
			print('%d Sets Finished'%i)
	print('Complete Plot')
	time.sleep(30)
	#===========Classify the Pic===========
	for i, dat in enumerate(tradearray[365:len(tradearray)-1]):
		assert tradearray[i+365][-1] == dat[-1],'No match'
		all_file = file+'_'+str(i)+'_*'
		if tradearray[i+366][-1] > dat[-1]:
			os.system('mv %s up/'%all_file)
		elif tradearray[i+366][-1] < dat[-1]:
			os.system('mv %s down/'%all_file)
		elif tradearray[i+366][-1] == dat[-1]:
			os.system('mv %s bal/'%all_file)

	os.system('rm *.dat')


datpath = sys.argv[1]
csvdt = [ fil for fil in os.listdir(datpath) if isfile(join(datpath,fil))]
for item in csvdt:
	docmt = join(datpath, item)
	stplot(docmt)



