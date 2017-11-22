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
	Error = []
	day_flag = ''
	for idx in fl.readlines():
		idx=idx.strip()
		point = idx.split(',')
#		print filname, idx
		day, values = point[0], point[5]
		if day != day_flag:
			day_flag = day
			tradec.append([])
			tradec[-1].append(float(values))
		else:
			tradec[-1].append(float(values))
			if len(tradec[-1])>48:
				print 'Error number in line %s in %s'%(idx, file)
				Error.append(idx)
	return Error

datpath = sys.argv[1]
csvdt = [ fil for fil in os.listdir(datpath) if isfile(join(datpath,fil))]
Total = []
for item in csvdt:
	#print item
	docmt = join(datpath, item)
	Total.append(stplot(docmt))
print len(Total)
print Total
