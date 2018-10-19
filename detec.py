#! /usr/bin/env python

''' The following section of code has references to crossDetectionStub.py posted in canvas with additional comments provided for easier understanding. 
    Following which is the code section of the algorithm to detect crossings in one particular link
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
import getopt, ast
from pylab import *

# Default inputs if not provided on command line
linkCombosToPlot = []
numNodes         = 10
infile           = sys.stdin
fromStdin        = True

# Accept command line inputs to list the links to use; the input file
myopts, args = getopt.getopt(sys.argv[1:],"l:n:f:")
for o, a in myopts:
    if o == "-l":
        inputList = ast.literal_eval( a )
        linkCombosToPlot.append(inputList)
    elif o == "-n":
        numNodes = int(a)
    elif o == "-f":
        infile = open(a)
        fromStdin = False

def linkNumForTxRxChLists(tx, rx, ch, nodeList, channelList):
    if (nodeList.count(tx) == 0) or (nodeList.count(rx) == 0) or (channelList.count(ch) == 0):
        sys.stderr.write('Error in linkNumForTxRx: tx, rx, or ch number invalid')
    rx_enum = nodeList.index(rx)
    tx_enum = nodeList.index(tx)
    ch_enum = channelList.index(ch)
    nodes = len(nodeList)
    links = nodes*(nodes-1)
    linknum = ch_enum*links + tx_enum*(nodes-1) + rx_enum
    if (rx_enum > tx_enum):
        linknum -= 1
    return linknum

if numNodes <= 1:
    sys.stderr.write('Usage: %s -n numberOfNodes -l "[tx,rx,ch]" ' % sys.argv[0])
    sys.stderr.write('numberOfNodes is required and must be greater than 1')

# If no link combination provided, just plot [tx=1,rx=2,ch=0]

buffL       = 5 # size of the moving window defined here different from the training period.
startSkip   = 1
training_period = 25 # first 15 seconds which is 15 rows where there is no movement, called the training period
r = 3 # defined in the paper 

# snippet retained from crossdetectionstub.py in order to get the linkNumList with parameters nodeList and channelList
for i in range(startSkip): 
    line = infile.readline()

lineInt = [float(i) for i in line.split()]
columns = len(lineInt)
rss  = [int(f) for f in lineInt[:-1]]  # take all columns except for last column representing time in seconds
numLinks = len(rss)
numChs  = (numLinks // ((numNodes-1)*numNodes))
nodeList = range(1,numNodes+1)
channelList = range(numChs)

if linkCombosToPlot == []:
    linkCombosToPlot.append([1,2,0])
links = len(linkCombosToPlot)

linkNumList = [] # gives the column numbers for respective tx, rx, ch pairs given.
for l in linkCombosToPlot:
    linkNumList.append(linkNumForTxRxChLists(l[0], l[1], l[2], nodeList, channelList))

# reading data from my '.txt' file and 
data = np.genfromtxt('cross3.txt')
rss = data[:,linkNumList] # linkNumList ensures we pick only relevant node pair channel data

# Filter out the bad readings which is 127's as  it is not a valid reading.
rss[rss==127] = np.nan
time_ms = data[:,-1]
time = (time_ms - time_ms[0])/1000.0 # convert into seconds 

var_avg = np.zeros(len(time) - buffL) # buffL must be taken into consideration else elements get kicked out as it is a moving window.

'''
* val:val+buffL ensures the sliding window is considered while taking rss values at every time instant.
* buffer shape is (7,8,) and we take columns from this as 8, since 'var_total' is variance monitored across all the channels(8 in number)
* var_avg is calculated as the total mean for all links. nanmean , nanvar is used since we replaced 127's as nan to be not considered (instead of changing them to zero)
'''
detections = []
for val in range(len(time)-buffL):
    buffer = rss[val:val+buffL,:]  
    var_total = [np.nanvar(buffer[:,k]) for k in range(buffer.shape[1])] 
    var_avg[val] = np.nanmean(var_total)

''' From youssef 2007, criterion for threshold based detection depends on the static/training period. 
    Hence average the variance_avg over first 15 rows of data which is the training period
    Quoting from paper, higher the  value of 'r' lesser false detections/alarms are observed '''
    
threshold = r * np.mean(var_avg[:training_period]) 

#create subplots to view detections marked beneath the rss plot.

figure(1)
plt.subplot(2,2,1)
plt.plot(time, rss)
plt.xlabel(" Time in seconds")
plt.ylabel(" Rss in dBm")
plt.subplot(2,2,3)
plt.plot(time[:-buffL],(var_avg > threshold)) #everytime the condition is satisfied it plots a high and otherwise.
plt.xlabel("Time in seconds")
plt.ylabel("Detections")
plt.show()

detections.append(var_avg > threshold)
data = np.array(detections, dtype=np.int32)

np.savetxt('output3.txt', data, fmt="%d")
