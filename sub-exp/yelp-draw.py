import numpy as np
# import matplotlib as plt
from matplotlib import pyplot as plt
import csv 
t=[]
HUCB=[]
CON=[]
LIN=[]

with open('reg-reward.csv', 'r') as f:
	reader = csv.reader(f)
	for i in reader:
		if i[0]!='Time':
			# print(i[0])
			t.append(200*int(i[0]))
			LIN.append(float(i[1]))
			CON.append(float(i[2]))
			HUCB.append(float(i[3]))
ft = 18
# plot it!



plt.figure(figsize=(8,6))
plt.plot(t, CON, lw=2, label=r'ConUCB', color='blue',linestyle='-.')
plt.plot(t, HUCB, lw=2, label=r'Hier-LinUCB', color='red')
plt.plot(t, LIN, lw=2, label=r'LinUCB', color='green',linestyle='--')
plt.legend(loc='upper left',fontsize=ft)
plt.xlabel('Iteration',fontsize=ft)
plt.ylabel('Cumulative Regret',fontsize=ft)
plt.yticks(range(0,80000,10000),fontsize=ft)
plt.xticks(fontsize=ft)
plt.xscale('log')
plt.subplots_adjust(left=0.2,right=0.95,bottom=0.2,top=0.98)
plt.savefig('reg.png',dpi=300)

plt.close()
t=[]
HUCB=[]
CON=[]
LIN=[]

with open('reg-reward.csv', 'r') as f:
	reader = csv.reader(f)
	for i in reader:
		if i[4]!='Time'and i[4]!='':
			# print(i[4])
			t.append(200*int(i[4]))
			LIN.append(float(i[5]))
			CON.append(float(i[6]))
			HUCB.append(float(i[7]))
ft = 18
# plot it!
plt.figure(figsize=(8,6))
plt.plot(t,CON, lw=2, label=r'ConUCB', color='blue',linestyle='-.')
plt.plot(t,HUCB, lw=2, label=r'Hier-LinUCB', color='red')
plt.plot(t,LIN, lw=2, label=r'LinUCB', color='green',linestyle='--')
plt.legend(loc='lower right',fontsize=ft)
plt.xlabel('Iteration',fontsize=ft)
plt.ylabel('Averaged Reward',fontsize=ft)
plt.xscale('log')
plt.yticks(fontsize=ft)
plt.xticks(fontsize=ft)
# plt.xscale('log')
plt.subplots_adjust(left=0.2,right=0.95,bottom=0.2,top=0.98)
plt.savefig('reward.png',dpi=300)