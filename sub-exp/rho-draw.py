# import imp
import numpy as np
# import matplotlib as plt
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon 
import csv 
t=[]
Con1=[]
Con2=[]
H1=[]
H2=[]
with open('Flexible-Rho.csv', 'r') as f:
	reader = csv.reader(f)
	for i in reader:
		if i[0]!='Time':
			t.append(int(i[0]))
			Con1.append(float(i[1]))
			H1.append(float(i[2]))
			Con2.append(float(i[3]))
			H2.append(float(i[4]))

ft = 14
# plot it!
fig, ax = plt.subplots(1)

ax.plot(t, Con1, lw=2, label=r'ConUCB (User 0)',color='navy')
ax.plot(t, Con2, lw=2, label=r'ConUCB (User 1)',color='navy',linestyle='--')
ax.plot(t, H1, lw=2, label=r'Hier-LinUCB (User 0)',color='orange')
ax.plot(t, H2, lw=2, label=r'Hier-LinUCB (User 1)',color='orange',linestyle='--')

k=50
ax.axvline(t[3],color='black',linestyle="-",ymax=H2[3]/350,ymin=50/350)
ax.axvline(t[9],color='black',linestyle="-",ymax=H1[9]/350,ymin=50/350)
# ax.axvline(t[5],color='black',linestyle=":",ymax=(Con2[5]+k)/350,ymin=(Con2[5])/350)
# ax.axvline(t[6],color='black',linestyle=":",ymax=(Con2[5]+k)/350,ymin=(Con2[5])/350)
# ax.axhline((Con2[5]+k)/350,color='black',linestyle=":",xmax=t[6],xmin=t[5])
# ax.axhline(Con2[5]/350,color='black',linestyle=":",xmax=t[6],xmin=t[5])
# ax.axvline(t[14],color='grey',linestyle=":",ymax=Con2[14]/350)
# ax.axvline(t[15],color='grey',linestyle=":",ymax=Con2[15]/350)
# ymax=(Con2[5]+k)/350
# ymin=(Con2[5])/350
# ax.add_patch(Polygon([[t[5], ymin], [t[6], ymin], [t[6], ymax], [t[5], ymax]], closed=True, fill=False, hatch="//"))

ax.text(t[2],15,'Switch Point',fontsize=10)
ax.text(t[8],15,'Switch Point',fontsize=10)
ax.text(t[2],30,'User 0',fontsize=10)
ax.text(t[8],30,'User 1',fontsize=10)


ax.legend(loc='upper left',fontsize=12)
ax.set_xlabel('Iteration',fontsize=ft)
ax.set_ylabel('Cumulative Regret',fontsize=ft)
ax.set_yticks(range(0,400,50))
fig.savefig('rho-switch.png',dpi=300)

