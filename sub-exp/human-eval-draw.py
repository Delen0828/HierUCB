from difflib import context_diff
from tkinter import font
import numpy as np
# import matplotlib as plt
from matplotlib import pyplot as plt
import csv 

# #Hotpot
# 3.490909091	ATTR
# 3.090489707	AVG
# 3.363534415	TOP(>3)

# #Western Food
# 3.654545455	ATTR
# 3.202095761	AVG
# 3.490241412	TOP(>3)

# #Coffee
# 3.563636364	ATTR
# 3.172822105	AVG
# 3.541079933	TOP(>3)

hotpot={'attr':3.490909091,'avg':3.090489707, 'top': 3.363534415}
west={'attr':3.654545455,'avg':3.202095761, 'top': 3.490241412}
coffee={'attr':3.563636364,'avg':3.172822105, 'top':3.541079933}
attr=[3.490909091,3.654545455,3.563636364]
avg=[3.090489707,3.202095761, 3.172822105]
top=[3.363534415,3.490241412,3.541079933]
size = 3
x = np.arange(size)
total_width, n = 0.6, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.figure(figsize=(9,7))
plt.bar(x, attr,  width=width,label='True Rating',color='grey',edgecolor='black')
plt.bar(x + width, top, width=width,tick_label=['Hotpot','Fine Dining','Coffee'], label='Top 20%',color='orange',hatch='//',edgecolor='black')
plt.bar(x + 2 * width, avg, width=width, label='Average',color='lightgrey',hatch='..',edgecolor='black')
plt.ylim(bottom=3,top=4)
plt.yticks([3,3.5,4],fontsize=16)
plt.ylabel('Rating',fontsize=18)
plt.xticks(fontsize=18)
# plt.xlabel('Categories')
# plt.xticks()
plt.legend(loc="upper left",prop={'size':16})
plt.savefig('human-eval.png',dpi=300)