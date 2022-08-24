# from pickle import TRUE
from cProfile import label
from tkinter import CENTER
from warnings import catch_warnings
from numpy import arange, array, bincount, busday_count, true_divide
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def is_food(word):
	if 'Food' in word:
		return True
	elif 'Restaurants' in word:
		return True
	else:
		return False

def trans_food(food):
	if food == 'Chinese, Restaurants':
		return 'Chinese'
	elif food == 'Restaurants, Italian':
		return 'Italian'
	elif food == 'Coffee & Tea, Food':
		return 'Coffee'
	elif food == 'Beer, Wine & Spirits, Food':
		return 'Bar'
	elif food == 'Restaurants, Mexican':
		return 'Mexican'
	elif food == 'Thai, Restaurants':
		return 'Thai'
	elif food == 'Pizza, Restaurants':
		return 'Pizza'
	else:
		return 'NULL'

def filter_food(food):
	if food == 'Restaurants, Chinese' or food =='Food, Ice Cream & Frozen Yogurt' or food=='Food, Coffee & Tea' or food=='Ice Cream & Frozen Yogurt, Food' or food =='Italian, Restaurants':
		return False
	else:
		return True

user_dict = {}
for line in open('user.txt', 'r'):
	# print(line, end='')
	# line.replace('\n','').replace('\r','')
	temp = line.split(',')
	# temp[1]=int(temp[1])
	user_dict[temp[0]] = int(temp[1])
# print(user_dict)
user_review = {}

business_dict = {}
business_star = {}
business_count = {}
count_sum = 0
with pd.read_json('yelp_academic_dataset_business.json', lines=True, chunksize=1000) as reader:
	# reader
	for chunk in tqdm(reader, desc="Load Business", total=161):
		# print(chunk)
		businesses = pd.Index.to_list(chunk['business_id'])
		tags = pd.Index.to_list(chunk['categories'])
		stars = pd.Index.to_list(chunk['stars'])
		count = pd.Index.to_list(chunk['review_count'])
		count_sum += sum(count)
		for i in range(len(businesses)):
			if businesses[i] not in user_dict:
				business_dict[businesses[i]] = []
			business_dict[businesses[i]].append(tags[i])
			business_star[businesses[i]] = stars[i]
			business_count[businesses[i]] = count[i]

truth_list = []
avg_list = []
top_list = []
namelist=[]
for i in tqdm(range(len(tags)),desc="Rating Calc"):
	rate_sum = wei_rate_sum =top_rate_sum= count_sum = counting = 0
	rate_data=[]
	for bus_id, tag_list in business_dict.items():
		if tags[i] in tag_list:
			rate_data.append(business_star[bus_id])
			rate_sum += business_star[bus_id]
			wei_rate_sum += business_star[bus_id]*business_count[bus_id]
			count_sum += business_count[bus_id]
			counting += 1
	rate_data =sorted(rate_data,reverse=True)
	if len(rate_data) > 0:
		# print(rate_data)
		for data in rate_data:
			if data<3:
				rate_data.remove(data)
		# print(rate_data)
	if len(rate_data)>0:
		top_rate_sum= sum(rate_data)/len(rate_data)
	if len(rate_data) > 0 and counting > 200:
		if is_food(tags[i]) and (trans_food(tags[i]) not in namelist) and filter_food(tags[i]):
			truth_list.append(wei_rate_sum/count_sum)
			avg_list.append(rate_sum/counting)
			top_list.append(top_rate_sum)
			namelist.append(trans_food(tags[i]))
print('Ground Truth:',truth_list)
print('Average:',avg_list)
print('Top:',top_list)
print(namelist)
print(len(truth_list),len(namelist))
truth = np.array(truth_list)
avg = np.array(avg_list)
top=np.array(top_list)



x=arange(len(truth))
plt.figure(figsize=(10,6))
plt.bar(x,truth,label='Weighted Average',width=0.25,color='grey',edgecolor='black')
plt.bar(x+0.25,top,label='Top 50%',width=0.25,color='orange',hatch='//',edgecolor='black')
plt.bar(x+0.5,avg,label='Average',width=0.25,color='lightgrey',hatch='..',edgecolor='black')
plt.ylim(bottom=3,top=4)
plt.yticks([3,3.5,4],fontsize=18)
plt.xticks(x+0.25,namelist,fontsize=18)
plt.ylabel('Rating',fontsize=18)
plt.legend(prop={'size':16},loc='upper left')
plt.subplots_adjust(left=0.1,right=0.97,top=0.97,bottom=0.07)
# plt.subplots_adjust(left=0.25,right=0.9,bottom=0.1,top=0.95)
plt.savefig('bar-h.jpg',dpi=300)