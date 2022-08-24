# from pickle import TRUE
import pandas as pd

user_dict={}
with pd.read_json('yelp_academic_dataset_review.json', lines=True, chunksize=10000) as reader:
	# reader
	for chunk in reader:
		# print(chunk)
		templist=pd.Index.to_list(chunk['user_id'])
		for i in range(len(templist)):
			if templist[i] in user_dict:
				user_dict[templist[i]]+=1
			else:
				user_dict[templist[i]]=0

f=open('user.txt','w')
for uid,rev in user_dict.items():
	if rev>999:
		f.write(str(uid)+' , '+str(rev)+'\n')
		# print(uid,rev)
f.close()
# print(user_dict)
# dataset=pd.read_json('yelp_academic_dataset_user.json',lines=True)
# print(type(dataset))
# print(dataset)