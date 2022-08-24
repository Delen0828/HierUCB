import math
linucb_para={'lambda':1,'sigma':0.1, 'alpha': 1} 
conucb_para={'lambda':0.5, 'sigma':0.1, 'tilde_lambda':1, 'alpha':1, 'tilde_alpha': 1}
hucb_para={'lambda':0.5,'sigma':0.1, 'alpha': 1,'k':1} 
train_iter=0
test_iter=1000
armNoiseScale=0.1
suparmNoiseScale=0.05
batch_size=10
switch=0.8 #HUCB switch condition
discount=0.5 #suparm-arm reward discount
bt= lambda t: 5*int(math.log(t+1))
seeds_set=[2756048, 675510, 807110,2165051, 9492253, 927,218,495,515,452]
