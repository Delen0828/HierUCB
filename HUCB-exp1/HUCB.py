import numpy as np
import math

class HUCBUserStruct:
	def __init__(self,dim,lambda_,init,sigma, alpha, norm=None):
		self.dim=dim
		self.lambda_=lambda_
		self.A=lambda_*np.identity(n=self.dim)
		self.Ainv=np.linalg.inv(self.A)
		self.b=np.zeros((self.dim,1))  
		if init!='zero':
			self.theta=np.random.rand(self.dim)
		else:
			self.theta=np.zeros(self.dim)
		self.time=1
		self.sigma=sigma
		self.alpha=alpha # store the new alpha calcuated in each iteratioin

		self.cal_alpha=False
		if self.alpha==-1:
			self.cal_alpha=True

		self.gtheta_norm=norm
		if self.cal_alpha:
			if self.gtheta_norm==None or self.alpha!=-1:
				raise AssertionError
			 #alpha=np.sqrt(self.dim *np.log((1+self.time*25/(self.lambda_*self.dim))/self.sigma))+np.sqrt(self.lambda_)*norm
			det_a=np.sqrt(np.linalg.det(self.A))
			self.alpha=(np.sqrt(2*np.log(det_a/(self.sigma*math.pow(self.lambda_,self.dim/2))))+np.sqrt(self.lambda_)*self.gtheta_norm)


	def getProb(self,fv):
		if self.alpha==-1:
			raise AssertionError
		mean=np.dot(self.theta.T,fv)
		var=np.sqrt(np.dot(np.dot(fv.T,self.Ainv),fv))
		pta=mean+self.alpha*var
		return pta, mean, self.alpha, var


	def getInv(self, old_Minv, nfv):
		# new_M=old_M+nfv*nfv'
		# try to get the inverse of new_M
		tmp_a=np.dot(np.outer(np.dot(old_Minv,nfv),nfv),old_Minv)
		tmp_b=1+np.dot(np.dot(nfv.T,old_Minv),nfv)
		new_Minv=old_Minv-tmp_a/tmp_b
		return new_Minv

		 
	def updateParameters(self, a_fv, reward ):
		self.A+=np.outer(a_fv, a_fv)
		self.b+=a_fv*reward
		#self.Ainv=np.linalg.inv(self.A)
		self.Ainv=self.getInv(self.Ainv,a_fv)
		self.theta=np.dot(self.Ainv, self.b)
		self.time+=1
		# calculate new alpha
		if self.cal_alpha:
			#alpha=np.sqrt(self.dim *np.log((1+self.time*25/(self.lambda_*self.dim))/self.sigma))+np.sqrt(self.lambda_)*norm
			det_a=np.sqrt(np.linalg.det(self.A))
			self.alpha=(np.sqrt(2*np.log(det_a/(self.sigma*math.pow(self.lambda_,self.dim/2))))+np.sqrt(self.lambda_)*self.gtheta_norm)

		
	  

class HUCB:
	def __init__(self,dim,para,arms,suparms,init='zero',bt=None):
		self.dim=dim
		self.lambda_=para['lambda']
		self.init=init
		self.users={}
		self.sigma=para['sigma']
		try:
			self.alpha=para['alpha']
		except:
			self.alpha=-1
		self.bt=bt
		self.supcount={}
		self.count={}
		self.need_switch=False
		for aid,suparm in suparms.items():
			self.supcount[suparm.id]=0
		for aid,arm in arms.items():
			self.count[arm.id]=0


	def get_budget(self,uid,norm):
		try:
			tmp=self.users[uid]
		except:
			self.users[uid]=HUCBUserStruct(self.dim,self.lambda_,self.init,self.sigma,self.alpha, norm)
		left_budget=self.bt(self.users[uid].time)- self.bt(self.users[uid].time-1)
		if left_budget>0:
			return int(left_budget)
		else:
			return -1


	def decide(self,armpool,suparm,uid, norm,debug_fw=None, best_arm=None):
		try:
			tmp=self.users[uid]
		except:
			self.users[uid]=HUCBUserStruct(self.dim,self.lambda_,self.init,self.sigma,self.alpha, norm)
		article_picked=None
		max_P=float('-inf')
		pool_arms=suparm.related_arms
		for index in pool_arms:
			x_pta, x_mean, x_alpha, x_var=self.users[uid].getProb(armpool[index].fv)
			if x_pta>max_P:
				article_picked=armpool[index]
				max_P=x_pta
		if article_picked ==None:
			raise AssertionError
		self.count[article_picked.id]+=1
		return article_picked

	def decide_suparms(self,pool_suparms,uid, norm,debug_fw=None, best_arm=None):
		try:
			tmp=self.users[uid]
		except:
			self.users[uid]=HUCBUserStruct(self.dim,self.lambda_,self.init,self.sigma,self.alpha, norm)
		suparm_picked=None
		max_P=float('-inf')

		for x,x_o in pool_suparms.items():
			x_pta, x_mean, x_alpha, x_var=self.users[uid].getProb(x_o.fv[uid])
			if x_pta>max_P:
				suparm_picked=x_o
				max_P=x_pta
		if suparm_picked ==None:
			raise AssertionError
		self.supcount[suparm_picked.id]+=1
		return suparm_picked

	def updateParameters(self, picked_arm, reward, uid):
		self.users[uid].updateParameters(picked_arm.fv,reward)

	def updateSupParameters(self,picked_suparm,reward,uid):
		self.users[uid].updateParameters(picked_suparm.fv[uid],reward)

	def getTheta(self, uid):
		return self.users[uid].theta

	def update_switch(self,uid,it,suparm,arm,k,discount):
		rho_sup=np.sqrt(np.log(it+1)/self.supcount[suparm.id])
		rho_arm=np.sqrt(np.log(it+1)/self.count[arm.id])
		s_pta, s_mean, s_alpha, s_var=self.users[uid].getProb(suparm.fv[uid])
		a_pta, a_mean, a_alpha, a_var=self.users[uid].getProb(arm.fv)
		s_pta=s_pta*discount
		# print(s_pta,rho_sup ,a_pta,rho_arm)
		print('Differ:',s_pta+rho_sup*k , a_pta-rho_arm*k)
		if s_pta+rho_sup*k < a_pta-rho_arm*k and it>9:
			self.need_switch=True
		else:
			self.need_switch=False

	def need_switch(self):
		return self.need_switch