import numpy as np


class SupArm():
	def __init__(self, suparm_id, fv, related_arms):
		self.id = suparm_id
		self.fv = fv
		self.related_arms = related_arms


class SupArmManager:
	def __init__(self):
		self.suparms = {}
		self.num_suparm = 0

	def load(self, suparm_dict_user,suparm_to_arms):
		self.suparms = {}
		for uid, suparm_dict in suparm_dict_user.items():
			for suparm_id, suparm_feat in suparm_dict.items():
				self.suparms[suparm_id] = SupArm(
					suparm_id, suparm_feat,suparm_to_arms[suparm_id])

		self.num_suparm = len(self.suparms)
		self.tilde_X={}
		for uid, suparm_dict in suparm_dict_user.items():
			# print(uid)
			self.tilde_X[uid] = np.vstack(
				[self.suparms[suparm_idx].fv.T for suparm_idx in range(self.num_suparm)])