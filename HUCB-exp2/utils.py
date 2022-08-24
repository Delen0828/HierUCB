import collections
import math
import os

from torch import zeros_like

import numpy as np
import torch

from env import ArmManager, SupArmManager, UserManager

BUDGET_FUNCTION = lambda t: 5 * int(math.log(t + 1))
DEVICE = torch.device("cuda")

def prepare_data(in_folder):
	# Load data.
	arm_to_suparms_filename = os.path.join(in_folder, "arm_to_suparms.npy")
	arm_feats_filename = os.path.join(in_folder, "arm_feats.npy")
	affinity_filename = os.path.join(in_folder, "affinity.npy")
	arm_to_suparms = np.load(arm_to_suparms_filename, allow_pickle=True).item()
	arm_feats = np.load(arm_feats_filename, allow_pickle=True).item()
	arm_affinity_matrix = np.load(affinity_filename)
	#generate user feature
	user_feats={}
	# sup_feat_list=[]
	# print(arm_feats)
	for uid in range(200):
		user_feats[uid]={}
		for arm_id, related_suparms in arm_to_suparms.items():
			fv=arm_affinity_matrix[uid][arm_id]/arm_feats[arm_id]
			# print(fv)
			user_feats[uid][arm_id]=fv
	# print(user_feats)
	# Construct key-term features.
	suparm_to_arms={}
	for arm_id, related_suparms in arm_to_suparms.items():
		for sup_id in related_suparms:
			if sup_id in suparm_to_arms.keys():
				suparm_to_arms[sup_id].append(arm_id)
			else:
				suparm_to_arms[sup_id]=[]
	suparm_feats={}
	for uid in range(200):
		suparm_feats[uid] = {}
		suparm_weights = {}
		for suparm_idx, related_arm_idxs in suparm_to_arms.items():
			maxf=float('-inf')
			best=0
			for arm_id in related_arm_idxs:
				# print(user_feats[uid][arm_id])
				# print(arm_feats[arm_id])
				if maxf<user_feats[uid][arm_id]@arm_feats[arm_id]:
					maxf=user_feats[uid][arm_id]@arm_feats[arm_id]
					best=arm_id
			suparm_feats[uid][suparm_idx] =arm_feats[best] ## get best
			suparm_weights[suparm_idx]={}
			for arm_id in related_arm_idxs:
				suparm_weights[suparm_idx][arm_id]=0
			suparm_weights[suparm_idx][best]=1
    # for suparm_idx in suparm_feats:
    #     suparm_feats[suparm_idx] /= np.sum(suparm_weights[suparm_idx])

	# Load arms.
	arm_manager = ArmManager()
	arm_manager.load_from_dict(arm_feats)
	X = arm_manager.X
	
	print("Finish loading arms: {}".format(arm_manager.n_arms))

	# Load suparms.
	super_arm_manager = SupArmManager()
	super_arm_manager.load(suparm_feats,suparm_to_arms)
	tilde_X = super_arm_manager.tilde_X


	print("Finish loading suparms: {}".format(super_arm_manager.num_suparm))

	return X, tilde_X, arm_affinity_matrix, arm_to_suparms, suparm_to_arms, 	arm_manager, super_arm_manager
