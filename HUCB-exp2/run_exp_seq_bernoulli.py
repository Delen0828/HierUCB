import argparse
import os
import random
from algorithm import ConUCB, RelativeConUCB, LinUCB, HUCB
from env import BernoulliEnv
import utils

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Main experiment.")
	parser.add_argument("--in_folder", dest="in_folder",
						help="input the folder containing input files")
	parser.add_argument("--out_folder", dest="out_folder",
						help="input the folder to output")
	parser.add_argument("--arm_pool_size", dest="arm_pool_size",
						type=int, help="pool_size of each iteration")
	parser.add_argument("--num_repeat", dest="num_repeat",
						type=int, help="# of repeat")
	parser.add_argument("--iter", dest="iter", type=int,
						help="each user iteration time")
	parser.add_argument("--user", dest="user", type=int, help="how many users")
	args = parser.parse_args()

	# pre generate user
	#################################randomnize#####################################
	user_list = []
	for j in range(args.iter):
		for i in range(0, args.user):
			user_list.append(i)
	# random.shuffle(user_list)
	# print(user_list)
	# Preprocess.
	datasetname = args.in_folder.strip("/").split("/")[-1]
	print("Using dataset {}".format(datasetname))
	print("Results will be save at '{}'".format(args.out_folder))
	X, tilde_X, arm_affinity_matrix, arm_to_suparms, suparm_to_arms, arms, suparms = utils.prepare_data(
		args.in_folder)
	assert X.shape[1] == tilde_X[0].shape[1]
	dim = X.shape[1]
	num_user = len(arm_affinity_matrix)

	# Initialize the experiment.
	assert os.path.exists(args.out_folder)
	simulate_exp = BernoulliEnv(
		X,
		tilde_X,
		arm_affinity_matrix,
		arm_to_suparms,
		suparm_to_arms,
		arms=arms,
		suparms=suparms,
		out_folder=os.path.join(args.out_folder, datasetname),
		device=utils.DEVICE,
		arm_pool_size=5000,
		relative_noise=0.1,
		budget_func=utils.BUDGET_FUNCTION,
		is_early_register=False,
		num_iter=args.user * args.iter
	)

	algorithms = {
		"LinUCB": LinUCB(dim, utils.DEVICE),
		"ConUCB": ConUCB(dim, utils.DEVICE),
		# "ConUCB_share-attribute": ConUCB(dim, utils.DEVICE, is_update_all_attribute=True),
		"Hier-LinUCB": HUCB(args.user,arms,suparms,dim,utils.DEVICE)
	}

	if args.user == 200:
		simulate_exp.run_algorithms(algorithms, num_repeat=args.num_repeat)
	else:
		simulate_exp.run_algorithms(
			algorithms, defined_user_sequence=user_list, num_repeat=args.num_repeat)
	print("Results was saved at '{}'".format(args.out_folder))
