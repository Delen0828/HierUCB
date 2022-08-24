IN_FOLDER="data/movielens/movielens"
OUT_FOLDER="result"
mkdir -p ${OUT_FOLDER}

python -W ignore run_exp_seq_bernoulli.py --num_repeat 5 --in_folder ${IN_FOLDER} --out_folder ${OUT_FOLDER} --arm_pool_size 2000 --iter 1000 --user 100