IN_FOLDER="data/synthetic/synthetic"
OUT_FOLDER="result"
mkdir -p ${OUT_FOLDER}

python -W ignore run_exp_seq_bernoulli.py --num_repeat 3 --in_folder ${IN_FOLDER} --out_folder ${OUT_FOLDER} --arm_pool_size 5000 --iter 500 --user 10
