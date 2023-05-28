python train.py --run_name noher_dense
python train.py --run_name her_dense --use_her
python train.py --run_name noher_sparse --sparse_reward
python train.py --run_name her_sparse --use_her --sparse_reward
python train.py --run_name her_sparse_tau_0.0001 --use_her --sparse_reward --soft_tau 0.0001
python train.py --run_name her_sparse_tau_0.01 --use_her --sparse_reward --soft_tau 0.01
python train.py --run_name her_sparse_tau_0.1 --use_her --sparse_reward --soft_tau 0.1
python train.py --run_name her_sparse_hard_update --use_her --sparse_reward --soft_tau 1
python train.py --run_name her_sparse_OUnoise --use_her --sparse_reward