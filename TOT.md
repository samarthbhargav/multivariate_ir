
srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1 

#### TAS-B 

srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1  sh eval_tot.sh sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco \
               /projects/0/prjs0907/multivariate_ir_experiments/experiments/tas_b_zeroshot \
                "" \
               /projects/0/prjs0907/multivariate_ir_experiments/experiments/tas_b_zeroshot/eval.log


#### DPR 


srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1  sh eval_tot.sh /projects/0/prjs0907/multivariate_ir_experiments/experiments/dpr_hs_db/3 \
               /projects/0/prjs0907/multivariate_ir_experiments/experiments/dpr_hs_db/3 \
                "" \
               /projects/0/prjs0907/multivariate_ir_experiments/experiments/dpr_hs_db/3/eval_log.log
               
 
#### CLDRD

srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1  sh eval_tot.sh /projects/0/prjs0907/multivariate_ir_experiments/experiments/cldrd \
               /projects/0/prjs0907/multivariate_ir_experiments/experiments/cldrd \
                "" \
               /projects/0/prjs0907/multivariate_ir_experiments/experiments/cldrd/eval_log.log


#### MVRL

srun -p gpu --time=96:00:00 --mem=120G -c12 --gres=gpu:1  sh eval_tot.sh /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl \
               /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl/updated \
                "--model_type mvrl --add_var_token  --embed_formulation updated --var_activation softplus --var_activation_param_b 2.5" \
               /projects/0/prjs0907/multivariate_ir_experiments/experiments/mvrl/updated/eval_log.log





