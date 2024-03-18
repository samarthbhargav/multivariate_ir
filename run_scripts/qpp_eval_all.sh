ROOT_PATH=/scratch-shared/sbhargav/multivariate_ir_experiments/experiments

QPP_METHODS=( "norm" "norm_recip" "det" "sum" )

for qpp_method in "${QPP_METHODS[@]}"
do
  echo "eval $qpp_method"

  sh qpp_eval_model.sh \
      ${ROOT_PATH}/mvrl/ \
      ${ROOT_PATH}/mvrl/qpp_${qpp_method} \
      "--model_type mvrl --add_var_token  --embed_formulation updated --var_activation softplus --var_activation_param_b 2.5 --qpp_method ${qpp_method}" \
      ${ROOT_PATH}/mvrl/qpp_${qpp_method}/qpp.log

  sh qpp_eval_model.sh \
      ${ROOT_PATH}/mvrl_nd_db/14 \
      ${ROOT_PATH}/mvrl_nd_db/14/qpp_${qpp_method} \
      "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation softplus --var_activation_param_b 2.5 --qpp_method ${qpp_method}" \
      ${ROOT_PATH}/mvrl_nd_db/14/qpp_${qpp_method}/qpp.log &


  sh qpp_eval_model.sh \
      ${ROOT_PATH}/mvrl_nd_db_logvar/3 \
      ${ROOT_PATH}/mvrl_nd_db_logvar/3/qpp_${qpp_method} \
      "--model_type mvrl_no_distill --add_var_token --var_activation logvar --embed_formulation updated --qpp_method ${qpp_method}" \
      ${ROOT_PATH}/mvrl_nd_db_logvar/3/qpp_${qpp_method}/qpp.log

  sh qpp_eval_model.sh \
        ${ROOT_PATH}/mvrl_nd_tasb_2 \
        ${ROOT_PATH}/mvrl_nd_tasb_2/qpp_${qpp_method} \
        "--model_type mvrl_no_distill --add_var_token  --embed_formulation original --var_activation softplus --var_activation_param_b 2.5 --qpp_method ${qpp_method}" \
        ${ROOT_PATH}/mvrl_nd_tasb_2/qpp_${qpp_method}/qpp.log

  sh qpp_eval_model.sh \
        ${ROOT_PATH}/mvrl_nd_tasb_logvar \
        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp_${qpp_method} \
        "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar --qpp_method ${qpp_method}" \
        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp_${qpp_method}/qpp.log


  # STOCH Models
  sh qpp_eval_model.sh \
        ${ROOT_PATH}/stoch_db \
        ${ROOT_PATH}/stoch_db/qpp_${qpp_method} \
        "--model_type stochastic --qpp_method ${qpp_method}" \
        ${ROOT_PATH}/stoch_db/qpp_${qpp_method}/qpp.log


  sh qpp_eval_model.sh \
        ${ROOT_PATH}/stoch_db_frozen \
        ${ROOT_PATH}/stoch_db_frozen/qpp_${qpp_method} \
        "--model_type stochastic --qpp_method ${qpp_method}" \
        ${ROOT_PATH}/stoch_db_frozen/qpp_${qpp_method}/qpp.log


  sh qpp_eval_model.sh \
        ${ROOT_PATH}/stoch_tasb_frozen \
        ${ROOT_PATH}/stoch_tasb_frozen/qpp_${qpp_method} \
        "--model_type stochastic --qpp_method ${qpp_method}" \
        ${ROOT_PATH}/stoch_tasb_frozen/qpp_${qpp_method}/qpp.log


  sh qpp_eval_model.sh \
        ${ROOT_PATH}/stoch_tasb \
        ${ROOT_PATH}/stoch_tasb/qpp_${qpp_method} \
        "--model_type stochastic --qpp_method ${qpp_method}" \
        ${ROOT_PATH}/stoch_tasb/qpp_${qpp_method}/qpp.log

done
#
## MVRL TASB Lovgar for diff checkpoints
# sh qpp_eval_model.sh \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/checkpoint-25000/ \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-25000 \
#        "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar --qpp_method ${qpp_method}" \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-25000/qpp.log  &
#
#
# sh qpp_eval_model.sh \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/checkpoint-50000/ \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-50000 \
#        "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar --qpp_method ${qpp_method}" \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-50000/qpp.log &
#
# sh qpp_eval_model.sh \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/checkpoint-75000/ \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-75000 \
#        "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar --qpp_method ${qpp_method}" \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-75000/qpp.log &
#
# sh qpp_eval_model.sh \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/checkpoint-100000/ \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-100000 \
#        "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar --qpp_method ${qpp_method}" \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-100000/qpp.log &
#
# sh qpp_eval_model.sh \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/checkpoint-125000/ \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-125000 \
#        "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar --qpp_method ${qpp_method}" \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-125000/qpp.log &
#
# sh qpp_eval_model.sh \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/checkpoint-150000/ \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-150000 \
#        "--model_type mvrl_no_distill --add_var_token  --embed_formulation updated --var_activation logvar --qpp_method ${qpp_method}" \
#        ${ROOT_PATH}/mvrl_nd_tasb_logvar/qpp-150000/qpp.log &