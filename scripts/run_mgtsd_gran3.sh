export model_name='mgtsd'
# export dataset="elec"
export dataset="solar"
# export dataset="cup"
# export dataset="taxi"
# export dataset="traf"
# export dataset="wiki"

export batch_size=128
export num_cells=128
export diff_steps=100

export cuda_num=0
export epoch=30
export mg_dict='1_4_12'
export num_gran=3
export share_ratio_list='1_0.6_0.6'
export weight_list='0.8_0.1_0.1'



if [ ! -d "./result" ]; then
    mkdir "./result"
fi
if [ ! -d "./log" ]; then
    mkdir "./log"
fi

export result_path="./result/${model_name}_${dataset}"
export log_path="./log/${model_name}_${dataset}"
if [ ! -d $result_path ]; then
    mkdir $result_path
fi
if [ ! -d $log_path ]; then
    mkdir $log_path
fi

for i in {1..1};
do
    echo "run $i"
    python src/run_mgtsd.py \
    --result_path $result_path \
    --model_name $model_name \
    --epoch $epoch \
    --cuda_num $cuda_num \
    --dataset $dataset \
    --diff_steps $diff_steps\
    --batch_size $batch_size\
    --num_cells $num_cells\
    --mg_dict $mg_dict\
    --num_gran $num_gran\
    --share_ratio_list $share_ratio_list\
    --weight_list $weight_list\
    --run_num $i\
    --log_metrics False \
    >>"${log_path}/gran_${num_gran}_share_${share_ratio_list}_weight_${weight_list}_run_${i}.txt" 2>&1
done
