# 定义参数的选区范围
datasets=("yelp")  # 示例数据集选区 "amazon-kindle" "iFashion" "douban"
ssl_reg_uis=(0.01 0.1 0.2 0.3 0.4 0.5) # 0.01 0.1 0.2 0.3 0.4 0.5
lambda_1s=(0.8) # 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1
scales=(0.2)
time_steps=(80) # 10,20,30,40,50,60,70,80,90,100
elbos=(0.2) # 0.01 0.1 0.2 0.3 0.4 0.5
epochs=(10)
gcn_layer0s=(2)
regs=(0.001)
commands=()
# 迭代执行命令
for dataset in "${datasets[@]}"; do
    for ssl_reg_ui in "${ssl_reg_uis[@]}"; do
        for lambda_1 in "${lambda_1s[@]}"; do
            for time_step in "${time_steps[@]}"; do
                for elbo in "${elbos[@]}"; do
                    for gcn_layer0 in "${gcn_layer0s[@]}"; do
                        for reg in "${regs[@]}"; do
                            for epoch in "${epochs[@]}"; do
                                # 构造日志文件名
                                log_file="Result/${dataset}/_ssl_reg_ui_${ssl_reg_ui}_lambda_1_${lambda_1}_time_step_${time_step}_elbo_${elbo}_gcn_layer0_${gcn_layer0}_reg_${reg}_alpha_0_epoch_${epoch}.txt"
                                # 执行 Python 脚本并将输出重定向到日志文件
                                cmd="python3 Main.py --data "$dataset" --ssl_reg_ui "$ssl_reg_ui" --lambda_1 "$lambda_1" --time_step "$time_step" --elbo "$elbo" --gcn_layer0 "$gcn_layer0" --reg "$reg" --alpha 0 --epoch "$epoch" > $log_file"
                                commands+=("$cmd")   
                                echo "Executed:  $cmd"     
                            done
                        done             
                    done
                done
            done
        done
    done
done
printf "%s\n" "${commands[@]}" | xargs -P 4 -I {} sh -c "{}"
echo "All tasks completed."