import os
import re

def extract_max_metrics_from_logs(folder_path):
    """
    从指定文件夹的日志文件中提取每个文件的最大Recall10/Recall20，以及对应行的NDCG10/NDCG20
    
    参数:
        folder_path: 日志文件所在的文件夹路径
    返回:
        字典，key为文件名，value为各指标结果的字典
    """
    # 定义正则表达式，匹配日志行中的四个指标数值
    pattern = re.compile(
        r'Recall10 = (\d+\.\d+), NDCG10 = (\d+\.\d+), Recall20 = (\d+\.\d+), NDCG20 = (\d+\.\d+)'
    )
    
    # 存储每个文件的结果（Recall最大值 + 对应NDCG）
    file_metrics = {}
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 跳过子文件夹，只处理文件
        if not os.path.isfile(file_path):
            continue
        
        # 初始化：Recall最大值为0，对应NDCG初始为0
        max_recall10 = 0.0
        ndcg10_at_max_recall10 = 0.0  # 最大Recall10对应的NDCG10
        max_recall20 = 0.0
        ndcg20_at_max_recall20 = 0.0  # 最大Recall20对应的NDCG20
        
        try:
            # 读取文件，处理常见编码（utf-8/gbk）
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"⚠️ 读取文件 {filename} 失败：{e}，已跳过该文件")
                continue
        except Exception as e:
            print(f"⚠️ 读取文件 {filename} 失败：{e}，已跳过该文件")
            continue
        
        # 逐行解析日志，更新Recall最大值及对应NDCG
        for line in lines:
            match = pattern.search(line)
            if match:
                # 提取匹配到的数值并转换为浮点数
                recall10 = float(match.group(1))
                ndcg10 = float(match.group(2))
                recall20 = float(match.group(3))
                ndcg20 = float(match.group(4))
                
                # 当Recall10更大时，同步更新Recall10最大值和对应NDCG10
                if recall10 > max_recall10:
                    max_recall10 = recall10
                    ndcg10_at_max_recall10 = ndcg10
                
                # 当Recall20更大时，同步更新Recall20最大值和对应NDCG20
                if recall20 > max_recall20:
                    max_recall20 = recall20
                    ndcg20_at_max_recall20 = ndcg20
        
        # 存储当前文件的结果（仅当至少匹配到一组数值时）
        if max_recall10 > 0 or max_recall20 > 0:
            file_metrics[filename] = {
                'Recall10': max_recall10,
                'NDCG10(对应Recall10最大值)': ndcg10_at_max_recall10,
                'Recall20': max_recall20,
                'NDCG20(对应Recall20最大值)': ndcg20_at_max_recall20
            }
        else:
            print(f"ℹ️ 文件 {filename} 中未找到有效指标数据")
    
    return file_metrics

def print_metrics_result(file_metrics):
    """格式化输出各文件的结果（Recall最大值 + 对应NDCG）"""
    if not file_metrics:
        print("❌ 未找到任何包含有效指标的日志文件")
        return
    
    # 打印表头（标注NDCG是对应Recall最大值的数值）
    print(f"{'文件名':<30} {'Recall10':<10} {'NDCG10(对应Recall10)':<20} {'Recall20':<10} {'NDCG20(对应Recall20)':<20}")
    print("-" * 100)
    
    # 打印每个文件的结果
    for filename, metrics in file_metrics.items():
        print(
            f"{filename:<30} {metrics['Recall10']:<10.6f} {metrics['NDCG10(对应Recall10最大值)']:<20.6f} "
            f"{metrics['Recall20']:<10.6f} {metrics['NDCG20(对应Recall20最大值)']:<20.6f}"
        )

if __name__ == "__main__":
    # 请修改为你的日志文件夹路径（绝对路径/相对路径均可）
    LOG_FOLDER_PATH = "./Result/yelp"  # 示例：当前目录下的log_files文件夹
    
    # 检查文件夹是否存在
    if not os.path.exists(LOG_FOLDER_PATH):
        print(f"❌ 文件夹 {LOG_FOLDER_PATH} 不存在，请检查路径是否正确")
    else:
        # 提取指标并输出
        metrics_result = extract_max_metrics_from_logs(LOG_FOLDER_PATH)
        print_metrics_result(metrics_result)
"""
douban-book
# lambda_3
_ssl_reg_ui_0.01_lambda_1_0.9_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133200   0.155200             0.187300   0.164100            
_ssl_reg_ui_0.1_lambda_1_0.9_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133400   0.154900             0.187800   0.164200            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133700   0.134600   0.155600             0.189500   0.165300 
_ssl_reg_ui_0.3_lambda_1_0.9_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133800   0.155000             0.188400   0.164900   
_ssl_reg_ui_0.4_lambda_1_0.9_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.131800   0.153200             0.186500   0.162900            
_ssl_reg_ui_0.5_lambda_1_0.9_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.131900   0.152700             0.187000   0.162500            

# lambda_1
_ssl_reg_ui_0.2_lambda_1_0_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_20.txt 0.134400   0.155400             0.188400   0.164800
_ssl_reg_ui_0.2_lambda_1_0.1_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134400   0.155400             0.188200   0.164700            
_ssl_reg_ui_0.2_lambda_1_0.2_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134400   0.155300             0.187800   0.164500 
_ssl_reg_ui_0.2_lambda_1_0.3_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134300   0.155500             0.188100   0.164700            
_ssl_reg_ui_0.2_lambda_1_0.4_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134100   0.155700             0.188100   0.164900            
_ssl_reg_ui_0.2_lambda_1_0.5_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133900   0.155900             0.188300   0.165100            
_ssl_reg_ui_0.2_lambda_1_0.6_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133800   0.155600             0.188600   0.165300            
_ssl_reg_ui_0.2_lambda_1_0.7_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134000   0.155800             0.188200   0.165200            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134000   0.156200             0.188700   0.164800            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133700   0.134600   0.155600             0.189500   0.165300 
_ssl_reg_ui_0.2_lambda_1_1_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_20.txt 0.134100   0.156300             0.188300   0.165500

# time T
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_10_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.117300   0.139000             0.164200   0.146300            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_20_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.131400   0.153700             0.184000   0.162600            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_30_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134300   0.155000             0.185700   0.164400            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_40_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134100   0.156200             0.188100   0.165500            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_50_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133700   0.156600             0.187800   0.165400            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_60_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133300   0.155600             0.187700   0.164600            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_70_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134600   0.155600             0.189500   0.165300            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133700   0.156000             0.188300   0.165400            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_90_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133100   0.154500             0.188700   0.164300            
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_100_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133200   0.154600             0.187900   0.163700            

# lambda_2
ssl_reg_uu_ii_0.2_ssl_reg_ui_0.2_lambda_1_0.9_time_step_80_elbo_0.01_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.131300   0.152100             0.186800   0.162000            
ssl_reg_uu_ii_0.2_ssl_reg_ui_0.2_lambda_1_0.9_time_step_80_elbo_0.1_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133000   0.153800             0.187000   0.162700            
ssl_reg_uu_ii_0.2_ssl_reg_ui_0.2_lambda_1_0.9_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134200   0.156300             0.188400   0.165500            
ssl_reg_uu_ii_0.2_ssl_reg_ui_0.2_lambda_1_0.9_time_step_80_elbo_0.3_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134500   0.156500             0.188200   0.165400            
ssl_reg_uu_ii_0.2_ssl_reg_ui_0.2_lambda_1_0.9_time_step_80_elbo_0.4_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.133500   0.155800             0.188500   0.165800            
ssl_reg_uu_ii_0.2_ssl_reg_ui_0.2_lambda_1_0.9_time_step_80_elbo_0.5_gcn_layer0_2_reg_0.001_alpha_0_epoch_100.txt 0.134000   0.157000             0.187700   0.165900            


yelp
# lambda_3
_ssl_reg_ui_0.01_lambda_1_0.8_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061200   0.036800             0.101200   0.048900            
_ssl_reg_ui_0.1_lambda_1_0.8_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061900   0.037400             0.101400   0.049300            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.0623	0.0381	0.1022	0.0494           
_ssl_reg_ui_0.3_lambda_1_0.8_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061200   0.037000             0.101100   0.048900            
_ssl_reg_ui_0.4_lambda_1_0.8_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061100   0.036900             0.101100   0.048700            
_ssl_reg_ui_0.5_lambda_1_0.8_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061200   0.036800             0.100800   0.048700  

# lambda_1
_ssl_reg_ui_0.2_lambda_1_0_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061800   0.037100             0.101000   0.048900            
_ssl_reg_ui_0.2_lambda_1_0.1_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061700   0.037100             0.100700   0.048900            
_ssl_reg_ui_0.2_lambda_1_0.2_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061600   0.037100             0.100700   0.048900            
_ssl_reg_ui_0.2_lambda_1_0.3_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061200   0.037000             0.100500   0.048800            
_ssl_reg_ui_0.2_lambda_1_0.4_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061500   0.037000             0.100500   0.048700            
_ssl_reg_ui_0.2_lambda_1_0.5_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061400   0.037000             0.100900   0.048900            
_ssl_reg_ui_0.2_lambda_1_0.6_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061300   0.037000             0.100500   0.048800            
_ssl_reg_ui_0.2_lambda_1_0.7_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061300   0.037000             0.100600   0.048800   
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.0623	0.0381	0.1022	0.0494          
_ssl_reg_ui_0.2_lambda_1_0.9_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061700   0.037100             0.100700   0.048900            
_ssl_reg_ui_0.2_lambda_1_1_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061400   0.037100             0.100500   0.048900    

# lambda_2
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_80_elbo_0.01_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061200   0.037000             0.102100   0.049300            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_80_elbo_0.1_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061300   0.037000             0.101600   0.049200     
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.0623	0.0381	0.1022	0.0494      
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_80_elbo_0.3_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.060000   0.036500             0.100000   0.048600            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_80_elbo_0.4_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.059900   0.036200             0.099100   0.047900            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_80_elbo_0.5_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.059500   0.035900             0.097900   0.047500  

# step T
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_10_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.050100   0.030500             0.083200   0.040500            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_20_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.057100   0.034400             0.093700   0.045500            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_30_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.058300   0.035300             0.097200   0.047100            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_40_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.059600   0.036000             0.098300   0.047700            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_50_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.059500   0.036000             0.099400   0.048000            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_60_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.060800   0.036700             0.100500   0.048600            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_70_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061100   0.036900             0.100400   0.048900     
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_80_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.0623	0.0381	0.1022	0.0494        
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_90_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.061500   0.037000             0.100700   0.048800            
_ssl_reg_ui_0.2_lambda_1_0.8_time_step_100_elbo_0.2_gcn_layer0_2_reg_0.001_alpha_0_epoch_10.txt 0.060900   0.036800             0.100900   0.049100            
w_o_diff.txt                   0.061000   0.036900             0.102300   0.049300            
w_o_AM.txt                     0.062200   0.037200             0.103000   0.049600            
w_o_dir.txt                    0.061200   0.037000             0.102100   0.049300            
w_diff_infer.txt               0.060200   0.036000             0.099900   0.048000            
w_o_uni.txt                    0.037700   0.022900             0.063800   0.030600           
"""