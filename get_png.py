import matplotlib.pyplot as plt
import numpy as np

# 你的原始数据（完全保留）
# #lambda_3
# noise_scale = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
# recall20_dict = [0.185, 0.186, 0.187, 0.188, 0.189, 0.190]
# ndcg20_dict = [0.162, 0.163, 0.164, 0.165, 0.166]
# recall20 = np.array([0.1873, 0.1878, 0.1895, 0.1884, 0.1865, 0.1870])
# ndcg20 = np.array([0.1641, 0.1642, 0.1653, 0.1649, 0.1629, 0.1625])
dataset = 'Douban-Book' # yelp
dataset = 'Yelp' # yelp
# # # #yelp lambda_3
noise_scale = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
recall20_dict = np.arange(99,104,1) / 1000
ndcg20_dict = np.arange(47,51,1) / 1000
recall20 = np.array([0.1012, 0.1014, 0.1022, 0.1011, 0.1011, 0.1008])
ndcg20 = np.array([0.0473,0.0479,0.0494,0.0489,0.0481,0.0477])

# #lambda_1
# noise_scale = np.array([0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1])
# recall20_dict = [0.185, 0.186, 0.187, 0.188, 0.189, 0.190]
# ndcg20_dict = [0.162, 0.163, 0.164, 0.165, 0.166]
# recall20 = np.array([0.188400, 0.1882, 0.1878, 0.1881, 0.1881, 0.1883, 0.1886,0.1882,0.1887,0.1895,0.1883])
# ndcg20 = np.array([0.1648,0.1647, 0.1645, 0.1647, 0.1649, 0.1651, 0.1653,0.1652,0.1648,0.1655,0.1632])

# # #yelp lambda_1
# noise_scale = np.array([0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1])
# recall20_dict = np.arange(99,104,1) / 1000
# ndcg20_dict = np.arange(47,51,1) / 1000
# recall20 = np.array([0.101,0.1007,0.1007,0.1017,0.1005,0.1005,0.1019,0.1016,0.1022,0.1010,0.1005])
# ndcg20 = np.array([0.0489,0.0489,0.0489,0.0492,0.0487,0.0489,0.0488,0.0488,0.0494,0.0489,0.0489])


# # #lambda_2
# noise_scale = np.array([0.01,0.1, 0.2, 0.3, 0.4, 0.5])
# recall20_dict = [0.185, 0.186, 0.187, 0.188, 0.189, 0.190]
# ndcg20_dict = [0.161 , 0.162, 0.163, 0.164, 0.165, 0.166]
# recall20 = np.array([0.1868, 0.1870, 0.1895, 0.1882, 0.1885, 0.1877])
# ndcg20 = np.array([0.1620,0.1627, 0.1653, 0.1640,  0.1641, 0.1639])

# # #yelp lambda_2
# noise_scale = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
# recall20_dict = np.arange(96,104,1) / 1000
# ndcg20_dict = np.arange(47,51,1) / 1000
# recall20 = np.array([0.1021,0.1016,0.1022,0.1000,0.0991,0.0979])
# ndcg20 = np.array([0.0493,0.0492,0.0494,0.0486,0.0479,0.0475])

# # #step T
# noise_scale = np.array([10,20,30,40,50,60,70,80,90,100])
# # recall20_dict = [0.164,0.184,0.185,0.186, 0.187, 0.188, 0.189, 0.190]
# recall20_dict = np.arange(160,194,2) / 1000
# ndcg20_dict = [0.161 , 0.162, 0.163, 0.164, 0.165, 0.166]
# recall20 = np.array([0.1642, 0.1840,0.1857, 0.1881, 0.1878, 0.1877, 0.1895,0.1883,0.1887,0.1879])
# ndcg20 = np.array([0.1643,0.1626,0.1644,0.1648,0.1647,0.1646,0.1653,0.1640,0.1643,0.1637])


# #yelp step T
# noise_scale = np.array([10,20,30,40,50,60,70,80,90,100])
# recall20_dict = np.arange(82,105,2) / 1000
# ndcg20_dict = np.arange(39,51,1) / 1000
# recall20 = np.array([0.0832,0.0937,0.0972,0.0983,0.0994,0.1005,0.1004,0.1022,0.1007,0.1009])
# ndcg20 = np.array([0.0405,0.0455,0.0471,0.0477,0.0480,0.0486,0.0489,0.0494,0.0488,0.0491])

# ========== 核心配置：解决刻度/布局问题 ==========
fig, ax1 = plt.subplots(figsize=(7, 4), dpi=150)  # 加宽画布，避免刻度重叠

# 绘制Recall@20（左Y轴）
color = 'tab:blue'
ax1.set_xlabel(r'$\lambda_{2}$', fontsize=12, fontweight='medium')  # 简化X轴标签
ax1.set_ylabel(r'Recall@20', color=color, fontsize=12, fontweight='medium')
# 用plot（线性轴）+ 精准数据点匹配
ax1.plot(noise_scale, recall20, color=color, marker='o', markersize=7, 
         linestyle='-', linewidth=1.5, label=rf'Recall@20({dataset})')
ax1.set_yticks(recall20_dict)  # 强制Recall纵轴刻度
ax1.tick_params(axis='y', labelcolor=color, labelsize=10,pad=5)
ax1.set_ylim(min(recall20_dict), max(recall20_dict))

# ========== 横轴终极优化（解决核心问题） ==========
ax1.set_xticks(noise_scale)  # 严格使用你的刻度值
ax1.tick_params(axis='x', pad=5)  # 调整X轴标签间距
ax1.set_xlim(min(noise_scale)-0.01, max(noise_scale)+0.01)  # 扩展X轴范围，让第一个/最后一个刻度不贴边

# 创建右Y轴（NDCG@20）
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel(r'NDCG@20', color=color, fontsize=12, fontweight='medium')
ax2.plot(noise_scale, ndcg20, color=color, marker='^', markersize=7, 
         linestyle='--', linewidth=1.5, label=rf'NDCG@20({dataset})')
ax2.tick_params(axis='y', labelcolor=color, labelsize=10)
ax2.set_yticks(ndcg20_dict)  # 强制NDCG纵轴刻度
ax2.set_ylim(min(ndcg20_dict), max(ndcg20_dict))

# ========== 图例/布局优化（贴合参考图） ==========
# 合并图例，调整位置和样式
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', 
           frameon=True, framealpha=0.9, fontsize=9)

fig.tight_layout()  # 自动适配布局

# 保存高清图片（无截断）
plt.savefig(f'{dataset}_lambda_2_tmp.svg', dpi=300, bbox_inches='tight', facecolor='white')