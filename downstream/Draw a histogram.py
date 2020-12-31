n = 26#召回的总数
# recall = 0
recall = int(np.floor(n*0.7))#总数中的多少个
print(recall)
# a = sorted(multi_pts_seq[:n])[recall]
b = sorted(PtsRep_triple_20[:n])[recall]
# c = sorted(multi_seq[:n])[recall]
d = sorted(unirep_triple_20[:n])[recall]
e = sorted(tape_triple_20[:n])[recall]
f = sorted(knr_triple_20[:n])[recall]
g = sorted(onehot_triple_20[:n])[recall]
print(a,b,c,d,e,f,g)

#画柱状图
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
num_list = np.fix(np.ceil([b,d, e, f, g]))
num_list = list(map(int, num_list))
length = range(len(num_list))
fontsize = 8
error_params=dict(elinewidth=3,ecolor='black',capsize=6,fontsize=fontsize)#设置误差标记参数
plt.bar(length[0], num_list[0], width = 0.6, color = ['orangered'], alpha = 1, label='PtsRep')
plt.bar(length[1], num_list[1], width = 0.6, color = ['seagreen'], alpha = 1, label='BERT')
plt.bar(length[2], num_list[2], width = 0.6, color = ['royalblue'], alpha = 1, label='UniRep')
plt.bar(length[3], num_list[3], width = 0.6, color = ['dimgrey'], alpha = 1, label='KNR')
plt.bar(length[4], num_list[4], width = 0.6, color = ['darkgrey'], alpha = 1, label='OneHot')


for a,b, c in zip(length,num_list, num_list):
    plt.text(a, b, c, ha='center', va= 'bottom',fontsize=fontsize) 
# plt.yscale('symlog', fontsize=fontsize)
plt.ylim((10, 6000))
x = range(0, 5, 1)

plt.ylabel("Max brightness observed", fontsize=12)#纵坐标名字
plt.xticks(x, ['PtsRep-BERT-UniRep', 'PtsRep','BERT-UniRep', 'BERT','UniRep','KNR'], fontsize=8)
plt.legend(loc = "upper left", fontsize = 8)
plt.title("18 of 26")
# plt.savefig("D:\\linux_test_data\\gfp_recall\\pdf\\brightness_single.pdf")
plt.show()
