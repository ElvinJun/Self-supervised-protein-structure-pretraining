import matplotlib.pyplot as plt
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #
plt.style.use('seaborn-whitegrid')
#折线图
#点的横坐标
n = 26
random = [i * 25517/n for i in range(1, n+1)]
# x = sorted(spsr[:n])#线1的纵坐标
y = [i/n  for i in range(1, n +1)]#线2的纵坐标
# plt.plot(sorted(multi_pts_seq[:n]),y,color = 'black',label="PtsRep-BERT-UniRep")#s-:方形

# plt.plot(sorted(multi_seq[:n]),y,color = 'orange',label="BERT-UniRep")#s-:方形

plt.plot(sorted(PtsRep_triple[:n]),y,color = 'orangered',label="PtsRep")#s-:方形

plt.plot(sorted(tape_triple[:n]),y,color = 'seagreen',label="TAPE-BERT")#s-:方形

plt.plot(sorted(unirep_triple[:n]),y,color = 'royalblue',label="UniRep")#s-:方形

plt.plot(sorted(knr_triple[:n ]),y,color = 'dimgrey',label="KNR")#s-:方形

plt.plot(sorted(onehot_triple[:n ]),y,color = 'darkgrey',label="One-hot")#s-:方形

# plt.plot(sorted(random[:n]),y,color = 'black',label="Random")#s-:方形

plt.xlabel("Testing budget", fontsize=12)#横坐标名字
plt.ylabel("Recall", fontsize=12)#纵坐标名字
plt.legend(loc = "best", fontsize=10)#图例

plt.xlim((10, 26000))
plt.ylim((0, 1))
plt.xscale('symlog')
# plt.savefig("recall_0.1%.pdf")
plt.savefig("D:\\Prof Lin’ Lab-Elvin\\撰写文章\\图例\\投稿\\材料_new\\recall_triple.pdf")
plt.show()

print('PtsRep_triple:', PtsRep_triple[0])
print('tape_triple:', tape_triple[0])
print('unirep_triple', unirep_triple[0])
print('knr_triple', knr_triple[0])
print('onehot_triple', onehot_triple[0])

--------------------------------------------------------

import matplotlib.pyplot as plt

n = 1
# recall = 0
recall = int(np.floor(n*0.7))
print(recall)
a = sorted(PtsRep_triple[:n])[recall]
c = sorted(unirep_triple[:n])[recall]
b = sorted(tape_triple[:n])[recall]
d = sorted(knr_triple[:n])[recall]
e = sorted(onehot_triple[:n])[recall]
print(a,b,c,d,e)


plt.style.use('seaborn-whitegrid')
num_list = np.fix(np.around ([a, b, c, d, 1391]))
num_list = list(map(int, num_list))
length = range(len(num_list))
fontsize = 12
error_params=dict(elinewidth=3,ecolor='black',capsize=6,fontsize=fontsize)#设置误差标记参数
plt.bar(length[0], num_list[0], width = 0.6, color = ['orangered'], alpha = 1, label='PtsRep')
plt.bar(length[1], num_list[1], width = 0.6, color = ['seagreen'], alpha = 1, label='TAPE-BERT')
plt.bar(length[2], num_list[2], width = 0.6, color = ['royalblue'], alpha = 1, label='UniRep')
plt.bar(length[3], num_list[3], width = 0.6, color = ['dimgrey'], alpha = 1, label='KNR')
plt.bar(length[4], num_list[4], width = 0.6, color = ['darkgrey'], alpha = 1, label='One-hot')


for a,b, c in zip(length,num_list, num_list):
    plt.text(a, b, c, ha='center', va= 'bottom',fontsize=fontsize) 
plt.yscale('symlog', fontsize=fontsize)
plt.ylim((10, 6000))
x = range(0, 5, 1)

plt.ylabel("Max brightness observed", fontsize=12)#纵坐标名字
plt.xticks(x, ['PtsRep', 'TAPE','UniRep', 'KNR','One-hot'], fontsize=12)
plt.legend(loc = "upper left", fontsize = 10)
# plt.title("18 of 256")
plt.savefig("D:\\Prof Lin’ Lab-Elvin\\撰写文章\\图例\\投稿\\材料_new\\brightness_triple.pdf")
plt.show()
