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
plt.plot(sorted(multi[:n]),y,color = 'black',label="Multi-model")#s-:方形

plt.plot(sorted(PtsRep_triple[:n]),y,color = 'orangered',label="PtsRep")#s-:方形

plt.plot(sorted(tape_triple[:n]),y,color = 'seagreen',label="BERT")#s-:方形

plt.plot(sorted(unirep_triple[:n]),y,color = 'royalblue',label="UniRep")#s-:方形

plt.plot(sorted(knr_triple[:n ]),y,color = 'dimgrey',label="KNR")#s-:方形

plt.plot(sorted(onehot_triple[:n ]),y,color = 'darkgrey',label="OneHot")#s-:方形

plt.plot(sorted(random[:n]),y,color = 'black',label="Random")#s-:方形

plt.xlabel("Testing budget", fontsize=12)#横坐标名字
plt.ylabel("Recall", fontsize=12)#纵坐标名字
plt.legend(loc = "best")#图例

plt.xlim((10, 26000))
plt.ylim((0, 1))
plt.xscale('symlog')
# plt.savefig("recall_0.1%.pdf")
plt.show()

print('PtsRep_triple:', PtsRep_triple[0])
print('tape_triple:', tape_triple[0])
print('unirep_triple', unirep_triple[0])
print('knr_triple', knr_triple[0])
print('onehot_triple', onehot_triple[0])
