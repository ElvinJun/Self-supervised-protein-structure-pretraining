file = open('D:\\linux_test_data\\mutant\\single_mutant\\gfp.txt', 'r').readlines()
datasets = {}
for f in range(len(file)):
    datasets[file[f].split()[0]] = float(file[f].split()[1])
# print(datasets)
name = open('D:\\linux_test_data\\mutant\\single_mutant\\name.txt', 'r').readlines()
mutant_count = open('D:\\linux_test_data\\mutant\\single_mutant\\mutant_count.txt', 'r').readlines()
dic = {}
mutant_train = []
mutant_test = []
for i in range(len(name)):
    key = name[i].strip('\n')
    value = mutant_count[i].strip('\n')
    if key in datasets:
        if int(value) <= 3 and datasets[key] >= 3:
            mutant_train.append(key)
        elif int(value) > 3 and datasets[key] >= 3:
            mutant_test.append(key)
print(len(mutant_test))
test_file = open('D:\\linux_test_data\\mutant\\test_3_more_only_bright.txt', 'w')
for n in mutant_test:
    test_file.write(n + '.npy' +'\n')
    
# test_file.close()
print(len(mutant_train))

import random
# file_list = open('D:\\linux_test_data\\nature_right\\nature_train.txt', 'r').readlines()
random.shuffle(mutant_train)#origin dataset
file_list = mutant_train[:]
test = []
train = []
# range_n = 1
count = 0
for i in range(len(file_list)):
    if i % 10 == 8 or i % 10 == 9:
        t = file_list[i]
        test.append(t)
    else:
        t = file_list[i]
        train.append(t)
print(len(test), len(train))
f_train = open('D:\\linux_test_data\\mutant\\cv_3_only_brigtness\\train_4.txt', 'w')
ft = open('D:\\linux_test_data\\mutant\\cv_3_only_brigtness\\valid_4.txt', 'w')
for t1 in train:
    f_train.write(t1 + '.npy' +'\n')
for t2 in test:
    ft.write(t2 + '.npy' +'\n')
f_train.close()
ft.close()
