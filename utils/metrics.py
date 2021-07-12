import math


def max_list(lt):
    temp = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
    return max_str


def kmeans_result_normalize(y_t, y_p):
    # 将聚类出的第几簇和实际的分类对应起来，每个簇的分类为该簇中元素最多的分类（purity）
    ture_dict = {index: value for index, value in enumerate(y_t)}
    index_dict = {}
    value_dict = {}
    for index, item in enumerate(y_p):
        if item not in index_dict.keys():
            index_dict[item] = [index]
            value_dict[item] = [ture_dict[index]]
        else:
            index_dict[item].append(index)
            value_dict[item].append(ture_dict[index])

    for item in value_dict.keys():
        temp = max_list(value_dict[item])
        for index in index_dict[item]:
            y_p[index] = temp
    return y_t, y_p


def entropy(y_t, y_p):
    pc = {}
    ac = {}
    length = len(list(set(y_t)))
    for index, value in enumerate(y_p):
        if value not in pc.keys():
            pc[value] = [index]
        else:
            pc[value].append(index)

    for index, value in enumerate(y_t):
        if value not in ac.keys():
            ac[value] = [index]
        else:
            ac[value].append(index)

    temp_result = {}
    for i in range(length):
        temp_sum = []
        for j in range(length):
            item = len(list(set(pc[i]) & set(ac[j]))) / len(pc[i])
            if item != 0:
                temp_sum.append(item * math.log(item, 2))
            else:
                temp_sum.append(0)
        temp_result[i] = -sum(temp_sum)
    result = 0.0
    topk = 10
    for i in range(topk):
        result += len(pc[i]) / len(y_t) * temp_result[i]
    return temp_result, result


def purity(y_t, y_p):
    pc = {}
    ac = {}
    length = len(list(set(y_t)))
    for index, value in enumerate(y_p):
        if value not in pc.keys():
            pc[value] = [index]
        else:
            pc[value].append(index)

    for index, value in enumerate(y_t):
        if value not in ac.keys():
            ac[value] = [index]
        else:
            ac[value].append(index)

    temp_result = {}
    for i in range(length):
        temp_fenzi = []
        for j in range(length):
            temp_fenzi.append(len(list(set(pc[i]) & set(ac[j]))))
        temp_result[i] = max(temp_fenzi) / len(pc[i])

    result = 0.0
    topk = 10
    for i in range(topk):
        result += len(pc[i]) / len(y_t) * temp_result[i]
    return temp_result, result


if __name__ == "__main__":
    y_t = [0, 1, 2, 1, 3, 1, 3, 3]
    y_p = [0, 1, 3, 1, 3, 1, 3, 2]

    print(kmeans_result_normalize(y_t, y_p))

    #
    # print(purity(y_t, y_p))
    # print(entropy(y_t, y_p))
