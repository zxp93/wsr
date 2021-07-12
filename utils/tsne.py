import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from sklearn.manifold import TSNE



def plot_embedding(data, label, title):

    fig = plt.figure()
    ax = plt.subplot(111)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    type7_x = []
    type7_y = []
    type8_x = []
    type8_y = []
    type9_x = []
    type9_y = []
    type10_x = []
    type10_y = []


    for i in range(data.shape[0]):
        if label[i] == 0:
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])
        if label[i] == 1:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])
        if label[i] == 2:
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])
        if label[i] == 3:
            type4_x.append(data[i][0])
            type4_y.append(data[i][1])
        if label[i] == 4:
            type5_x.append(data[i][0])
            type5_y.append(data[i][1])
        if label[i] == 5:
            type6_x.append(data[i][0])
            type6_y.append(data[i][1])
        if label[i] == 6:
            type7_x.append(data[i][0])
            type7_y.append(data[i][1])
        if label[i] == 7:
            type8_x.append(data[i][0])
            type8_y.append(data[i][1])
        if label[i] == 8:
            type9_x.append(data[i][0])
            type9_y.append(data[i][1])
        if label[i] == 9:
            type10_x.append(data[i][0])
            type10_y.append(data[i][1])


    color = plt.cm.Set3(0)
    color = np.array(color).reshape(1, 4)
    color1 = plt.cm.Set3(1)
    color1 = np.array(color1).reshape(1, 4)
    color2 = plt.cm.Set3(2)
    color2 = np.array(color2).reshape(1, 4)
    color3 = plt.cm.Set3(3)
    color3 = np.array(color3).reshape(1, 4)

    type1 = plt.scatter(type1_x, type1_y, s=10, c='#377EB8')
    type2 = plt.scatter(type2_x, type2_y, s=10, c='#66C2A5')
    type3 = plt.scatter(type3_x, type3_y, s=10, c='#FF6C91')
    type4 = plt.scatter(type4_x, type4_y, s=10, c='#FF7400')
    type5 = plt.scatter(type5_x, type5_y, s=10, c='#00A13B')
    type6 = plt.scatter(type6_x, type6_y, s=10, c='#D62728')
    type7 = plt.scatter(type7_x, type7_y, s=10, c='#A38CF4')
    type8 = plt.scatter(type8_x, type8_y, s=10, c='#F461DD')
    type9 = plt.scatter(type9_x, type9_y, s=10, c='#FFD92F')
    type10 = plt.scatter(type10_x, type10_y, s=10, c='#8C564B')

    mapping = {'Financial': 0.0,
               'Tools': 1.0,
               'Messaging': 2.0,
               'eCommerce': 3.0,
               'Payments': 4.0,
               'Social': 5.0,
               'Enterprise': 6.0,
               'Mapping': 7.0,
               'Science': 8.0,
               'Government': 9.0}
    plt.legend((type1, type2, type3, type4, type5, type6, type7, type8, type9, type10),
               ('Financial', 'Tools', 'Messaging', 'eCommerce', 'Payments', 'Social', 'Enterprise', 'Mapping', 'Science', 'Government'),
               loc=(0.97, 0.5))

    # plt.xticks()
    # plt.yticks()
    # plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    ax.spines['right'].set_visible(False)  # 去除右边框
    ax.spines['top'].set_visible(False)  # 去除上边框
    return fig

def plot_2D(data, label, file_name):
    #n_samples, n_features = data.shape
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=2019, perplexity=50, learning_rate=100, n_iter=2000,) #使用TSNE对特征降到二维
    #t0 = time()
    result = tsne.fit_transform(data) #降维后的数据
    #print(result.shape)
    #画图
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)')
                         #% (time() - t0))
    fig.subplots_adjust(right=0.8)  #图例过大，保存figure时无法保存完全，故对此参数进行调整
    fig.savefig(file_name,dpi=500)
