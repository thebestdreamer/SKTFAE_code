import torch


def kmeans(data, k, max_time=100):
    # the k-means based on torch tensor
    n, m = data.shape
    ini = torch.randperm(n)[:k]  # 只有一维需要逗号
    midpoint = data[ini]  # 随机选择k个起始点
    time = 0
    last_label = 0
    while (time < max_time):
        d = data.unsqueeze(0).repeat(k, 1, 1)  # shape k*n*m
        mid_ = midpoint.unsqueeze(1).repeat(1, n, 1)  # shape k*n*m
        dis = torch.sum((d - mid_) ** 2, 2)  # 计算距离
        label = dis.argmin(0)  # 依据最近距离标记label
        if torch.sum(label != last_label) == 0:  # label没有变化,跳出循环
            return label
        last_label = label
        for i in range(k):  # 更新类别中心点，作为下轮迭代起始
            kpoint = data[label == i]
            if i == 0:
                midpoint = kpoint.mean(0).unsqueeze(0)
            else:
                midpoint = torch.cat([midpoint, kpoint.mean(0).unsqueeze(0)], 0)
        time += 1
    return label


def main():
    a = 1
    b = 1
    kmeans(a, b)


if __name__ == '__main__':
    main()
