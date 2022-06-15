"""
This file is derived from [PlaneAE](https://github.com/svip-lab/PlanarReconstruction).
Modified for [PlanarRecon](https://github.com/neu-vi/PlanarRecon) by Yiming Xie.

Original License:
MIT License

Copyright (c) 2019 Zehao Yu, Jia Zheng, Dongze Lian, Zihan Zhou, Shenghua Gao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Bin_Mean_Shift(nn.Module):
    def __init__(self, train_iter=5, test_iter=10, bandwidth=0.3, device='cpu'):
        super(Bin_Mean_Shift, self).__init__()
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.bandwidth = bandwidth / 2.
        self.anchor_num = 5
        self.sample_num = 1000
        self.device = device
        self.threshold = 7

    def generate_seed(self, point, bin_num):
        """
        :param point: tensor of size (K, 2)
        :param bin_num: int
        :return: seed_point
        """

        def get_start_end(a, b, k):
            start = a + (b - a) / ((k + 1) * 2)
            end = b - (b - a) / ((k + 1) * 2)
            return start, end

        min_a, min_b, min_c, min_x, min_y, min_z = point.min(dim=0)[0]
        max_a, max_b, max_c, max_x, max_y, max_z = point.max(dim=0)[0]

        start_a, end_a = get_start_end(min_a.item(), max_a.item(), bin_num)
        start_b, end_b = get_start_end(min_b.item(), max_b.item(), bin_num)
        start_c, end_c = get_start_end(min_c.item(), max_c.item(), bin_num)
        start_x, end_x = get_start_end(min_x.item(), max_x.item(), bin_num)
        start_y, end_y = get_start_end(min_y.item(), max_y.item(), bin_num)
        start_z, end_z = get_start_end(min_z.item(), max_z.item(), bin_num)

        a = torch.linspace(start_a, end_a, bin_num).view(bin_num, 1, 1, 1, 1, 1)
        b = torch.linspace(start_b, end_b, bin_num).view(1, bin_num, 1, 1, 1, 1)
        c = torch.linspace(start_c, end_c, bin_num).view(1, 1, bin_num, 1, 1, 1)
        x = torch.linspace(start_x, end_x, bin_num).view(1, 1, 1, bin_num, 1, 1)
        y = torch.linspace(start_y, end_y, bin_num).view(1, 1, 1, 1, bin_num, 1)
        z = torch.linspace(start_z, end_z, bin_num).view(1, 1, 1, 1, 1, bin_num)

        a_repeat = a.repeat(1, bin_num, bin_num, bin_num, bin_num, bin_num).view(-1, 1)
        b_repeat = b.repeat(bin_num, 1, bin_num, bin_num, bin_num, bin_num).view(-1, 1)
        c_repeat = c.repeat(bin_num, bin_num, 1, bin_num, bin_num, bin_num).view(-1, 1)
        x_repeat = x.repeat(bin_num, bin_num, bin_num, 1, bin_num, bin_num).view(-1, 1)
        y_repeat = y.repeat(bin_num, bin_num, bin_num, bin_num, 1, bin_num).view(-1, 1)
        z_repeat = z.repeat(bin_num, bin_num, bin_num, bin_num, bin_num, 1).view(-1, 1)

        return torch.cat((a_repeat, b_repeat, c_repeat, x_repeat, y_repeat, z_repeat), dim=1).to(self.device)

    def filter_seed(self, point, prob, seed_point, bandwidth, min_count=3):
        """
        :param point: tensor of size (K, 2)
        :param seed_point: tensor of size (n, 2)
        :param prob: tensor of size (K, 1) indicate probability of being plane
        :param min_count:  mini_count within a bandwith of seed point
        :param bandwidth: float
        :return: filtered_seed_points
        """
        distance_matrix = self.cal_distance_matrix(seed_point, point)  # (n, K)
        thres_matrix = (distance_matrix < bandwidth).type(torch.float32) * prob.t()
        count = thres_matrix.sum(dim=1)  # (n, 1)
        valid = count > min_count
        return seed_point[valid]

    def cal_distance_matrix(self, point_a, point_b):
        """
        :param point_a: tensor of size (m, 4)
        :param point_b: tensor of size (n, 4)
        :return: distance matrix of size (m, n)
        """
        m, n = point_a.size(0), point_b.size(0)

        a_repeat = point_a.repeat(1, n).view(n * m, 6)  # (n*m, 2)
        b_repeat = point_b.repeat(m, 1)  # (n*m, 2)

        distance = torch.nn.PairwiseDistance(keepdim=True)(a_repeat, b_repeat)  # (n*m, 1)

        return distance.view(m, n)

    def shift(self, point, prob, seed_point, bandwidth):
        """
        shift seed points
        :param point: tensor of size (K, 2)
        :param seed_point: tensor of size (n, 2)
        :param prob: tensor of size (K, 1) indicate probability of being plane
        :param bandwidth: float
        :return:  shifted points with size (n, 2)
        """
        distance_matrix = self.cal_distance_matrix(seed_point, point)  # (n, K)
        kernel_matrix = torch.exp((-0.5 / bandwidth ** 2) * (distance_matrix ** 2)) * (
                1. / (bandwidth * np.sqrt(2 * np.pi)))
        weighted_matrix = kernel_matrix * prob.t()

        # normalize matrix
        normalized_matrix = weighted_matrix / weighted_matrix.sum(dim=1, keepdim=True)
        shifted_point = torch.matmul(normalized_matrix, point)  # (n, K) * (K, 2) -> (n, 2)

        return shifted_point

    def label2onehot(self, labels):
        """
        convert a label to one hot vector
        :param labels: tensor with size (n, 1)
        :return: one hot vector tensor with size (n, max_lales+1)
        """
        n = labels.size(0)
        label_num = torch.max(labels).int() + 1

        onehot = torch.zeros((n, label_num))
        onehot.scatter_(1, labels.long(), 1.)

        return onehot.to(self.device)

    def merge_center(self, seed_point, bandwidth=0.25):
        """
        merge close seed points
        :param seed_point: tensor of size (n, 2)
        :param bandwidth: float
        :return: merged center
        """
        n = seed_point.size(0)

        # 1. calculate intensity
        distance_matrix = self.cal_distance_matrix(seed_point, seed_point)  # (n, n)
        intensity = (distance_matrix < bandwidth).type(torch.float32).sum(dim=1)

        # merge center if distance between two points less than bandwidth
        sorted_intensity, indices = torch.sort(intensity, descending=True)
        is_center = np.ones(n, dtype=np.bool)
        indices = indices.cpu().numpy()
        center = np.zeros(n, dtype=np.uint8)

        labels = np.zeros(n, dtype=np.int32)
        cur_label = 0
        for i in range(n):
            if is_center[i]:
                labels[indices[i]] = cur_label
                center[indices[i]] = 1
                for j in range(i + 1, n):
                    if is_center[j]:
                        if distance_matrix[indices[i], indices[j]] < bandwidth:
                            is_center[j] = 0
                            labels[indices[j]] = cur_label
                cur_label += 1
        # print(labels)
        # print(center)
        # return seed_point[torch.ByteTensor(center)]

        # change mask select to matrix multiply to select points
        one_hot = self.label2onehot(torch.Tensor(labels).view(-1, 1))  # (n, label_num)
        weight = one_hot / one_hot.sum(dim=0, keepdim=True)  # (n, label_num)

        return torch.matmul(weight.t(), seed_point)

    def cluster(self, point, center):
        """
        cluter each point to nearset center
        :param point: tensor with size (K, 2)
        :param center: tensor with size (n, 2)
        :return: clustering results, tensor with size (K, n) and sum to one for each row
        """
        # plus 0.01 to avoid divide by zero
        distance_matrix = 1. / (self.cal_distance_matrix(point, center) + 0.01)  # (K, n)
        invalid =  distance_matrix.max(dim=1)[0] < self.threshold
        segmentation = F.softmax(distance_matrix, dim=1)
        return segmentation, invalid

    def bin_shift(self, prob, embedding):
        """
        discrete seeding mean shift in training stage
        :param prob: tensor with size (1, h, w) indicate probability of being plane
        :param embedding: tensor with size (2, h, w)
        :param param: tensor with size (3, h, w)
        :param gt_seg: ground truth instance segmentation, used for sampling planar embeddings
        :param bandwidth: float
        :return: segmentation results, tensor with size (h*w, K), K is cluster number, row sum to 1
                 sampled segmentation results, tensor with size (N, K) where N is sample size, K is cluster number, row sum to 1
                center, tensor with size (K, 2) cluster center in embedding space
                sample_prob, tensor with size (N, 1) sampled probability
                sample_seg, tensor with size (N, 1) sampled ground truth instance segmentation
                sample_params, tensor with size (3, N), sampled params
        """

        n = embedding.shape[0]

        # random sample planar region data points using ground truth label to speed up training
        # rand_index = np.random.choice(np.arange(0, n)[seg.cpu().numpy() != 20], self.sample_num)
        rand_index = np.random.choice(np.arange(0, n), self.sample_num)
        # rand_index = np.arange(0, n)

        sample_embedding = embedding[rand_index]
        sample_prob = prob[rand_index]

        # generate seed points and filter out those with low density to speed up training
        seed_point = self.generate_seed(sample_embedding, self.anchor_num)
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=3)
        if torch.numel(seed_point) <= 0:
            return None, None, None, None

        with torch.no_grad():
            iter = self.train_iter if self.training else self.test_iter
            for iter in range(iter):
                seed_point = self.shift(sample_embedding, sample_prob, seed_point, self.bandwidth)

        # filter again and merge seed points
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=10)
        if torch.numel(seed_point) <= 0:
            return None, None, None, None

        center = self.merge_center(seed_point, bandwidth=self.bandwidth)

        # cluster points
        segmentation, invalid = self.cluster(embedding, center)

        return segmentation, center, invalid, sample_prob

    def forward(self, logit, embedding):
        # apply mean shift to every item
        segmentation, center, invalid, prob = self.bin_shift(torch.sigmoid(logit), embedding)

        return segmentation, center, invalid, prob

    def test_forward(self, prob, embedding, param, mask_threshold):
        """
        :param prob: probability of planar, tensor with size (1, h, w)
        :param embedding: tensor with size (2, h, w)
        :param mask_threshold: threshold of planar region
        :return: clustering results: numpy array with shape (h, w),
                 sampled segmentation results, tensor with size (N, K) where N is sample size, K is cluster number, row sum to 1
                 sample_params, tensor with size (3, N), sampled params
        """

        c, h, w = embedding.size()

        embedding = embedding.view(c, h * w).t()
        prob = prob.view(h * w, 1)
        param = param.view(3, h * w)

        # random sample planar region data points
        rand_index = np.random.choice(np.arange(0, h * w)[prob.cpu().numpy().reshape(-1) > mask_threshold],
                                      self.sample_num)

        sample_embedding = embedding[rand_index]
        sample_prob = prob[rand_index]
        sample_param = param[:, rand_index]

        # generate seed points and filter out those with low density
        seed_point = self.generate_seed(sample_embedding, self.anchor_num)
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=3)

        with torch.no_grad():
            # start shift points
            for iter in range(self.test_iter):
                seed_point = self.shift(sample_embedding, sample_prob, seed_point, self.bandwidth)

        # filter again and merge seed points
        seed_point = self.filter_seed(sample_embedding, sample_prob, seed_point, bandwidth=self.bandwidth, min_count=10)

        center = self.merge_center(seed_point, bandwidth=self.bandwidth)

        # cluster points using sample_embedding
        segmentation = self.cluster(embedding, center)

        sampled_segmentation = segmentation[rand_index]

        return segmentation, sampled_segmentation, sample_param
