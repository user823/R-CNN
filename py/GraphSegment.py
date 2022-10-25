import cv2 as cv
import numpy as np


# 点通过图像按行编号定义
# 边定义
class Edge:
    # end1和end2表示边的两个端点，端点通过编号来表示
    def __init__(self, end1, end2, weight):
        self.end1 = end1
        self.end2 = end2
        self.weight = weight

    def __lt__(self, edge):
        return self.weight < edge.weight

    def __str__(self):
        return "({},{},{})".format(self.end1, self.end2, self.weight)


class GraphSegmentation:
    def __init__(self, K, sigma, minsize):
        self.K = K
        self.sigma = sigma
        self.minsize = minsize
        self.parent = []
        self.edges = []
        self.Int_ = []
        self.num = []
        self.img = None

    def filter_(self):
        if self.img is not None:
            self.img = cv.GaussianBlur(self.img, (0, 0), self.sigma, self.sigma)

    def initial(self, img):
        self.img = img

        self.parent = []
        self.edges = []
        self.Int_ = []
        self.num = []
        h, w = self.img.shape[:2]
        size = h * w
        for i in range(size):
            self.parent.append(i)
            self.Int_.append(0)
            self.num.append(1)

        n_channels = 1 if np.ndim(self.img) == 2 else 3
        for i in range(h):
            for j in range(w):
                if j < w - 1:
                    total = 0
                    for k in range(n_channels):
                        diff = int(img[i][(j + 1) * n_channels + k]) - int(img[i][j * n_channels + k])
                        total += diff * diff
                    total = np.sqrt(total)
                    self.edges.append(Edge(i * w + j, i * w + j + 1, total))

                if i < h - 1:
                    total = 0
                    for k in range(n_channels):
                        diff = int(img[i + 1][j * n_channels + k]) - int(img[i][j * n_channels + k])
                        total += diff * diff
                    total = np.sqrt(total)
                    self.edges.append(Edge(i * w + j, (i + 1) * w + j, total))

    def query(self, x):
        if (self.parent[x] != x):
            self.parent[x] = self.query(self.parent[x])
        return self.parent[x]

    def join(self, root1, root2):
        self.parent[root2] = root1
        self.num[root1] += self.num[root2]

    def segmentGraph(self):
        self.edges = sorted(self.edges)
        for edge in self.edges:
            root1 = self.query(edge.end1)
            root2 = self.query(edge.end2)
            Mint_i_j = min(self.Int_[root1] + self.K / self.num[root1], self.Int_[root2] + self.K / self.num[root2])
            if edge.weight < Mint_i_j:
                self.join(root1, root2)
                self.Int_[root1] = edge.weight
                edge.weight = 0

    def filterSmallAreas(self):
        for edge in self.edges:
            if edge.weight > 0:
                root1 = self.query(edge.end1)
                root2 = self.query(edge.end2)
                if self.num[root1] < self.minsize and self.num[root2] < self.minsize:
                    self.join(root1, root2)

    def process(self, img):
        self.initial(img)
        self.filter_()

        ma = np.zeros(self.img.shape[:2])
        h, w = ma.shape
        given_id = np.full(h * w, -1)
        last_id = -1
        self.segmentGraph()
        self.filterSmallAreas()

        for i in range(h):
            for j in range(w):
                root = self.query(i * w + j)
                if given_id[root] == -1:
                    last_id += 1
                    given_id[root] = last_id
                ma[i][j] = given_id[root]
        return ma


