import cv2
import numpy as np


def get_affine_matrix(src: np.float32, dst: np.float32):
    """
    计算仿射变换转换矩阵
    :param src: 坐标系1
    :param dst: 坐标系2
    :return: 转换矩阵
    """
    affine_matrix = cv2.getAffineTransform(src, dst)
    affine_matrix_d3 = np.vstack((affine_matrix, np.float32([0, 0, 1])))
    return affine_matrix_d3


def get_affine(src: np.float32, affine_matrix_d3: np.float32):
    """
    坐标转换
    :param src: 源坐标
    :param affine_matrix_d3: 参数
    :return:
    """
    source_d3 = np.append(src, 1)
    target_d3 = np.dot(source_d3, affine_matrix_d3.T)
    return target_d3[0: 2]


class Affinity:
    def __init__(self, sample_src: np.float32, sample_dst: np.float32, src: np.float32):
        self.sample_src = sample_src
        self.sample_dst = sample_dst
        self.src = src
        self.dst = []
        self.process()

    def process(self):
        src = self.sample_src[0: 3]
        dst = self.sample_dst[0: 3]
        affine_matrix_d3 = get_affine_matrix(src, dst)

        for i in range(len(self.src)):
            s = self.src[i]
            t = get_affine(s, affine_matrix_d3)
            self.dst.append(t)
            print(t)


if __name__ == '__main__':
    sample_source = np.float32([
        [555094.01215, 3195171.38886],
        [555226.74873, 3195283.80892],
        [555727.40368, 3195284.29477],
        [555868.63769, 3195115.89274],
        [556010.68477, 3194947.39155],
        [556293.33127, 319461116258],
        [556115.76748, 3194461.73850],
        [555783.88142, 3194104.27088],
        [55574314923, 3194541.49745],
        [555194.07770, 3195052.97003]
    ])
    sample_target = np.float32([
        [-525.1, -310.56],
        [-526.07, -136.751],
        [-204.35, 246.8],
        [15.98, 246.65],
        [235.88, 246.56],
        [675.68, 246.53],
        [675.71, 15.38],
        [735.32, -469.22],
        [373.8, -218.9],
        [-369.6, -310.2]
    ])
    source = np.float32([
        [555094.01215, 3195171.38886],
        [555226.74873, 3195283.80892],
        [555727.40368, 3195284.29477],
        [555868.63769, 3195115.89274],
        [556010.68477, 3194947.39155],
        [556293.33127, 319461116258],
        [556115.76748, 3194461.73850],
        [555783.88142, 3194104.27088],
        [55574314923, 3194541.49745],
        [555194.07770, 3195052.97003]
    ])
    Affinity(sample_src=sample_source,
             sample_dst=sample_target,
             src=source)
