import numpy as np
from tqdm import tqdm
import os
import scipy
import pywt
import matplotlib.pyplot as plt


"""
# 使用前预装 PyWavelets
pip install PyWavelets -i https://mirror.baidu.com/pypi/simple 
"""


def split_data_with_overlap(data, step=512, overlap_ratio=0.5):
    result = []
    total_size = data.shape[0]
    for i in range(int(total_size/(step*overlap_ratio))):
        start_idx = int(i * step * overlap_ratio)
        result.append(data[start_idx:start_idx+step])
    return result


def build_image():
    folder_path = "../dataset/raw"
    ret = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.mat'):
            file_path = os.path.join(folder_path, filename)
            # 加载 .mat 文件
            data = scipy.io.loadmat(file_path)
            key = next(filter(lambda x: 'DE_time' in x, data.keys()))
            time_series = data[key].reshape(-1)
            convert_data = split_data_with_overlap(time_series, step=10240)
            i = 0
            dest_path = "../dataset/pics/{}".format(filename)
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
                for one in tqdm(convert_data):
                    sampling_period  = 1.0 / 12000
                    totalscal = 128    
                    wavename = 'cmor1-1'
                    fc = pywt.central_frequency(wavename)
                    cparam = 2 * fc * totalscal
                    scales = cparam / np.arange(totalscal, 0, -1)
                    coefficients, frequencies = pywt.cwt(one, scales, wavename, sampling_period)
                    amp = abs(coefficients)
                    # frequ_max = frequencies.max()
                    ## print(frequencies.shape)
                    ## print(frequencies)
                    ## print(coefficients.shape)
                    ## print(coefficients)
                    t = np.linspace(0, sampling_period, one.shape[0], endpoint=False)
                    ## print(t)
                    plt.figure(figsize=(8, 6), dpi=100)
                    plt.contourf(t, frequencies, amp, cmap='jet')
                    plt.xticks([])  # 隐藏 x 轴刻度
                    plt.yticks([])  # 隐藏 y 轴刻度
                    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
                    plt.gca().spines['right'].set_visible(False)  # 隐藏右侧边框
                    plt.savefig('../dataset/pics/{}/{}.png'.format(filename, i), dpi=100, bbox_inches='tight')
                    plt.close()
                    i += 1
            else:
                print("{} is converted".format(dest_path))


if __name__ == "__main__":
    build_image()