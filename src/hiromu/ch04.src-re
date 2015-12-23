= 混合ガウスモデルによる声質変換

この章では、GMMを用いた最も基本的な声質変換の解説及び実装を行う。
なお、声質変換の手法については、戸田氏の論文「Voice Conversion Based on Maximum Likelihood Estimation of Spectral Parameter Trajectory@<bib>{toda-traj}」を、
実装については、r9y9氏のブログ@<bib>{r9y9-gmm}を大いに参照した。

== 混合ガウスモデルの学習処理

学習の処理はかなり単純である。
変換元の特徴量と変換先の特徴量を各フレームごとに結合して得られるデータに対して、学習処理を行ってGMMのパラメータを推定する。
つまり、@<m>{\boldsymbol{x\}_t, \boldsymbol{y\}_t}をそれぞれ@<m>{t}番目のフレームにおける変換元、変換先の特徴量とすると、
@<m>{\boldsymbol{z\}_t = \left [ \boldsymbol{x\}_t ^ \top, \boldsymbol{y\}_t ^ \top \right ] ^ \top}を学習データとして用いることとなる。
ここで、特徴量として@<m>{D}次元のMFCCを用いるとすると、学習データは@<m>{2D}次元となる。
式で表すと以下のようになる。

//texequation{
    \begin{gathered}
        P(\boldsymbol{z}_t | \boldsymbol{\lambda} ^ {(z)}) = \sum_{m = 1}^{M} w_l \mathcal{N} (\boldsymbol{z}_t; \boldsymbol{\mu}_m ^ {(z)}, \boldsymbol{\Sigma}_m ^ {(z)}) \\
        \hat{\boldsymbol{\lambda}} ^ {(z)} = \underset{\boldsymbol{\lambda} ^ {(z)}}{argmax} \prod_{t = 1}^{T} P(\boldsymbol{z}_t, \boldsymbol{\lambda} ^ {(z)})
    \end{gathered}
//}

ここで、@<m>{\boldsymbol{z\}_t}は、@<m>{\boldsymbol{x\}_t, \boldsymbol{y\}_t}の結合特徴量であることから、平均ベクトルと分散共分散行列は以下のように表すことができる。

//texequation{
    \boldsymbol{\mu}_m ^ {(z)} =
        \begin{bmatrix}
            \boldsymbol{\mu}_m ^ {(x)} \\
            \boldsymbol{\mu}_m ^ {(y)}
        \end{bmatrix}, 
    \boldsymbol{\Sigma}_m ^ {(z)} =
        \begin{bmatrix}
            \boldsymbol{\Sigma}_m ^ {(xx)} & \boldsymbol{\Sigma}_m ^ {(xy)} \\ 
            \boldsymbol{\Sigma}_m ^ {(yx)} & \boldsymbol{\Sigma}_m ^ {(yy)}
        \end{bmatrix}
//}

@<m>{\boldsymbol{\mu\}_m ^ {(x)\}, \boldsymbol{\mu\}_m ^ {(y)\}}は、それぞれ@<m>{m}番目のガウス分布における変換元、変換先の平均ベクトルを意味し、
@<m>{\boldsymbol{\Sigma\}_m ^ {(xx)\}, \boldsymbol{\Sigma\}_m ^ {(yy)\}}は、それぞれ@<m>{m}番目のガウス分布における変換元、変換先の分散共分散行列、
そして@<m>{\boldsymbol{\Sigma\}_m ^ {(xy)\}, \boldsymbol{\Sigma\}_m ^ {(yx)\}}は、それぞれ変換元、変換先間の相互共分散行列と表している。

これは、学習したパラメータをプロットしてみるとよく分かる。
以下の図は、変換元、変換先の特徴量をそれぞれ16次元のMFCCとして、混合数32のGMMで学習した際の平均ベクトルをプロットしたものである。
縦軸が混合数、横軸が特徴量の次元であり、左半分が@<m>{\boldsymbol{\mu\}_m ^ {(x)\}}、右半分が@<m>{\boldsymbol{\mu\}_m ^ {(y)\}}を表しているが、
大まかに似たような分布になっていることが分かる。

//image[gmm-mean][学習したGMMの平均ベクトル][scale=0.5]

以下の図は、同様にして得られた分散共分散行列のうち、1番目と2番目のガウス分布のものをプロットしたものである。
4つの領域で同様の分布になっており、さらにそれぞれが対称行列のようになっていることがわかる。

//image[gmm-covar][学習したGMMの分散共分散行列][scale=0.5]

=== 学習処理の実装

GMMMapというクラスに学習済みGMMインスタンスを与えることによって変換処理を行うインスタンスを作るものとして、
まず複数のSTFデータを読みこんで、結合特徴量を作り、GMMに学習させる処理までを実装する。

GMMの学習処理には、scikit-learnのGMM実装であるsklearn.mixture.GMMを用いる。
ここで、気をつけるべきポイントはコンストラクタに@<code>{covariance_type = 'full'}を指定するという点である。
scikit-learnは、デフォルトではパラメータの学習の際に、分散共分散行列を対角行列に制限することで
計算量を削減しているが、より自然な音声を生成するにはその制約を用いずに行列をすべて更新するのがよい。

また、学習結果はPythonのpickleモジュールを利用してシリアライズし、保存することとする。
同様に、DTWはフレーム数に比例して計算量が大きくなるが、同じ音声データに対してのDTWの結果は常に一致することから、
計算結果をpickleでキャッシュすることができる。

実装したソースは、以下の通りである。
特に複雑な処理を実装しているわけではないので、詳細についてはコメントを参照していただきたい。

//listnum[learn_gmmmap][learn_gmmmap.py][python]{
#!/usr/bin/env python

import math
import numpy
import os
import pickle
import re
import sklearn
import sklearn.mixture
import sys

from gmmmap import GMMMap

from stf import STF
from mfcc import MFCC
from dtw import DTW

D = 16  # 利用するMFCCの次元数
M = 32  # GMMの混合数

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print ('Usage: %s [list of source stf] [list of target stf] ' + \
                            '[dtw cache directory] [output file]') % sys.argv[0]
        sys.exit()

    # 対応するSTFファイルのパスが同じ行に書かれたリストを入力として受け取る
    source_list = open(sys.argv[1]).read().strip().split('\n')
    target_list = open(sys.argv[2]).read().strip().split('\n')

    assert len(source_list) == len(target_list)

    learn_data = None

    for i in xrange(len(source_list)):
        # 変換元のSTFを読み取る
        source = STF()
        source.loadfile(source_list[i])

        # 変換元のスペクトル包絡から各フレームごとにMFCCを計算する
        mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = D)
        source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) \
                                    for frame in xrange(source.SPEC.shape[0])])
    
        # 変換先のSTFを読み取る
        target = STF()
        target.loadfile(target_list[i])

        # 変換先のスペクトル包絡から各フレームごとにMFCCを計算する
        mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency, dimension = D)
        target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) \
                                    for frame in xrange(target.SPEC.shape[0])])

        # DTWのキャッシュが存在しない場合はDPマッチングの計算処理を行う
        cache_path = os.path.join(sys.argv[3], '%s_%s.dtw' % \
            tuple(map(lambda x: re.sub('[./]', '_', re.sub('^[./]*', '', x)), \
                                            [source_list[i], target_list[i]])))
        if os.path.exists(cache_path):
            dtw = pickle.load(open(cache_path))
        else:
            dtw = DTW(source_mfcc, target_mfcc, \
                    window = abs(source.SPEC.shape[0] - target.SPEC.shape[0]) * 2)
            with open(cache_path, 'wb') as output:
                pickle.dump(dtw, output)

        # DTWにより変換元のMFCCのフレーム数を変換先と合わせる
        warp_mfcc = dtw.align(source_mfcc)

        # 変換元と変換先のMFCCを結合し、各フレームごとに2D次元の特徴量となるようにする
        data = numpy.hstack([warp_mfcc, target_mfcc])
        # STFファイルごとの結合特徴量を時間方向に繋げて1つの行列とする
        if learn_data is None:
            learn_data = data
        else:
            learn_data = numpy.vstack([learn_data, data])

    # GMMの学習処理を行う
    gmm = sklearn.mixture.GMM(n_components = M, covariance_type = 'full')
    gmm.fit(learn_data)
    gmmmap = GMMMap(gmm)

    # 学習済みインスタンスをpickleでシリアライズする
    with open(sys.argv[4], 'wb') as output:
        pickle.dump(gmmmap, output)
//}

== 混合ガウスモデルによる変換処理

変換処理は、入力として変換元の特徴量@<m>{\boldsymbol{x\}_t}が与えられた時の、変換後の特徴量の条件付き確率密度関数@<m>{P(\boldsymbol{y\}_t | \boldsymbol{x\}_t, \boldsymbol{\lambda\} ^ {(z)\})}を求め、
それが最大化されるような@<m>{\boldsymbol{y\}_t}を求めることによって行われる。

ここで、

//texequation{
    P(\boldsymbol{y}_t | \boldsymbol{x}_t, \boldsymbol{\lambda} ^ {(z)}) = \sum_{m = 1}^{M} P(m | \boldsymbol{x}_t, \boldsymbol{\lambda} ^ {(z)}) P(\boldsymbol{y}_t | \boldsymbol{x}_t, m, \boldsymbol{\lambda} ^ {(z)})
//}

となるが、ここで@<m>{P(m | \boldsymbol{x\}_t, \boldsymbol{\lambda\} ^ {(z)\})}は事後確率として以下のように導出できる。

//texequation{
    P(m | \boldsymbol{x}_t, \boldsymbol{\lambda} ^ {(z)}) = \frac{ w_m \mathcal{N}(\boldsymbol{x}_t; \boldsymbol{\mu}_m ^ {(z)}, \boldsymbol{\Sigma}_m ^ {(z)}) }{ \sum_{n = 1}^{M} w_n \mathcal{N}(\boldsymbol{x}_t, \boldsymbol{\mu}_n ^ {(z)}, \boldsymbol{\Sigma}_n ^ {(z)}) }
//}

さらに、@<m>{P(\boldsymbol{y\}_t | \boldsymbol{x\}_t, m, \boldsymbol{\lambda\} ^ {(z)\})}もGMMでモデル化することができ、その平均ベクトル@<m>{\boldsymbol{E\}_{m, t\} ^ {(y)\}}及び分散共分散行列@<m>{\boldsymbol{D\}_m^{(y)\}}は以下のように表される@<fn>{prml-gmm}。
//footnote[prml-gmm][この平均ベクトル・分散共分散行列の導出については、「パターン認識と機械学習」の「2.3.1 条件付きガウス分布」を参照されたい。]

//texequation{
    \begin{gathered}
        \boldsymbol{E}_{m, t} ^ {(y)} = \boldsymbol{\mu}_m ^ {(y)} + \boldsymbol{\Sigma}_m ^ {(yx)} {\boldsymbol{\Sigma}_m ^ {(xx)}} ^ {-1} (\boldsymbol{x}_t - \boldsymbol{\mu}_m ^ {(x)}) \\
        \boldsymbol{D}_m ^ {(y)} = \boldsymbol{\Sigma}_m ^ {(yy)} + \boldsymbol{\Sigma}_m ^ {(yx)} {\boldsymbol{\Sigma}_m ^ {(xx)}} ^ {-1} \boldsymbol{\Sigma}_m ^ {(xy)}
    \end{gathered}
//}

このとき、最小平均二乗誤差推定(MMSE)によって変換後の特徴量を求めるならば、推定される特徴量を@<m>{\hat{\boldsymbol{y\}_t\}}として、以下の通りとなる。

//texequation{
    \hat{\boldsymbol{y}_t} = E[\boldsymbol{y}_t | \boldsymbol{x}_t] = \int P(\boldsymbol{y}_t | \boldsymbol{x}_t, \boldsymbol{\lambda} ^ {(z)}) \boldsymbol{y}_t d \boldsymbol{y}_t
//}

これに、得られた@<m>{P(\boldsymbol{y\}_t | \boldsymbol{x\}_t, \boldsymbol{\lambda\} ^ {(z)\})}を代入する。

//texequation{
    \begin{split}
        \int P(\boldsymbol{y}_t | \boldsymbol{x}_t, \boldsymbol{\lambda} ^ {(z)}) \boldsymbol{y}_t d \boldsymbol{y}_t
                = & \int \sum_{m = 1}^{M} P(m | \boldsymbol{x}_t, \boldsymbol{\lambda} ^ {(z)}) P(\boldsymbol{y}_t | \boldsymbol{x}_t, m, \boldsymbol{\lambda} ^ {(z)}) \boldsymbol{y}_t d \boldsymbol{y}_t \\
                = & \sum_{m = 1}^{M} P(m | \boldsymbol{x}_t, \boldsymbol{\lambda} ^ {(z)}) \int \mathcal{N}(\boldsymbol{x}_t, \boldsymbol{\mu}_n ^ {(z)}, \boldsymbol{\Sigma}_n ^ {(z)}) \boldsymbol{y}_t d \boldsymbol{y}_t \\
                = & \sum_{m = 1}^{M} P(m | \boldsymbol{x}_t, \boldsymbol{\lambda} ^ {(z)}) \boldsymbol{E}_{m, t} ^ {(y)}
    \end{split}
//}

以上より、@<m>{P(m | \boldsymbol{x\}_t, \boldsymbol{\lambda\} ^ {(z)\})}と@<m>{\boldsymbol{E\}_{m, t\} ^ {(y)\}}の算出を実装すれば変換処理ができることがわかる。

=== 変換処理の実装

変換処理についても、学習処理と同じくコメントにて解説を入れる。
基本的な構造としては、コンストラクタで学習済みのGMMを引数として取り、変換元データと独立な部分を算出した後、
convertメソッドで変換元データを引数として取って、変換処理を行う。

1つ注意すべきポイントは、@<m>{P(m | \boldsymbol{x\}_t, \boldsymbol{\lambda\} ^ {(z)\})}の算出を、前章の定義をそのまま実装するのではなく、
sklearn.mixture.GMMのpredict_probaメソッドを用いて行っているという点である。
このpredict_probaメソッドではGMMから事後確率を計算することができるので、
コンストラクタ内で@<m>{\boldsymbol{\mu\}_m ^ {(x)\}}を平均ベクトル、@<m>{\boldsymbol{\Sigma\}_m ^ {(xx)\}}を分散共分散行列とするGMMインスタンスを作成している。

また、sklearn.mixture.GMMでは、それぞれのガウス分布の平均ベクトルと分散共分散行列を
まとめて多次元配列として扱っているということも把握しておく必要がある。
つまり、平均ベクトルは@<m>{M \times 2D}次元、分散共分散行列は@<m>{M \times 2D \times 2D}次元の配列となっている。

//listnum[gmmmap][gmmmap.py][python]{
#!/usr/bin/python
# coding: utf-8

import numpy as np
from sklearn.mixture import GMM

class GMMMap(object):
    def __init__(self, gmm, swap = False):
        # GMMの学習に用いられるのは結合特徴量なので、その半分がMFCCの次元数となる
        self.M, D = gmm.means_.shape[0], gmm.means_.shape[1] / 2
        self.weights = gmm.weights_

        # 学習済みGMMの平均ベクトルをxとyに分ける
        self.src_means = gmm.means_[:, :D]
        self.tgt_means = gmm.means_[:, D:]

        # 学習済みGMMの分散共分散行列をxx, xy, yx, yyの4つに分ける
        self.covarXX = gmm.covars_[:, :D, :D]
        self.covarXY = gmm.covars_[:, :D, D:]
        self.covarYX = gmm.covars_[:, D:, :D]
        self.covarYY = gmm.covars_[:, D:, D:]

        # GMMの学習時と逆に変換先の話者から変換元の話者へと変換する場合は
        # 平均ベクトルと分散共分散行列を逆に扱えばよい
        if swap:
            self.tgt_means, self.src_means = self.src_means, self.tgt_means
            self.covarYY, self.covarXX = self.covarXX, self.covarYY
            self.covarYX, self.covarXY = self.covarXY, self.covarYX

        # 事後確率の計算のために、それぞれのガウス分布の重みはそのままで
        # xの平均ベクトルとxxの分散共分散行列を用いたGMMのインスタンスを生成する
        self.px = GMM(n_components = self.M, covariance_type = "full")
        self.px.means_ = self.src_means
        self.px.covars_ = self.covarXX
        self.px.weights_ = self.weights

    def convert(self, src):
        D = len(src)

        # ベクトルEをすべてのガウス分布についてまとめて計算する
        E = np.zeros((self.M, D))
        for m in range(self.M):
            # 逆行列に行列を掛け合わせる処理はnumpy.linalg.solveを使うと高速である
            xx = np.linalg.solve(self.covarXX[m], src - self.src_means[m])
            E[m] = self.tgt_means[m] + self.covarYX[m].dot(xx.transpose())
                
        # 事後確率P(m|x)を計算する
        posterior = self.px.predict_proba(np.atleast_2d(src))

        # 事後確率とEの積が求める特徴量となる
        return posterior.dot(E)
//}

== 変換特徴量から音声データへの変換

ここまでで、GMMを用いて特徴量を変換することはできたが、最後に得られた特徴量を音声データへと変換する処理を実装する必要がある。
GMMによって変換される特徴量はMFCCなので、スペクトル包絡へと逆変換した後に、変換元データとして与えられた
STFファイルのスペクトル包絡を上書きして保存する。
そして、TANDEM-STRAIGHTによって、変換後のSTFファイルを音声データへと変換する。

この時に、F0周波数も変換することで、より変換先の話者に似せることができる。
F0の変換については、複雑な処理は行わずに以下のように線形変換をする。

//texequation{
    \hat{\boldsymbol{y}_t} = \frac{\rho ^ {(y)}}{\rho ^ {(x)}} (\boldsymbol{x}_t - \mu ^ {(x)}) + \mu ^ {(y)}
//}

ここで、@<m>{\boldsymbol{x\}_t, \boldsymbol{y\}_t}は対数尺度での変換元、変換先のF0周波数とし、
@<m>{\mu ^ {(x)\}, \rho ^ {(x)\}}はそれぞれ変換元の対数F0周波数の平均及び標準偏差、
同様に、@<m>{\mu ^ {(y)\}, \rho ^ {(y)\}}はそれぞれ変換先の対数F0周波数の平均及び標準偏差とする。
これもスペクトル包絡と同様に、変換元STFのF0データを変換したデータで上書きする。

=== F0変換パラメータの算出処理

まず、F0の変換に用いる、対数F0周波数の平均と標準偏差の算出処理を実装する。
算出に用いるSTFファイルが1つとは限らないので、ファイルごとに対数F0周波数の
平均と二乗平均を更新していき、最後に二乗平均と平均の二乗の差の平方根を取ることで
標準偏差を算出している。

算出結果は、タプルとしてpickleでシリアライズして保存する。

//listnum[learn_f0][learn_f0.py][python]{
#!/usr/bin/env python

import math
import numpy
import pickle
import sklearn
import sys

from stf import STF
from mfcc import MFCC
from dtw import DTW

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: %s [list of source stf] ' + \
                '[list of target stf] [output file]' % sys.argv[0]
        sys.exit()

    source_list = open(sys.argv[1]).read().strip().split('\n')
    target_list = open(sys.argv[2]).read().strip().split('\n')

    assert len(source_list) == len(target_list)

    f0_count = [0, 0]
    f0_mean = [0.0, 0.0]
    f0_square_mean = [0.0, 0.0]

    for i in xrange(len(source_list)):
        source = STF()
        source.loadfile(source_list[i])

        target = STF()
        target.loadfile(target_list[i])

        for idx, stf in enumerate([source, target]):
            count = (stf.F0 != 0).sum()
            f0_mean[idx] = (f0_mean[idx] * f0_count[idx] + \
                    numpy.log(stf.F0[stf.F0 != 0]).sum()) / (f0_count[idx] + count)
            f0_square_mean[idx] = (f0_square_mean[idx] * f0_count[idx] + \
                    (numpy.log(stf.F0[stf.F0 != 0]) ** 2).sum()) / \
                    (f0_count[idx] + count)
            f0_count[idx] += count

    f0_deviation = [math.sqrt(f0_square_mean[i] - f0_mean[i] ** 2) \
                                                    for i in xrange(2)]
    f0 = (tuple(f0_mean), tuple(f0_deviation))

    print f0
    output = open(sys.argv[3], 'wb')
    pickle.dump(f0, output)
    output.close()
//}

=== STFファイルへの変換処理

保存されたGMMMapインスタンスを用いて特徴量を変換し、
MFCCの逆変換によってスペクトル包絡を復元した後にSTFファイルに保存する処理を実装する。

//listnum[convert_gmmmap][convert_gmmmap.py][python]{
#!/usr/bin/env python

import math
import numpy
import pickle
import sklearn
import sys

from gmmmap import GMMMap

from stf import STF
from mfcc import MFCC
from dtw import DTW

K = 32
DIMENSION = 16

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Usage: %s [gmmmap] [f0] [input] [output]' % sys.argv[0]
        sys.exit()

    # 保存されたGMMMapインスタンスを読み込む
    gmm_file = open(sys.argv[1], 'rb')
    gmmmap = pickle.load(gmm_file)
    gmm_file.close()

    # F0の変換パラメータを読み込む
    f0_file = open(sys.argv[2], 'rb')
    f0 = pickle.load(f0_file)
    f0_file.close()

    source = STF()
    source.loadfile(sys.argv[3])
    # F0の有声部分について、パラメータに基づいて変換する
    source.F0[source.F0 != 0] = \
        numpy.exp((numpy.log(source.F0[source.F0 != 0]) - f0[0][0]) \
                                        * f0[1][1] / f0[1][0] + f0[0][1])

    # 変換元のMFCCを計算する
    mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = DIMENSION)
    source_data = numpy.array([mfcc.mfcc(source.SPEC[frame]) \
                                        for frame in xrange(source.SPEC.shape[0])])

    # GMMMapで特徴量を変換し、MFCCの逆変換でスペクトル包絡を復元する
    output_mfcc = numpy.array([gmmmap.convert(source_data[frame])[0] \
                                        for frame in xrange(source_data.shape[0])])
    output_spec = numpy.array([mfcc.imfcc(output_mfcc[frame]) \
                                        for frame in xrange(output_mfcc.shape[0])])

    # STFファイルのスペクトル包絡を上書きして保存する
    source.SPEC = output_spec
    source.savefile(sys.argv[4])
//}

== 混合ガウスモデルによる変換処理の結果

以下の画像は、GMMによる変換の結果を表したグラフである。
それぞれのグラフはMFCCの第1次係数(数値が小さい方)及び第2次係数(数値が大きい方)の推移を表しており、
上から、変換先話者のデータ、GMMによる変換結果のデータ、GMMによる変換に用いた変換元話者のデータをDTWによって伸縮させたものである。
非常にわかりにくいが、下段のグラフよりは中段のグラフの方が、上段のグラフに近いように見えなくもない。
また、この手法ではフレームごとに独立して変換を行っているため、変換結果のグラフがかなりギザギザになっていることもわかる。

//image[gmm-result][GMMによる変換結果と変換先話者データの比較][scale=0.6]
