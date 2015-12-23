= トラジェクトリベースな声質変換

前章では、最も素朴な声質変換手法について説明したが、これには1つ大きな問題点がある。
それは、前後フレームとの関連性を全く考慮せずにフレームごとに独立に変換しているために、
グラフでも推移がギザギザになっていた通り、自然性(声のナチュラルさ)が失われてしまっているという点である。

この章では、前後フレームとのつながりを考慮した、トラジェクトリベースな変換と呼ばれる変換手法について解説する。
なお、この章についても、戸田氏の論文@<bib>{toda-traj}及び、r9y9氏のブログ@<bib>{r9y9-gmm}を参照した。

== 動的特徴量の導入

前後フレームとのつながりを考慮した変換を行うために、MFCCの抽出の際に実装した動的特徴量を導入する。
ここでは、動的特徴量@<m>{\Delta \boldsymbol{x\}_t}を以下のように定義する。

//texequation{
    \Delta \boldsymbol{x}_t = \frac{1}{2} (\boldsymbol{x}_{t + 1} - \boldsymbol{x}_{t - 1})
//}

そして、前章での変換元、変換先の特徴量の代わりに、動的変化量を結合した特徴量
@<m>{\boldsymbol{X\}_t =  [\boldsymbol{x\}_t ^ \top, \Delta \boldsymbol{x\}_t ^ \top ], \boldsymbol{Y\}_t =  [\boldsymbol{y\}_t ^ \top, \Delta \boldsymbol{y\}_t ^ \top ]}を用いる。
時系列で結合した全体の特徴量は、以下の通りとなる。

//texequation{
    \boldsymbol{X} = [\boldsymbol{X}_1 ^ \top, \boldsymbol{X}_2 ^ \top, ..., \boldsymbol{X}_t ^ \top, ..., \boldsymbol{X}_T ^ \top], \ 
    \boldsymbol{Y} = [\boldsymbol{Y}_1 ^ \top, \boldsymbol{Y}_2 ^ \top, ..., \boldsymbol{Y}_t ^ \top, ..., \boldsymbol{Y}_T ^ \top]
//}

この時、@<m>{\boldsymbol{x\}_t, \boldsymbol{y\}_t}が共に@<m>{D}次元のベクトルだとすると、
@<m>{\boldsymbol{X\}, \boldsymbol{Y\}}は共に@<m>{2D \times T}次元の行列となる。

また、前章と同様に変換元と変換先の特徴量を結合した@<m>{4D}次元の特徴量@<m>{\boldsymbol{Z\}_t = [\boldsymbol{X\}_t ^ \top, \boldsymbol{Y\}_t ^ \top]}を用いて
GMMの学習処理を行い、@<m>{\boldsymbol{\lambda\} ^ {(Z)\}}を求める。

=== 学習処理の実装

学習処理については、特徴量に動的変化量を結合する以外には、前章との大きな違いはない。
ここでも、変換処理はTrajectoryGMMMapというクラスで行うものとして、まず学習処理のみを実装する。

//listnum[learn_trajectory][learn_trajectory.py][python]{
#!/usr/bin/env python

import math
import numpy
import os
import pickle
import re
import sklearn
import sklearn.mixture
import sys

from trajectory import TrajectoryGMMMap

from stf import STF
from mfcc import MFCC
from dtw import DTW

D = 16
M = 32

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Usage: %s [list of source stf] [list of target stf] ' + \
                            [dtw cache directory] [output file]' % sys.argv[0]
        sys.exit()

    source_list = open(sys.argv[1]).read().strip().split('\n')
    target_list = open(sys.argv[2]).read().strip().split('\n')

    assert len(source_list) == len(target_list)

    learn_data = None

    for i in xrange(len(source_list)):
        source = STF()
        source.loadfile(source_list[i])

        mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = D)
        source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) \
                                    for frame in xrange(source.SPEC.shape[0])])

        target = STF()
        target.loadfile(target_list[i])

        mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency, dimension = D)
        target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) \
                                    for frame in xrange(target.SPEC.shape[0])])
    
        cache_path = os.path.join(sys.argv[3], '%s_%s.dtw' \
            % tuple(map(lambda x: re.sub('[./]', '_', re.sub('^[./]*', '', x)), \
                                                [source_list[i], target_list[i]])))
        if os.path.exists(cache_path):
            dtw = pickle.load(open(cache_path))
        else:
            dtw = DTW(source_mfcc, target_mfcc, \
                    window = abs(source.SPEC.shape[0] - target.SPEC.shape[0]) * 2)
            with open(cache_path, 'wb') as output:
                pickle.dump(dtw, output)

        warp_mfcc = dtw.align(source_mfcc)

        # 変換元、変換先共に、動的変化量を結合する
        warp_data = numpy.hstack([warp_mfcc, mfcc.delta(warp_mfcc)])
        target_data = numpy.hstack([target_mfcc, mfcc.delta(target_mfcc)])

        data = numpy.hstack([warp_data, target_data])
        if learn_data is None:
            learn_data = data
        else:
            learn_data = numpy.vstack([learn_data, data])

    gmm = sklearn.mixture.GMM(n_components = M, covariance_type = 'full')
    gmm.fit(learn_data)

    gmmmap = TrajectoryGMMMap(gmm, learn_data.shape[0])

    with open(sys.argv[4], 'wb') as output:
        pickle.dump(gmmmap, output)
//}

== トラジェクトリベースな変換処理

=== 時系列に結合した特徴量に対する確率密度関数

前章では、フレームごとに確率密度関数を求め、変換処理を行っていたが
今回は前後でのつながりを考えるために、@<m>{\boldsymbol{X\}, \boldsymbol{Y\}}に関する確率密度関数を考える。
@<m>{\boldsymbol{X\}, \boldsymbol{Y\}}は単純にフレームごとの特徴量を時系列でつなげたものなので、前章で用いた確率密度関数のすべてのフレームにおける積を考えればよい。

//texequation{
    P(\boldsymbol{Y} | \boldsymbol{X}, \boldsymbol{\lambda} ^ {(Z)}) = \prod_{t=1}^{T} \sum_{m=1}^{M} P(m| \boldsymbol{X}_t, \boldsymbol{\lambda} ^ {(Z)}) P(\boldsymbol{Y}_t | \boldsymbol{X}_t, m, \boldsymbol{\lambda} ^ {(Z)})
//}

このとき、前章と同様に、@<m>{P(m| \boldsymbol{X\}_t, \boldsymbol{\lambda\} ^ {(Z)\})}及び@<m>{P(\boldsymbol{Y\}_t | \boldsymbol{X\}_t, m, \boldsymbol{\lambda\} ^ {(Z)\})}は以下のように表すことができる。

//texequation{
    P(m | \boldsymbol{X}_t, \lambda ^ {(Z)}) = \frac{ w_m \mathcal{N}(\boldsymbol{X}_t; \boldsymbol{\mu}_m ^ {(Z)}, \boldsymbol{\Sigma}_m ^ {(Z)}) }{ \sum_{n = 1}^{M} w_n \mathcal{N}(\boldsymbol{X}_t, \boldsymbol{\mu}_n ^ {(Z)}, \boldsymbol{\Sigma}_n ^ {(Z)}) }
//}

//texequation{
    P(\boldsymbol{Y}_t | \boldsymbol{X}_t, m, \boldsymbol{\lambda} ^ {(Z)}) = \mathcal{N} (\boldsymbol{Y}_t; \boldsymbol{E}_{m, t} ^ {(Y)}, \boldsymbol{D}_m ^ {(Y)})
//}

ここで、@<m>{\boldsymbol{E\}_{m, t\} ^ {(y)\}}及び@<m>{\boldsymbol{D\}_m^{(y)\}}は以下の通りである。

//texequation{
    \begin{gathered}
        \boldsymbol{E}_{m, t} ^ {(Y)} = \boldsymbol{\mu}_m ^ {(Y)} + \boldsymbol{\Sigma}_m ^ {(YX)} {\boldsymbol{\Sigma}_m ^ {(XX)}} ^ {-1} (\boldsymbol{X}_t - \boldsymbol{\mu}_m ^ {(X)}) \\
        \boldsymbol{D}_m ^ {(Y)} = \boldsymbol{\Sigma}_m ^ {(YY)} + \boldsymbol{\Sigma}_m ^ {(YX)} {\boldsymbol{\Sigma}_m ^ {(XX)}} ^ {-1} \boldsymbol{\Sigma}_m ^ {(XY)}
    \end{gathered}
//}

また、確率密度関数@<m>{P(\boldsymbol{X\} | \boldsymbol{Y\}, \boldsymbol{\lambda\} ^ {(Z)\})}は、分布系列@<m>{\boldsymbol{m\} = \lbrace m_1, m_2, ..., m_t, ..., m_T \rbrace}を用いて、以下のように表すことができる。

//texequation{
    P(\boldsymbol{Y} | \boldsymbol{X}, \boldsymbol{\lambda} ^ {(Z)}) = \sum_{all \  \boldsymbol{m}} P(\boldsymbol{m} | \boldsymbol{X}_t, \boldsymbol{\lambda} ^ {(Z)}) P(\boldsymbol{Y}_t | \boldsymbol{X}_t, \boldsymbol{m}, \boldsymbol{\lambda} ^ {(Z)})
//}

この確率密度関数に基づいて、求める特徴量@<m>{\hat{\boldsymbol{y\}\}}は以下のように定式化できる。

//texequation{
    \hat{\boldsymbol{y}} = \underset{\boldsymbol{y}}{argmax} \ P(\boldsymbol{Y} | \boldsymbol{X}, \boldsymbol{\lambda} ^ {(Z)})
//}

ただし、この確率密度関数は@<m>{\boldsymbol{X\}}が与えられた時の@<m>{\boldsymbol{Y\}}についての確率を記述したものなので、
@<m>{\boldsymbol{y\}}と@<m>{\boldsymbol{Y\}}の関係性を与える必要がある。
そこで、以下のように定義される変換行列@<m>{\boldsymbol{W\}}を導入する。

//texequation{
    \begin{gathered}
        \boldsymbol{W} = [\boldsymbol{W}_1, \boldsymbol{W}_2, ..., \boldsymbol{W}_t, ..., \boldsymbol{W}_T] ^ \top \otimes \boldsymbol{I}_{\boldsymbol{D} \times \boldsymbol{D}} \\
        \boldsymbol{W}_t = [\boldsymbol{w}_t ^ {(0)}, \boldsymbol{w}_t ^ {(1)}] \\
        \boldsymbol{w}_t ^ {(0)} = [\overset{1st}{0}, \overset{2nd}{0}, ..., \overset{t - 1\ th}{0}, \overset{t\ th}{1}, \overset{t + 1\ th}{0}, ..., \overset{T\ th}{0}] \\
        \boldsymbol{w}_t ^ {(1)} = [\overset{1st}{0}, \overset{2nd}{0}, ..., \overset{t - 1\ th}{-0.5}, \overset{t\ th}{0}, \overset{t + 1\ th}{0.5}, ..., \overset{T\ th}{0}]
    \end{gathered}
//}

ここで、各フレームについて@<m>{\boldsymbol{W\}_t}が@<m>{\boldsymbol{y\}_t}から@<m>{\boldsymbol{Y\}_t}への変換を担っているが、
@<m>{\boldsymbol{w\}_t ^ {(0)\}}が@<m>{\boldsymbol{y\}_t}をそのまま移し、@<m>{\boldsymbol{w\}_t ^ {(1)\}}が、この章の冒頭で定義した動的特徴量と対応していることが分かる。
ただし、実装に合わせるために、@<m>{\Delta \boldsymbol{x\}_0 = \frac{1\}{2\} (\boldsymbol{x\}_{1\} - \boldsymbol{x\}_{0\})}及び@<m>{\Delta \boldsymbol{x\}_T = \frac{1\}{2\} (\boldsymbol{x\}_{T\} - \boldsymbol{x\}_{T - 1\})}とし、
変換行列についても、@<m>{\boldsymbol{W\}_0, \boldsymbol{W\}_T}を以下の図のように設定する。

//image[w-matrix][変換行列Wのイメージ(@<bib>{toda-traj}にあった図を元に編集)][scale=0.25]

元論文@<bib>{toda-traj}では、EMアルゴリズムを用いて、この定式化に基づいた変換パラメータの導出方法も紹介されているが、ここでは準最適な分布系列を用いて計算量を削減する手法を用いる。
これは、それぞれのフレームごとに用いるガウス分布を1つにするということで、
つまり、分布系列@<m>{\boldsymbol{m\}}を準最適な@<m>{\hat{\boldsymbol{m\}\} = [\hat{m\}_1, \hat{m\}_2, ..., \hat{m\}_t, ..., \hat{m\}_T]}の1つに固定することによって、以下のように近似する。

//texequation{
    P(\boldsymbol{Y} | \boldsymbol{X}, \boldsymbol{\lambda} ^ {(Z)}) \simeq P(\hat{\boldsymbol{m}} | \boldsymbol{X}_t, \boldsymbol{\lambda} ^ {(Z)}) P(\boldsymbol{Y}_t | \boldsymbol{X}_t, \hat{\boldsymbol{m}}, \boldsymbol{\lambda} ^ {(Z)})
//}

このとき、@<m>{\hat{\boldsymbol{m\}\}}は以下のように決定される。

//texequation{
    \hat{\boldsymbol{m}} = \underset{\boldsymbol{m}}{argmax} \ P(\boldsymbol{m} | \boldsymbol{X}, \boldsymbol{\lambda} ^ {(Z)})
//}

これらをまとめると、求める特徴量は以下のように表される。

//texequation{
    \begin{split}
        \hat{\boldsymbol{y}} & = \underset{\boldsymbol{y}}{argmax} \ P(\boldsymbol{Y} | \boldsymbol{X}, \boldsymbol{\lambda} ^ {(Z)}) \ subject \ to \ \boldsymbol{Y} = \boldsymbol{W} \boldsymbol{y} \\
                             & \simeq P(\hat{\boldsymbol{m}} | \boldsymbol{X}_t, \boldsymbol{\lambda} ^ {(Z)}) P(\boldsymbol{Y}_t | \boldsymbol{X}_t, \hat{\boldsymbol{m}}, \boldsymbol{\lambda} ^ {(Z)}) \ subject \ to \ \boldsymbol{Y} = \boldsymbol{W} \boldsymbol{y} \\
                             & \qquad \qquad \qquad \qquad \qquad where \ \hat{\boldsymbol{m}} = \underset{\boldsymbol{m}}{argmax} \ P(\boldsymbol{m} | \boldsymbol{X}, \boldsymbol{\lambda} ^ {(Z)})
    \end{split}
//}

=== 変換特徴量の導出

求めた確率密度関数に基づいて、最尤推定を行い、変換後の特徴量を導出する。
対数尤度を考えることによって、以下のように特徴量を求めることができるが、具体的な導出手法については筆者の理解の浅さと7時間後に迫った締め切りのために割愛させていただく。
詳しくは、元論文@<bib>{toda-traj}のAppendixを参照してほしい。

//texequation{
    \hat{\boldsymbol{y}} = (\boldsymbol{W} ^ \top {\boldsymbol{D}_{\hat{\boldsymbol{m}}} ^ {(Y)}} ^ {-1} \boldsymbol{W}) ^ {-1} \boldsymbol{W} ^ \top {\boldsymbol{D}_{\hat{\boldsymbol{m}}} ^ {(Y)}} ^ {-1} \boldsymbol{E}_{\hat{\boldsymbol{m}}} ^ {(Y)}
//}

ここで、@<m>{\boldsymbol{E\}_{\hat{\boldsymbol{m\}\}\} ^ {(Y)\}, {\boldsymbol{D\}_{\hat{\boldsymbol{m\}\}\} ^ {(Y)\}\} ^ {-1\}}は以下のように与えられる。

//texequation{
    \begin{gathered}
        \boldsymbol{E}_{\hat{\boldsymbol{m}}} ^ {(Y)} = [\boldsymbol{E}_{\hat{m}_1, 1} ^ {(Y)}, \boldsymbol{E}_{\hat{m}_2, 2} ^ {(Y)}, ..., \boldsymbol{E}_{\hat{m}_t, t} ^ {(Y)}, ..., \boldsymbol{E}_{\hat{m}_T, T} ^ {(Y)}] \\
        {\boldsymbol{D}_{\hat{\boldsymbol{m}}} ^ {(Y)}} ^ {-1} = diag \ [{\boldsymbol{D}_{\hat{\boldsymbol{m_1}}} ^ {(Y)}} ^ {-1}, {\boldsymbol{D}_{\hat{\boldsymbol{m_2}}} ^ {(Y)}} ^ {-1}, ..., {\boldsymbol{D}_{\hat{\boldsymbol{m_t}}} ^ {(Y)}} ^ {-1}, ..., {\boldsymbol{D}_{\hat{\boldsymbol{m_T}}} ^ {(Y)}} ^ {-1}]
    \end{gathered}
//}

=== 変換処理の実装

導出結果に基づいて、変換処理の実装を行う。
具体的には、学習処理の項でも述べたように、GMMMapクラスを継承したTrajectoryGMMMapクラスを実装する。

まず、コンストラクタにて、変換元のデータと独立で求められる@<m>{\boldsymbol{D\}_m ^ {(Y)\}}の計算処理を行う。

//emlist[行列Dの算出][python]{
    def __init__(self, gmm, swap = False):
        # GMMMapのコンストラクタを呼び出す
        super(TrajectoryGMMMap, self).__init__(gmm, swap)

        D = gmm.means_.shape[1] / 2

        # すべてのmについてP(Y|X)における分散共分散行列Dをまとめて1つの変数として扱う
        self.D = np.zeros((self.M, D, D))
        for m in range(self.M):
            xx_inv_xy = np.linalg.solve(self.covarXX[m], self.covarXY[m])
            self.D[m] = self.covarYY[m] - np.dot(self.covarYX[m], xx_inv_xy)
//}

次に、変換処理を実装するにあたり、@<m>{\boldsymbol{y\}}から@<m>{\boldsymbol{Y\}}への変換行列@<m>{\boldsymbol{W\}}を生成するメソッドを実装する。
@<m>{\boldsymbol{W\}}の生成は、@<m>{T}回のループで@<m>{\boldsymbol{w\}_t ^ {(0)\}, \boldsymbol{w\}_t ^ {(1)\}}を生成し、つなげていくことで行っている。

また、ここで注意すべきなのはscipy.sparseモジュールを使うことで高速化を図っているという点である。
scipy.sparseは疎行列を扱うためのモジュールで、@<m>{\boldsymbol{W\}}は@<img>{w-matrix}を見れば分かるように
対角成分付近以外はほとんどが0の疎行列なので、これを用いることで計算量や必要なメモリを減らすことができる。

scipy.sparseモジュールには幾つかの疎行列の実装があるが、ここではlil_matrixとcsr_matrixの2つを用いている。
lil_matrixではインデックスを指定してデータを割り当てができる一方で、行列に対する計算操作はcsr_matrixの方が高速なので、
lil_matrixで行列の生成をした後にcsr_matrixへの変換を行っている。

//emlist[行列Wの算出][python]{
    def __construct_weight_matrix(self, T, D):
        W = None

        for t in range(T):
            # 図の各行に対応する行列を生成する
            w0 = scipy.sparse.lil_matrix((D, D * T))
            w1 = scipy.sparse.lil_matrix((D, D * T))

            # scipy.sparse.diagsを使って図の「1」のマスに該当する部分の対角成分に1を代入する
            w0[0:, t * D: (t + 1) * D] = scipy.sparse.diags(np.ones(D), 0)

            # 図の「-0.5」のマスに該当する部分の対角成分に-0.5を代入する
            tmp = np.zeros(D).fill(-0.5)
            if t == 0:
                w1[0:, :D] = scipy.sparse.diags(tmp, 0)
            else:
                # t == 0でない場合はt - 1番目のマスに代入する
                w1[0:, (t - 1) * D: t * D] = scipy.sparse.diags(tmp, 0)

            # 図の「0.5」のマスに該当する部分の対角成分に0.5を代入する
            tmp = np.zeros(D).fill(-0.5)
            if t == T - 1:
                w1[0:, t * D:] = scipy.sparse.diags(tmp, 0)
            else:
                # t == 1でない場合はt + 1番目のマスに代入する
                w1[0:, (t + 1) * D: (t + 2) * D] = scipy.sparse.diags(tmp, 0)

            # w0とw1を結合したものを積み重ねていく
            W_t = scipy.sparse.vstack([w0, w1])
            if W == None:
                W = W_t
            else:
                W = scipy.sparse.vstack([W, W_t])

        # 最後にlil_matrixをcsr_matrixへと変換する
        return W.tocsr()
//}

最後に、変換処理本体を実装する。
まず、変換行列@<m>{\boldsymbol{W\}}を生成した後に、@<m>{\hat{\boldsymbol{m\}\}}を求め、
それに基づいて@<m>{\boldsymbol{E\}_{\hat{\boldsymbol{m\}\}\} ^ {(Y)\}, {\boldsymbol{D\}_{\hat{\boldsymbol{m\}\}\} ^ {(Y)\}\} ^ {-1\}}を計算する。
そして、求めた行列の積をscipy.sparse.linalg.spsolveによって計算し、特徴量を求める。

注意すべき点の1つは、準最適な分布系列@<m>{\hat{\boldsymbol{m\}\}}を求める際に、sklearn.mixture.GMMのpredictメソッドを用いているということである。
@<m>{\hat{m\}_t}は@<m>{t}番目のフレームにおける、最も事後確率が高いガウス分布のインデックスと同義なので、predictメソッドが返すラベルをそのまま用いることができるのだ。

//emlist[変換処理][python]{
    def convert(self, src):
        T, D = src.shape[0], src.shape[1] / 2
        W = self.__construct_weight_matrix(T, D)

        # 準最適な分布系列を求める
        optimum_mix = self.px.predict(src)

        # 行列Eを用意する
        E = np.zeros((T, D * 2))
        for t in range(T):
            m = optimum_mix[t]
            # フレームごとにEtを代入する
            xx = np.linalg.solve(self.covarXX[m], src[t] - self.src_means[m])
            E[t] = self.tgt_means[m] + np.dot(self.covarYX[m], xx)
        E = E.flatten()

        # コンストラクタで計算したself.Dを元にD^-1の対角要素を計算する
        D_inv = np.zeros((T, D * 2, D * 2))
        for t in range(T):
            m = optimum_mix[t]
            # フレームごとに分布系列に対応するself.Dの逆行列を代入する
            D_inv[t] = np.linalg.inv(self.D[m])
        # 計算した要素を対角成分とする
        D_inv = scipy.sparse.block_diag(D_inv, format = 'csr')

        # 計算した行列を用いて変換後の特徴量を求める
        mutual = W.T.dot(D_inv)
        covar = mutual.dot(W)
        mean = mutual.dot(E)
        # numpy.linalg.solveに対応する疎行列向けのメソッドを使う
        y = scipy.sparse.linalg.spsolve(covar, mean, use_umfpack = False)

        return y.reshape((T, D))
//}

以上で変換処理が実装できる。
念のため、ほぼ情報量はないが、TrajectoryGMMMapの実装全体を載せておく。

//listnum[trajectory][trajectory.py][python]{
#!/usr/bin/python
# coding: utf-8

import numpy as np

from sklearn.mixture import GMM

import scipy.sparse
import scipy.sparse.linalg

from gmmmap import GMMMap

class TrajectoryGMMMap(GMMMap):
    def __init__(self, gmm, swap = False):
        <省略>

    def __construct_weight_matrix(self, T, D):
        <省略>

    def convert(self, src):
        <省略>
//}

=== STFファイルへの変換

変換処理を呼び出し、結果をSTFファイルに保存するスクリプトは前章とほとんど変わらない。
MFCCの動的変化量を結合するのと、フレームごとに呼び出していた変換処理をまとめて呼び出すように変更するのみである。

以下に、前章のスクリプトとトラジェクトリベースな変換のためのスクリプトのdiffを載せておく。

//listnum[convert_trajectory][convert_gmmmap.pyとconvert_trajectory.pyのdiff][python]{
9c9
< from gmmmap import GMMMap
---
> from trajectory import TrajectoryGMMMap
38c38
<     source_data = numpy.array([mfcc.mfcc(source.SPEC[frame]) \
---
>     source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) \
39a40
>     source_data = numpy.hstack([source_mfcc, mfcc.delta(source_mfcc)])
41,42c42
<     output_mfcc = numpy.array([gmmmap.convert(source_data[frame])[0] \
<                                         for frame in xrange(source_data.shape[0])])
---
>     output_mfcc = gmmmap.convert(source_data)
//}

== トラジェクトリベースな変換処理の結果

以下の画像は、トラジェクトリベースな変換の結果を表したグラフである。
前章と同様に、それぞれのグラフはMFCCの第1次係数(数値が小さい方)及び第2次係数(数値が大きい方)の推移を表しており、
上から、変換先話者のデータ、トラジェクトリベースな変換を行ったデータ、GMMによるフレームごとに独立な変換処理を行ったデータをDTWによって伸縮させたものである。
中段のグラフの方が、下段よりなめらかに推移し、より上段の変換先話者のデータに近づいていることが分かるだろう。

//image[trajectory-result][トラジェクトリベースな変換とフレームごとに独立な変換による結果データの比較][scale=0.6]

このように、前後フレームとのつながりを考えた変換手法を採用することで、
より自然な変換音声を生成することができる。
