= 固有声に基づく多対多声質変換

ここまで、変換元と変換先の話者のパラレルデータが存在していることを前提に、一対一の変換モデルを学習していく仕組みを紹介してきたが、
ここからは、固有声(Eigenvoice)という概念に基づいた、変換元と変換先の話者のパラレルデータがなくても変換できるような仕組みを紹介し、実装していく。
まずは、一対多の変換、つまりある特定の人の声質を任意の相手の声質に変換できる仕組みについて説明した後に、
多対一の変換、つまり任意の人の声質を特定の相手の声質に変換できる仕組みについて説明し、その2つを組み合わせることで多対多の変換を実現するものとする。

固有声による声質変換については「Eigenvoice Conversion Based on Gaussian Mixture Model@<bib>{evgmm}」を参照した。

== 固有声の導入

固有声とは、顔画像認識で用いられている固有顔という概念を元にしたものある。
固有顔とは、顔画像を主成分分析した際に得られる固有ベクトルを指し、
様々な顔画像の、この固有ベクトルで形成される部分空間において類似度を取ることによって、低計算量かつ高精度で顔認識を実現することができる。
これと同様に、声質変換においても、主成分分析を導入することによって一対多及び多対一の変換を実現する。
ただし、ここで主成分分析の対象とするのは、音声の特徴量ではなく、GMMの平均ベクトルであるという大きな違いがある。

ここからは、まず一対多の声質変換について説明する。
一対多の場合は、モデル構築のために変換元の話者と第三者(以降、「事前学習用出力話者」という)のパラレルデータが多数用意されている状況を仮定する。
このとき、変換元話者と事前学習用出力話者のそれぞれとの間で、共通の重みと分散共分散行列を用いるという制約の下、GMMに学習させて平均ベクトルを求めておく。
そして、この平均ベクトルの出力成分に対して主成分分析をすることで固有ベクトル、つまり、いわゆる固有声を求めることができる。
すると、任意の話者に対しても、この固有声空間に射影してやれば、変換に用いる平均ベクトルを求められるようになる。

つまり、変換元話者と@<m>{s}番目の事前学習用出力話者とのパラレルデータから学習したGMMの@<m>{i}番目のガウス分布における平均ベクトルを
@<m>{\boldsymbol{\mu\}_i (s) = [{\boldsymbol{\mu\}_i ^ {(x)\} (s)\} ^ \top, {\boldsymbol{\mu\}_i ^ {(y)\} (s)\} ^ \top] ^ \top}とすると、
@<m>{\boldsymbol{\mu\}_i ^ {(y)\} (s)}に対して主成分分析を行うことで以下のように表すことができる。

//texequation{
    \boldsymbol{\mu}_i ^ {(y)} (s) \simeq \boldsymbol{B}_i \boldsymbol{w} ^ {(s)} + \boldsymbol{b}_i ^ {(0)}
//}

ここで、@<m>{\boldsymbol{b\}_i ^ {(0)\}}がバイアスベクトル、@<m>{\boldsymbol{B\}_i = [\boldsymbol{b\}_{i, 1\}, \boldsymbol{b\}_{i, 2\}, ..., \boldsymbol{b\}_{i, J\}]}が固有ベクトルである。
この@<m>{\boldsymbol{B\}_i}で張られる部分空間においては、@<m>{\boldsymbol{\mu\}_i ^ {(y)\} (s)}はJ次元の重みベクトル@<m>{\boldsymbol{w\}_i ^ {(s)\} = [w_{i, 1\}, w_{i, 2\}, ..., w_{i, J\}] ^ \top}によって表される。
そして、変換先話者に対応する重みベクトル@<m>{\boldsymbol{w\} ^ {(tar)\}}を推定することができれば、平均ベクトル@<m>{\boldsymbol{\mu\}_i (tar)}を得ることができ、
GMMを用いて特徴量の変換ができる。

ここまでの説明を図で表すと以下の通りとなる。
これを見ると、固有声に基づく変換は事前学習処理と話者適応処理の2つからなることが分かるだろう。

//image[evgmm][固有声を用いた変換の仕組み][scale=0.35]

以上が、一対多の声質変換の基本的な仕組みである。
多対一の場合は、GMMの学習の際に変換元話者と事前学習用出力話者を入れ替えた上で、平均ベクトルの入力成分に対して主成分分析してやれば同様に実現することができる。


== 固有声による声質変換の事前学習

まず、固有声による声質変換(以降、「固有声GMM」という)の事前学習処理について説明する。
事前学習処理は、以下の3つのステップからなる。

 1. すべての事前学習用出力話者とのGMMで共通に用いられる重みと分散共分散行列の推定
 2. それぞれの事前学習用出力話者に対する平均ベクトルの推定
 3. 主成分分析による固有ベクトルとバイアスベクトルの決定

具体的な説明にあたって、変換元話者の特徴量を@<m>{\boldsymbol{X\}_t}、@<m>{s}番目の事前学習用出力話者の特徴量を@<m>{\boldsymbol{Y\}_t ^ {(s)\}}、
それを結合したものを@<m>{\boldsymbol{Z\}_t ^ {(s)\} = [\boldsymbol{X\}_t ^ \top, {\boldsymbol{Y\}_t ^ {(s)\}\} ^ \top] ^ \top}とし、
事前学習用出力話者は@<m>{S}人いるものとする。

まず、重みと分散共分散行列の推定を行う。この際は、すべての事前学習用出力話者の特徴量に対して尤度が最大となるように学習してやればよい。
つまり、推定されるパラメータを@<m>{\boldsymbol{\lambda\} ^ {(0)\}}とすると、以下の通りである。

//texequation{
    \hat{\boldsymbol{\lambda}} ^ {(0)} = \underset{\boldsymbol{\lambda}}{argmax} \prod_{s = 1}^{S} \prod_{t = 1}^{T} P(\boldsymbol{Z}_t ^ {(s)} | \boldsymbol{\lambda})
//}

ここで、@<m>{T}は学習データのフレーム数を表している。

次に、それぞれの事前学習用出力話者に対する平均ベクトルの推定を行う。ここで注意すべきなのは、@<m>{\boldsymbol{\lambda\} ^ {(0)\}}の重みと分散共分散行列はそのままで、平均ベクトルのみ更新するという点である。

//texequation{
    \hat{\boldsymbol{\lambda}} ^ {(s)} = \underset{\boldsymbol{\lambda}}{argmax} \prod_{t = 1}^{T} P(\boldsymbol{Z}_t ^ {(s)} | \boldsymbol{\lambda} ^ {(s)})
//}

最後に、主成分分析によって、固有ベクトルとバイアスベクトルの推定を行う。まず、それぞれの事前学習用出力話者に対して@<m>{\boldsymbol{\lambda\} ^ {(s)\}}の出力平均ベクトルを@<m>{\boldsymbol{\mu\}_i ^ {(Y)\} (s)}として、
@<m>{2DM}次元のスーパーベクトル@<m>{SV ^ {(s)\} = [{\boldsymbol{\mu\}_1 ^ {(Y)\} (s)\} ^ \top, {\boldsymbol{\mu\}_2 ^ {(Y)\} (s)\} ^ \top, ..., {\boldsymbol{\mu\}_M ^ {(Y)\} (s)\} ^ \top] ^ \top}を求める。
そして、全出力話者のスーパーベクトルに対して主成分分析を行うことによって、バイアスベクトル@<m>{\boldsymbol{b\}_i ^ {(0)\}}及び固有ベクトル@<m>{\boldsymbol{B\}_i}を決定する。
このとき、@<m>{SV ^ {(s)\}}は以下のように表すことができる。

//texequation{
    \begin{split}
        SV ^ {(s)} & \simeq [\boldsymbol{B}_1 ^ \top, \boldsymbol{B}_2 ^ \top, ..., \boldsymbol{B}_M ^ \top] ^ \top \boldsymbol{w} ^ {(s)} + [\boldsymbol{b}_1 ^ {(0)}, \boldsymbol{b}_2 ^ {(0)}, ..., \boldsymbol{b}_M ^ {(0)}] \\
                   & where \ \boldsymbol{b}_i ^ {(0)} = \frac{1}{S} \sum_{s = 1}^{S} \boldsymbol{\mu}_i ^ {(Y)} (s)
    \end{split}
//}

=== 学習データの構築処理

固有声GMMについても、特徴量を受け取って変換処理を担うクラスと、与えられたSTFファイルのリストから特徴量を抽出して学習データを構築し、事前学習処理を呼び出す部分、
そして保存された学習済みインスタンスを読み込んで変換処理を呼び出し、結果をSTFファイルとして保存する部分の3つに分けて実装する。

まずは、学習データの構築と学習処理の呼び出しを実装する。今回は、一対多及び多対一の両方に対応できるようにする。
はじめに、変換元話者と事前学習用出力話者のリストを受け取って、一対多の学習データを構築する関数を実装する。
事前学習用出力話者の数だけ繰り返している以外はこれまでの学習処理と大きな差はないはずである。

//emlist[一対多声質変換のための学習データの構築][python]{
def one_to_many(source_list, target_list, dtw_cache):
    source_mfcc = []

    for i in xrange(len(source_list)):
        source = STF()
        source.loadfile(source_list[i])

        mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = D)
        source_mfcc.append(numpy.array([mfcc.mfcc(source.SPEC[frame]) \
                                    for frame in xrange(source.SPEC.shape[0])]))

    total_data = []

    for i in xrange(len(target_list)):
        learn_data = None

        for j in xrange(len(target_list[i])):
            print i, j

            target = STF()
            target.loadfile(target_list[i][j])

            mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency, dimension = D)
            target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) \
                                        for frame in xrange(target.SPEC.shape[0])])

            cache_path = os.path.join(dtw_cache, '%s_%s.dtw' % \
                tuple(map(lambda x: re.sub('[./]', '_', re.sub('^[./]*', '', x)), \
                                                [source_list[j], target_list[i][j]])))
            if os.path.exists(cache_path):
                dtw = pickle.load(open(cache_path))
            else:
                dtw = DTW(source_mfcc[j], target_mfcc, \
                    window = abs(source_mfcc[j].shape[0] - target_mfcc.shape[0]) * 2)
                with open(cache_path, 'wb') as output:
                    pickle.dump(dtw, output)

            warp_data = dtw.align(target_mfcc, reverse = True)

            data = numpy.hstack([source_mfcc[j], warp_data])
            if learn_data is None:
                learn_data = data
            else:
                learn_data = numpy.vstack([learn_data, data])

        total_data.append(learn_data)

    return total_data
//}

次に、多対一の場合の処理を実装する。

//emlist[多対一声質変換のための学習データの構築][python]{
def many_to_one(source_list, target_list, dtw_cache):
    target_mfcc = []

    for i in xrange(len(target_list)):
        target = STF()
        target.loadfile(target_list[i])

        mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency, dimension = D)
        target_mfcc.append(numpy.array([mfcc.mfcc(target.SPEC[frame]) \
                                    for frame in xrange(target.SPEC.shape[0])]))

    total_data = []

    for i in xrange(len(source_list)):
        learn_data = None

        for j in xrange(len(source_list[i])):
            print i, j

            source = STF()
            source.loadfile(source_list[i][j])

            mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = D)
            source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) \
                                        for frame in xrange(source.SPEC.shape[0])])

            cache_path = os.path.join(sys.argv[3], '%s_%s.dtw' % \
                tuple(map(lambda x: re.sub('[./]', '_', re.sub('^[./]*', '', x)), \
                                                [source_list[i][j], target_list[j]])))
            if os.path.exists(cache_path):
                dtw = pickle.load(open(cache_path))
            else:
                dtw = DTW(source_mfcc, target_mfcc[j], \
                    window = abs(source_mfcc.shape[0] - target_mfcc[j].shape[0]) * 2)
                with open(cache_path, 'wb') as output:
                    pickle.dump(dtw, output)

            warp_data = dtw.align(source_mfcc)

            data = numpy.hstack([warp_data, target_mfcc[j]])
            if learn_data is None:
                learn_data = data
            else:
                learn_data = numpy.vstack([learn_data, data])

        total_data.append(learn_data)

    return total_data
//}

最後に、実行時引数として与えられた話者データのリストから、一対多か多対一かどうかを判定し、
学習データを構築した後にEVGMMクラスを生成する部分を実装する。
学習データの構築の部分と合わせると、以下のようになる。

//listnum[learn_evgmm][learn_evgmm.py][python]{
#!/usr/bin/env python
# coding: utf-8

from stf import STF
from mfcc import MFCC
from dtw import DTW
from evgmm import EVGMM

import numpy
import os
import pickle
import re
import sys

D = 16

def one_to_many(source_list, target_list, dtw_cache):
    <省略>

def many_to_one(source_list, target_list, dtw_cache):
    <省略>

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Usage: %s [list of source stf] [list of target] ' + \
                    '[dtw cache directory] [output file]' % sys.argv[0]
        sys.exit()

    source_list = open(sys.argv[1]).read().strip().split('\n')
    target_list = open(sys.argv[2]).read().strip().split('\n')

    if len(filter(lambda s: not s.endswith('.stf'), source_list)) == 0:
        target_list = [open(target).read().strip().split('\n') \
                                                for target in target_list]
        total_data = one_to_many(source_list, target_list, sys.argv[3])
        evgmm = EVGMM(total_data)
    elif len(filter(lambda s: not s.endswith('.stf'), target_list)) == 0:
        source_list = [open(source).read().strip().split('\n') \
                                                for source in source_list]
        total_data = many_to_one(source_list, target_list, sys.argv[3])
        evgmm = EVGMM(total_data, True)

    with open(sys.argv[4], 'wb') as output:
        pickle.dump(evgmm, output)
//}

=== 事前学習処理の実装

続いて、変換のためのクラスEVGMMを実装する。このコンストラクタによって事前学習を行う。

ここで注意すべきなのは、それぞれの事前学習用出力話者に対する平均ベクトルを推定する際に、平均ベクトルのみを更新するよう
GMMのコンストラクタに@<code>{init_params = ''}と@<code>{params = 'm'}を指定しているという点である。
@<code>{init_params = ''}とすることで、初期化の際にすでに代入した@<m>{\boldsymbol{\lambda\} ^ {(0)\}}のパラメータを上書きしないように設定し、
@<code>{params = 'm'}とすることで、学習の際に平均ベクトルのみを更新するように設定することができるのである。

また、主成分分析にはsklearn.decomposition.PCAを用いている。

//emlist[固有声GMMの事前学習処理][python]{
    # 学習データは[[Xt, Yt(1)], [Xt, Yt(2)], ..., [Xt, Yt(S)]]のS個の要素を持つリスト
    def __init__(self, learn_data, swap = False):
        S = len(learn_data)
        D = learn_data[0].shape[1] / 2

        # すべての学習データについてパラメータを推定する
        initial_gmm = GMM(n_components = M, covariance_type = 'full')
        initial_gmm.fit(np.vstack(learn_data))

        # λ(0)から得たパラメータを保存しておく
        self.weights = initial_gmm.weights_
        self.source_means = initial_gmm.means_[:, :D]
        self.target_means = initial_gmm.means_[:, D:]
        self.covarXX = initial_gmm.covars_[:, :D, :D]
        self.covarXY = initial_gmm.covars_[:, :D, D:]
        self.covarYX = initial_gmm.covars_[:, D:, :D]
        self.covarYY = initial_gmm.covars_[:, D:, D:]

        # スーパーベクトルはすべての出力話者についてまとめてS * 2DM次元の行列とする
        sv = []

        # 各出力話者について平均ベクトルを推定する
        for i in xrange(S):
            # 平均ベクトル以外は更新しないように設定する
            gmm = GMM(n_components = M, params = 'm', init_params = '', \
                                                    covariance_type = 'full')
            gmm.weights_ = initial_gmm.weights_
            gmm.means_ = initial_gmm.means_
            gmm.covars_ = initial_gmm.covars_
            gmm.fit(learn_data[i])

            # 平均ベクトルを結合したスーパーベクトルを更新する
            sv.append(gmm.means_)

        sv = np.array(sv)

        # スーパーベクトルの入力平均ベクトルにあたる部分を主成分分析にかける
        source_pca = PCA()
        source_pca.fit(sv[:, :, :D].reshape((S, M * D)))

        # スーパーベクトルの出力平均ベクトルにあたる部分を主成分分析にかける
        target_pca = PCA()
        target_pca.fit(sv[:, :, D:].reshape((S, M * D)))

        # 入力平均ベクトルと出力平均ベクトルに対する固有ベクトルのタプル
        self.eigenvectors = source_pca.components_.reshape((M, D, S)), \
                                    target_pca.components_.reshape((M, D, S))
        # 入力平均ベクトルと出力平均ベクトルに対するバイアスベクトルのタプル
        self.biasvectors = source_pca.mean_.reshape((M, D)), \
                                    target_pca.mean_.reshape((M, D))

        # 話者適応の際に更新するようの平均ベクトルの変数を用意しておく
        self.fitted_source = self.source_means
        self.fitted_target = self.target_means

        self.swap = swap
//}

== 固有声による声質変換の話者適応処理

固有声GMMにおける話者適応処理は、変換先話者の特徴量から固有声の部分空間における重みベクトル@<m>{\boldsymbol{w\} ^ {(tar)\}}を求めることによって行われる。
つまり、話者適応によって得られるGMMの変換パラメータを@<m>{\boldsymbol{\lambda\} ^ {(tar)\}}とすると、その出力平均ベクトル@<m>{\boldsymbol{\mu\}_i ^ {(X)\} (tar)}は以下のように表すことができる。

//texequation{
    \boldsymbol{\mu}_i ^ {(X)} (tar) = \boldsymbol{B}_i w_i ^ {(tar)} + \boldsymbol{b}_i ^ {(0)}
//}

このとき、変換先話者の特徴量を@<m>{\boldsymbol{Y\} ^ {(tar)\}}として、求める重みベクトル@<m>{\boldsymbol{w\} ^ {(tar)\}}は以下のように表される。

//texequation{
    \begin{split}
        \hat{\boldsymbol{w}} ^ {(tar)} = & \ \underset{\boldsymbol{w}}{argmax} \int P([\boldsymbol{X} ^ \top, {\boldsymbol{Y} ^ {(tar)}} ^ \top] ^ \top, \boldsymbol{\lambda} ^ {(tar)}) d \boldsymbol{X} \\
                                      = & \ \underset{\boldsymbol{w}}{argmax} \ P(\boldsymbol{Y} ^ {(tar)} | \boldsymbol{\lambda} ^ {(tar)})
    \end{split}
//}

ここで、確率密度関数@<m>{P(\boldsymbol{Y\} ^ {(tar)\} | \boldsymbol{\lambda\} ^ {(tar)\})}はGMMでモデル化されているので、EMアルゴリズムに基づき、以下のQ関数を最大化することで求められる。

//texequation{
    \mathcal{Q} (\boldsymbol{w} ^ {(tar)}, \hat{\boldsymbol{w}} ^ {(tar)}) = \sum_{all \ \boldsymbol{m}} P(\boldsymbol{m} | \boldsymbol{Y} ^ {(tar)}, \boldsymbol{\lambda} ^ {(tar)}) \ log \ P(\boldsymbol{Y} ^ {(tar)}, \boldsymbol{m} | \hat{\boldsymbol{\lambda}} ^ {(tar)})
//}

このとき、@<m>{\hat{\boldsymbol{w\}\} ^ {(tar)\}}は以下のように求められる。

//texequation{
    \hat{\boldsymbol{w}} ^ {(tar)} = \Biggl\{ \sum_{i = 1}^{M} \overline{\gamma}_i ^ {(tar)} \boldsymbol{B}_i ^ \top {\boldsymbol{\Sigma}_i ^ {(yy)}} ^ {-1} \boldsymbol{B}_i \Biggr\} ^ {-1} \sum_{m = 1}^{M} \boldsymbol{B}_i ^ \top {\boldsymbol{\Sigma}_i ^ {(yy)}} ^ {-1} \overline{\boldsymbol{Y}}_i ^ {(tar)}
//}

ただし、@<m>{\overline{\gamma\}_i ^ {(tar)\}, \overline{\boldsymbol{Y\}\}_i ^ {(tar)\}}は以下の通りである。

//texequation{
    \begin{split}
                \overline{\gamma}_i ^ {(tar)} & = \sum_{t = 1}^{T} P(m_i | \boldsymbol{Y}_t ^ {(tar)}, \boldsymbol{\lambda} ^ {(tar)}) \\
        \overline{\boldsymbol{Y}}_i ^ {(tar)} & = \sum_{t = 1}^{T} P(m_i | \boldsymbol{Y}_t ^ {(tar)}, \boldsymbol{\lambda} ^ {(tar)}) (\boldsymbol{Y}_t ^ {(tar)} - \boldsymbol{b}_i ^ {(0)})
    \end{split}
//}

以上を繰り返すことによって、@<m>{\hat{\boldsymbol{w\}\} ^ {(tar)\}}が求められる。ただし、@<m>{\boldsymbol{\lambda\} ^ {(tar)\}}の初期パラメータとしては@<m>{\boldsymbol{\lambda\} ^ {(0)\}}を用いるものとする。

=== 話者適応処理の実装

話者適応処理はEVGMMのメソッドとして実装し、コンストラクタで求めた事前学習パラメータと引数として与えられる特徴量から適応済みパラメータを求めることとする。
まずは、一対多の声質変換に関する適応処理を実装する。

//emlist[一対多の声質変換に関する話者適応処理][python]{
    def fit_target(self, target, epoch):
        # P(m|Y)を算出するためのGMMインスタンスを生成する
        py = GMM(n_components = M, covariance_type = 'full')
        py.weights_ = self.weights
        py.means_ = self.target_means
        py.covars_ = self.covarYY

        for x in xrange(epoch):
            # P(m|Y)を算出する
            predict = py.predict_proba(np.atleast_2d(target))
            # Yを算出する
            y = np.sum([predict[:, i: i + 1] * (target - self.biasvectors[1][i]) \
                                                        for i in xrange(M)], axis = 1)
            # γを算出する
            gamma = np.sum(predict, axis = 0)

            # 重みベクトルwを算出する
            left = np.sum([gamma[i] * np.dot(self.eigenvectors[1][i].T, \
                            np.linalg.solve(py.covars_, self.eigenvectors[1])[i]) \
                                                        for i in xrange(M)], axis = 0)
            right = np.sum([np.dot(self.eigenvectors[1][i].T, \
                    np.linalg.solve(py.covars_, y)[i]) for i in xrange(M)], axis = 0)
            weight = np.linalg.solve(left, right)

            # 重みベクトルから平均出力ベクトルを求め、GMMのパラメータを更新する
            self.fitted_target = np.dot(self.eigenvectors[1], weight) \
                                                                + self.biasvectors[1]
            py.means_ = self.fitted_target
//}

同様にして、多対一の声質変換に関する適応処理を実装する。用いる固有ベクトルとバイアスベクトルを平均入力ベクトルによるものにすればよい。

//emlist[多対一の声質変換に関する話者適応処理][python]{
    def fit_source(self, source, epoch):
        # P(m|X)を算出するためのGMMインスタンスを生成する
        px = GMM(n_components = M, covariance_type = 'full')
        px.weights_ = self.weights
        px.means_ = self.source_means
        px.covars_ = self.covarXX

        for x in xrange(epoch):
            # P(m|X)を算出する
            predict = px.predict_proba(np.atleast_2d(source))
            # Xを算出する
            x = np.sum([predict[:, i: i + 1] * (source - self.biasvectors[0][i]) \
                                                        for i in xrange(M)], axis = 1)
            # γを算出する
            gamma = np.sum(predict, axis = 0)

            # 重みベクトルwを算出する
            left = np.sum([gamma[i] * np.dot(self.eigenvectors[0][i].T, \
                            np.linalg.solve(px.covars_, self.eigenvectors[0])[i]) \
                                                        for i in xrange(M)], axis = 0)
            right = np.sum([np.dot(self.eigenvectors[0][i].T, \
                    np.linalg.solve(px.covars_, x)[i]) for i in xrange(M)], axis = 0)
            weight = np.linalg.solve(left, right)

            # 重みベクトルから平均入力ベクトルを求め、GMMのパラメータを更新する
            self.fitted_source = np.dot(self.eigenvectors[0], weight) \
                                                                + self.biasvectors[0]
            px.means_ = self.fitted_source
//}

== 固有声による声質変換処理

ここまでで、変換に必要なパラメータをすべて求めることができたので、変換処理を実装する。
ここで注意したいのは、EVGMMで新たに実装した処理は変換のパラメータを求めるための処理であり、
パラメータが求まった後の変換処理はこれまでと全く変わらないという点である。
変換処理は、一対一の声質変換と同様に実装すればよい。

=== 変換処理の実装

変換処理の実装は、第2章で紹介した実装とほとんど同じである。
ただし、@<m>{\boldsymbol{E\}_{m, t\}}を求める際に、平均ベクトルとして話者適応処理で得られたものを使う必要があるので、
その部分のみ変更を加えている。

ここまでの実装と合わせて、EVGMMクラス全体を載せておく。

//emlist[EVGMMの実装][python]{
class EVGMM(object):
    def __init__(self, learn_data, swap = False):
        <省略>

    def fit(self, data, epoch = 1000):
        if self.swap:
            self.fit_source(data, epoch)
        else:
            self.fit_target(data, epoch)

    def fit_source(self, source, epoch):
        <省略>

    def fit_target(self, target, epoch):
        <省略>

    def convert(self, source):
        D = source.shape[0]

        # 話者適応処理で得たパラメータからEを算出する
        E = np.zeros((M, D))
        for m in xrange(M):
            xx = np.linalg.solve(self.covarXX[m], source - self.fitted_source[m])
            E[m] = self.fitted_target[m] + np.dot(self.covarYX[m], xx)

        px = GMM(n_components = M, covariance_type = 'full')
        px.weights_ = self.weights
        px.means_ = self.source_means
        px.covars_ = self.covarXX

        posterior = px.predict_proba(np.atleast_2d(source))
        return np.dot(posterior, E)
//}

=== トラジェクトリベースの変換処理の実装

先ほど説明したように、EVGMMにおける変換処理は第2章で扱った変換手法とほとんど同じで、
相違点は、平均ベクトルの代わりに話者適応で得られたパラメータを用いているのみである。
つまり、学習や変換に用いる特徴量に動的特徴量を結合してやれば、
全く同じようにトラジェクトリベースの声質変換を適用することができる。

以下に、EVGMMクラスを継承して、トラジェクトリベースの声質変換を行うTrajectoryEVGMMクラスの実装を載せる。
ただし、convertメソッドは@<m>{\boldsymbol{E\}_{m, t\}}の導出以外は前章と変わらないので、該当部分以外は省略している。

EVGMMとTrajectoryGMMの実装を合わせたevgmm.pyは以下の通りである。

//listnum[evgmm][evgmm.py][python]{
#!/usr/bin/env python
# coding: utf-8

import numpy as np

from sklearn.decomposition import PCA
from sklearn.mixture import GMM

import scipy.sparse
import scipy.sparse.linalg

M = 32

class EVGMM(object):
    <省略>

class TrajectoryEVGMM(EVGMM):
    def __construct_weight_matrix(self, T, D):
        <省略>
        
    def convert(self, src):
        <省略>

        # 話者適応処理で得たパラメータからEを算出する
        E = np.zeros((T, D * 2))
        for t in range(T):
            m = optimum_mix[t]
            xx = np.linalg.solve(self.covarXX[m], src[t] - self.fitted_source[m])
            E[t] = self.fitted_target[m] + np.dot(self.covarYX[m], xx)
        E = E.flatten()

        <省略>
        return y.reshape((T, D))
//}

最後に、変換先話者のSTFファイルを読み込んで話者適応をした後に、変換元話者の読み込んで変換処理を行い、
結果をSTFファイルとして保存するという部分を実装する。
ここでは、一対多か多対一かを実行時引数の数で判定している。というのも、一対多の場合は話者適応のために変換先話者の特徴量が必要だが、
多対一の場合は変換に用いる変換元話者の特徴量で話者適応も行うことができるからである。

//listnum[convert_trajevgmm][convert_trajevgmm.py][python]{
#!/usr/bin/env python

import math
import numpy
import pickle
import sklearn
import sys

from evgmm import GMM

from stf import STF
from mfcc import MFCC
from dtw import DTW

D = 16

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print 'Usage: %s [gmmmap] [f0] [source speaker stf] ' + \
                                (target speaker stf) [output]' % sys.argv[0]
        sys.exit()
    
    with open(sys.argv[1], 'rb') as infile:
        evgmm = pickle.load(infile)

    with open(sys.argv[2], 'rb') as infile:
        f0 = pickle.load(infile)

    source = STF()
    source.loadfile(sys.argv[3])

    mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency, dimension = D)
    source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) \
                                    for frame in xrange(source.SPEC.shape[0])])
    source_data = numpy.hstack([source_mfcc, mfcc.delta(source_mfcc)])

    if len(sys.argv) == 5:
        evgmm.fit(source_data)
    else:
        target = STF()
        target.loadfile(sys.argv[4])

        mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency, dimension = D)
        target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) \
                                    for frame in xrange(target.SPEC.shape[0])])
        target_data = numpy.hstack([target_mfcc, mfcc.delta(target_mfcc)])

        evgmm.fit(target_data)

    output_mfcc = evgmm.convert(source_data)
    output_spec = numpy.array([mfcc.imfcc(output_mfcc[frame]) \
                                    for frame in xrange(output_mfcc.shape[0])])

    source.SPEC = output_spec
    source.F0[source.F0 != 0] = numpy.exp((numpy.log(source.F0[source.F0 != 0]) - \
                                        f0[0][0]) * f0[1][1] / f0[1][0] + f0[0][1])

    if len(sys.argv) == 5:
        source.savefile(sys.argv[4])
    else:
        source.savefile(sys.argv[5])
//}

== 固有声に基づく多対多声質変換

ここまでで扱った多対一と一対多の声質変換を組み合わせることによって、多対多の声質変換を実現することができる。
つまり、任意の話者の発話データを多対一変換にて特定の話者に変換し、一対多変換によってその特定の話者から任意の話者へと変換することができる。
この仕組みを図で表すと以下のようになる。

//image[ref-evgmm][多対一変換及び一対多変換を組み合わせた多対多声質変換の仕組み(@<bib>{ref-evgmm}より引用)][scale=0.29]

紙面と締め切りの都合上、詳細な説明は割愛するが、詳しくは「Many-to-many eigenvoice conversion with reference voice@<bib>{ref-evgmm}」という論文を参照してほしい。

== 固有声に基づく声質変換の変換処理の結果

以下の画像は、固有声に基づく多対一の声質変換の結果を表したグラフである。
前章と同様に、それぞれのグラフはMFCCの第1次係数(数値が小さい方)及び第2次係数(数値が大きい方)の推移を表しており、
上から、変換先話者のデータ、変換元話者と変換先話者のパラレルデータを用いずに固有声に基づく多対一変換を行ったデータ、パラレルデータを用いて一対一変換を行ったデータをDTWによって伸縮させたものである。

//image[evgmm-result][多対一の声質変換処理と一対一の声質変換処理の結果データの比較][scale=0.6]

ここでは、モデルの学習に11人の事前学習用入力話者についてそれぞれ10文のパラレルデータを用いた。また、話者適応には1文のみ、変換元の特徴量をそのまま用いた。
図を見るとわかるように、パラレルデータを用いた一対一の声質変換に比べても遜色ない精度で変換出来ていることがわかる。

しかし、MFCCの第2次係数を見ると、多対一変換においては推移が滑らかになりすぎていることがわかる。
これは過剰な平滑化と呼ばれる現象で、GMMをベースとした声質変換の1つの問題点である。
これを解決する手法としては、系列内変動(Global Variance)や変調スペクトル(Modulation Spectrum)といった仕組みが考案されているが、
ここでは説明を割愛する。
