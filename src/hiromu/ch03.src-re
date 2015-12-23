= 声質変換のための様々なモジュールの実装

この章では、声質変換のために必要となる様々なモジュールの実装と
簡単な解説を行う。

== STFファイルの読み書き

声質変換を行うためには、まずTANDEM-STRAIGHTで抽出された特徴量をPythonで扱える形式に変換しなければならない。
しかし、PythonからTANDEM-STRAIGHTをライブラリとして呼び出し、結果を受け取るのはかなり煩雑なので、
TANDEM-STRAIGHTの音声データから抽出した特徴量をSTFファイルという独自形式で保存する機能を利用する。

まず、音声データをSTFファイルに変換するには、配布されているTANDEM-STRAIGHTライブラリ内の
GenerateSTFという実行ファイルを利用できる。
そうして得られるSTFファイルを、Pythonで扱いやすいようデータを読み込んで、numpyの配列に変換するような
ライブラリを実装する。

STFファイルについての具体的な仕様は配布されているライブラリ内のドキュメントに詳しく記載されているが、
簡単に説明すると、1つのSTFファイルは複数のチャンクからなり、
それぞれのチャンクは格納しているデータの種類を表すチャンク名、チャンクサイズ、そしてデータから構成されている。
チャンクのうち、特徴量に直接関係のない制御用のものは以下の3つである。

 * STFT: エンディアンやサンプリング周波数などがヘッダ情報を表すチャンク
 * CHKL: そのSTFファイルに含まれるチャンクのリストを表すチャンク
 * NXFL: 複数ファイルに亘る場合に用いられるチャンク

ただし、GenerateSTFで得られるSTFファイルではNXFLチャンクは用いられないので、
STFTとCHKLのみに対して処理を行えばよい。
また、特徴量を表すチャンクはいくつかあるが、以下の3つのチャンクが存在していれば合成が可能なので、
これら以外は無視してもよい。

 * F0: 音声データの基本周波数
 * SPEC: 音声データのスペクトル包絡
 * APSG: 音声データの非周期成分をシグモイド関数のパラメータとして表したもの

これをPythonで実装したのが、以下のソースコードである。

//listnum[stf][stf.py][python]{
#!/usr/bin/env python

import numpy
import os
import struct
import sys

class STF:
    def __init__(self, filename = None):
        self.endian = '>'
        self.chunks = ['APSG', 'F0  ', 'SPEC']

    def loadfile(self, filename):
        with open(filename, 'rb') as stf_file:
            self.load(stf_file)

    def load(self, stf_file):
        filesize = os.fstat(stf_file.fileno()).st_size

        while stf_file.tell() < filesize:
            chunk = stf_file.read(4)

            if chunk == 'STRT':
                if stf_file.read(2) == '\xff\xfe':
                    self.endian = '<'
                chunk_size, self.version, self.channel, self.frequency \
                    = struct.unpack(self.endian + 'IHHI', stf_file.read(12))
            else:
                chunk_size, = struct.unpack(self.endian + 'I', stf_file.read(4))

                if chunk == 'CHKL' or chunk == 'NXFL':
                    data = stf_file.read(chunk_size)
                    if chunk == 'CHKL':
                        self.chunks += [data[i: i + 4] \
                            for i in range(0, chunk_size, 4) \
                            if data[i: i + 4] not in self.chunks]
                else:
                    self.shift_length, frame_count, argument, \
                    self.bit_size, self.weight, data_size \
                        = struct.unpack(self.endian + 'dIIHdI', stf_file.read(30))
                    data = stf_file.read(data_size)

                    element = data_size / (self.bit_size / 8)
                    matrix = numpy.fromstring(data, count = element)

                    for c in self.chunks:
                        if chunk == c:
                            if element / frame_count == 1:
                                self.__dict__[c.strip()] = matrix
                            else:
                                self.__dict__[c.strip()] \
                                    = matrix.reshape( \
                                        (frame_count, element / frame_count))
                            break

        for c in self.chunks:
            if c.strip() not in self.__dict__:
                self.__dict__[c.strip()] = None

    def savefile(self, filename):
        with open(filename, 'wb') as stf_file:
            self.save(stf_file)

    def save(self, stf_file):
        stf_file.write('STRT')
        if self.endian == '>':
            stf_file.write('\xfe\xff')
        elif self.endian == '<':
            stf_file.write('\xff\xfe')
        stf_file.write(struct.pack(self.endian + 'IHHI', 8, \
            self.version, self.channel, self.frequency))

        stf_file.write('CHKL')
        stf_file.write(struct.pack(self.endian + 'I', \
            len(''.join(self.chunks))) + ''.join(self.chunks))

        for c in self.chunks:
            if self.__dict__[c.strip()] is None:
                continue

            matrix = self.__dict__[c.strip()]
            if len(matrix.shape) == 1:
                argument = 1
            else:
                argument = matrix.shape[1]
            data_size = matrix.shape[0] * argument * 8

            header = struct.pack(self.endian + 'dIIHdI', self.shift_length, \
                matrix.shape[0], argument, self.bit_size, self.weight, data_size)
            stf_file.write(c + \
                struct.pack(self.endian + 'I', len(header) + data_size) + header)

            for i in xrange(matrix.shape[0]):
                if argument == 1:
                    stf_file.write(struct.pack(self.endian + 'd', matrix[i]))
                else:
                    for j in xrange(matrix.shape[1]):
                        stf_file.write(struct.pack(self.endian + 'd', \
                                                            matrix[i, j]))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: %s <stf_file>' % sys.argv[0]
        sys.exit()

    stf = STF()
    stf.loadfile(sys.argv[1])
    print stf.F0
//}

基本的に、structモジュールを用いてデータの読み込み・書き込みを行っている。
STFクラスのself.chunksに含まれているチャンクは、インスタンス変数としてnumpyの配列に変換されるようになっている。

== MFCCの抽出

次に、STFファイルから読み込んだスペクトル包絡からMFCCを抽出するライブラリを実装する。
また、声質変換後の音声合成のために、MFCCからスペクトル包絡を復元する機能も実装する。
なお、実装にあたっては、「人工知能に関する断創録@<bib>{mfcc-aidiary}」を参考にした。

まず、周波数をMFCCの算出に用いるメル尺度という音高の知覚的尺度に変換する関数と、その逆の処理をする関数を用意する。
hz2melによって周波数をメル尺度に、mel2hzによってメル尺度を周波数に変換する。

//emlist[周波数とメル尺度の変換][python]{
    def hz2mel(self, f):
        return 1127.01048 * numpy.log(f / 700.0 + 1.0)

    def mel2hz(self, m):
        return 700.0 * (numpy.exp(m / 1127.01048) - 1.0)
//}

次に、メルフィルタバンクを計算する。
これは、メル尺度上で等間隔にならぶバンドパスフィルタを並べたものであり、
このフィルタをスペクトル包絡に適用することで人間が知覚しにくい領域の重みが小さくなる。

//emlist[メルフィルタバンクの導出][python]{
    def melFilterBank(self):
        # サンプリング周波数の半分(ナイキスト周波数)までを対象とする
        fmax = self.frequency / 2
        melmax = self.hz2mel(fmax)

        # 周波数に合わせて、サンプル数の半分の標本数で計算する
        nmax = self.nfft / 2
        df = self.frequency / self.nfft

        # フィルタごとの中心となるメル尺度を計算する
        dmel = melmax / (self.channels + 1)
        melcenters = numpy.arange(1, self.channels + 1) * dmel
        fcenters = self.mel2hz(melcenters)

        # それぞれの標本が対象とする周波数の範囲を計算する
        indexcenter = numpy.round(fcenters / df)
        indexstart = numpy.hstack(([0], indexcenter[0: self.channels - 1]))
        indexstop = numpy.hstack((indexcenter[1: self.channels], [nmax]))

        # フィルタごとにindexstartを始点、indexcenterを頂点、
        # indexstopを終点とする三角形のグラフを描くように計算する
        filterbank = numpy.zeros((self.channels, nmax))
        for c in numpy.arange(0, self.channels):
            increment = 1.0 / (indexcenter[c] - indexstart[c])
            for i in numpy.arange(indexstart[c], indexcenter[c]):
                filterbank[c, i] = (i - indexstart[c]) * increment
            decrement = 1.0 / (indexstop[c] - indexcenter[c])
            for i in numpy.arange(indexcenter[c], indexstop[c]):
                filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)
            filterbank[c] /= (indexstop[c] - indexstart[c]) / 2

        return filterbank, fcenters
//}

ここで、self.channelsはメルフィルタバンクに用いられるバンドパスフィルタの数を示しており、
後に出てくるMFCCの次元数と同じか、より大きくする必要がある。

このメルフィルタバンクをスペクトル包絡に適用した後に、離散コサイン変換によって得られる係数がMFCCである。
つまり、MFCCを得る関数は以下のようになる。

//emlist[MFCCの導出][python]{
    def mfcc(self, spectrum):
        # スペクトル包絡として負の値が与えられた場合は、0として扱う
        spectrum = numpy.maximum(numpy.zeros(spectrum.shape), spectrum)
        # スペクトル包絡とメルフィルタバンクの積の対数を取る
        mspectrum = numpy.log10(numpy.dot(spectrum, self.filterbank.transpose()))
        # scipyを用いて離散コサイン変換をする
        return scipy.fftpack.dct(mspectrum, norm = 'ortho')[:self.dimension]
//}

ここで、self.dimensionは離散コサイン変換の結果のうち低次の係数から何次元だけ用いるかを示しており、
一般には16次元の係数を用いることが多いが、より精度を求める場合には32次元や64次元と設定することもある。

また、MFCCからスペクトル包絡への逆変換は、離散コサイン変換の逆変換を用いて、以下のように実装できる。
この実装においては、逆変換の後にscipy.interpolateを用いてスプライン補間をしている。

//emlist[MFCCからのスペクトル包絡の導出][python]{
    def imfcc(self, mfcc):
        # MFCCの削られた部分に0を代入した上で、逆離散コサイン変換をする
        mfcc = numpy.hstack([mfcc, [0] * (self.channels - self.dimension)])
        mspectrum = scipy.fftpack.idct(mfcc, norm = 'ortho')
        # 得られた離散的な値をスプライン補間によって連続的にする
        tck = scipy.interpolate.splrep(self.fcenters, numpy.power(10, mspectrum))
        return scipy.interpolate.splev(self.fscale, tck)
//}

そして、今後のために動的変化量を求める関数を実装しておく。
これは、あるフレームの前後数フレームでのMFCCの変化量を微分したもの(回帰係数)であり、
ここでは簡単のため、1つ前のフレームと1つ後のフレームの差を2で割ったものを用いることとする。

//emlist[MFCCの動的変化量の導出][python]{
    def delta(self, mfcc):
        # データの開始部と終了部は同じデータが続いているものとする
        mfcc = numpy.concatenate([[mfcc[0]], mfcc, [mfcc[-1]]])

        delta = None
        for i in xrange(1, mfcc.shape[0] - 1):
            # 前後のフレームの差を2で割ったものを動的変化量とする
            slope = (mfcc[i + 1] - mfcc[i - 1]) / 2
            if delta is None:
                delta = slope
            else:
                delta = numpy.vstack([delta, slope])

        return delta
//}

これまでのソースコードを全部組み合わせたmfcc.pyは以下の通りである。
なお、このモジュールをコマンドライン引数にSTFファイルを与えて実行すると、MFCCの出力及び
元データとMFCCからの逆変換で得られるデータの波形の違いを確認することができる。

//listnum[mfcc][mfcc.py][python]{
#!/usr/bin/env python

import numpy
import scipy.fftpack
import scipy.interpolate
import scipy.linalg
import sys

from stf import STF

class MFCC:
    '''
    MFCC computation from spectrum information

    Reference
    ---------
     - http://aidiary.hatenablog.com/entry/20120225/1330179868
    '''

    def __init__(self, nfft, frequency, dimension = 16, channels = 20):
        self.nfft = nfft
        self.frequency = frequency
        self.dimension = dimension
        self.channels = channels

        self.fscale = \
            numpy.fft.fftfreq(self.nfft, d = 1.0 / self.frequency)[: self.nfft / 2]
        self.filterbank, self.fcenters = self.melFilterBank()

    def hz2mel(self, f):
        return 1127.01048 * numpy.log(f / 700.0 + 1.0)

    def mel2hz(self, m):
        return 700.0 * (numpy.exp(m / 1127.01048) - 1.0)

    def melFilterBank(self):
        <省略>

    def mfcc(self, spectrum):
        <省略>

    def delta(self, mfcc):
        <省略>

    def imfcc(self, mfcc):
        <省略>

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: %s <stf_file>' % sys.argv[0]
        sys.exit()

    stf = STF()
    stf.loadfile(sys.argv[1])

    mfcc = MFCC(stf.SPEC.shape[1] * 2, stf.frequency)
    res = mfcc.mfcc(stf.SPEC[stf.SPEC.shape[0] / 5])
    spec = mfcc.imfcc(res)

    print res

    import pylab

    pylab.subplot(211)
    pylab.plot(stf.SPEC[stf.SPEC.shape[0] / 5])
    pylab.ylim(0, 1.2)
    pylab.subplot(212)
    pylab.plot(spec)
    pylab.ylim(0, 1.2)
    pylab.show()
//}

以下は、同じ音声データをMFCCに変換し、逆変換した結果を、
16次元まで求めたものと、64次元まで求めたもので比べたものである。
16次元でも低周波数ではある程度復元できているものの、64次元の方がより元のスペクトル包絡に近くなっていることがわかる。
また、64次元のものでも、高周波数帯になるにつれて線が滑らかになってしまっているのがわかるだろう。

//image[mfcc][MFCCの次元数による精度の違い][scale=0.25]

== 動的時間伸縮の実装

GMMを使った学習処理を行う前に、もう1つ実装しておくべき機能が、動的時間伸縮(Dynamic Time Warping、以下、「DTW」という)である。
これは、学習に使うパラレルデータを作るために欠かせない処理であり、同じ発話内容の音声データであっても
話すスピードや間の長さの違いによってフレームごとでは対応が取れていない場合が多いので、
DPマッチングに基づいてデータを伸縮させるという処理である。
つまり、以下の図のように、同じ音を発している部分の対応を見つけ、両方が同じフレーム数になるように伸縮させるという処理である。

//image[diff-spec][話者による波形の違い(@<bib>{dtw-ut}より引用)][scale=0.25]

そのためには、まずDPマッチングを実装する必要がある。
DPマッチングは、その名の通り動的計画法を用いた手法で、以下のように説明できる。

 1. 1つ目のデータを@<m>{a_1, a_2, ..., a_N}、2つ目のデータを@<m>{b_1, b_2, ..., b_M}として、N×Mのグリッドグラフを用意する。
 2. 頂点@<m>{(i, j)}に対し、@<m>{a_i}と@<m>{b_j}の類似度をコストとして設定する。
 3. 頂点@<m>{(1, 1)}から頂点@<m>{(N, M)}への最小コストのパスを検索する。
 4. 最小コストの値からパスを逆順に求めていき、頂点@<m>{(i, j)}を通過したならば、@<m>{a_i}と@<m>{b_j}が対応していることがわかる。

図で表すと以下の様なイメージとなる。
同じ内容の発話をしているフレームに対応する頂点を、赤いパスが通過していることがわかる。

//image[dtw-graph][DPマッチングのイメージ(@<bib>{dtw-ut}より引用)][scale=0.4]

Pythonでの実装は以下の通りとなる。
なお、self.distanceはフレーム間の類似度を算出する関数である。

//emlist[DPマッチング][python]{
    def dtw(self):
        M, N = len(self.source), len(self.target)
        cost = sys.maxint * numpy.ones((M, N))

        # グリッドグラフの1行目、1列目だけ先に処理しておく
        cost[0, 0] = self.distance(self.source[0], self.target[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + self.distance(self.source[i], self.target[0])
        for i in range(1, N):
            cost[0, i] = cost[0, i - 1] + self.distance(self.source[0], self.target[i])

        # 各頂点までの最短パスの長さを計算する
        for i in range(1, M):
            # 各フレームの前後self.windowフレームだけを参照する
            for j in range(max(1, i - self.window), min(N, i + self.window)):
                cost[i, j] = \
                        min(cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]) + \
                        self.distance(self.source[i], self.target[j])

        m, n = M - 1, N - 1
        self.path = []
        
        # 最短パスの経路を逆順に求めていく
        while (m, n) != (0, 0):
            self.path.append((m, n))
            m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), \
                                    key = lambda x: cost[x[0], x[1]])
            if m < 0 or n < 0:
                break

        self.path.append((0, 0))
//}

ここで注意すべき点は、self.windowによる探索範囲の制限である。
これは、あるフレームに対して比較対象を前後self.windowフレームのみに限定するというもので、
探索時間と、同じ発話内容が繰り返し現れた時に異なる繰り返しを参照してしまう可能性の低減を目的としている。
これまで使ってみた限りでは、self.windowを2つのデータのフレーム数の差の2倍程度にちょうど設定するくらいがよいであろう。

次に、DPマッチングの結果に応じてデータを伸縮させる処理を実装する。
この際、単に対応するフレームを用いるのではなく、1つ前のフレームが対応するフレームとの間で最も類似度が高いものを選ぶようにしている。
つまり、DPマッチングの説明の時と同じ記号を用いると、パスが@<m>{(i, j)}の次に@<m>{(i + 1, k)}を通るとすると、
@<m>{a_{i+1\}}に対応するフレームは、@<m>{b_j, b_{j+1\}, ..., b_k}の中で最も@<m>{a_{i+1\}}に類似度が高いフレームを選ぶ。

//emlist[DPマッチングによる伸縮][python]{
    def align(self, data, reverse = False):
        # reverse = Trueの時は、targetのフレーム数に合わせるようにする
        if reverse:
            path = [(t[1], t[0]) for t in self.path]
            source = self.target
            target = self.source
        else:
            path = self.path
            source = self.source
            target = self.target

        path.sort(key = lambda x: (x[1], x[0]))

        shape = tuple([path[-1][1] + 1] + list(data.shape[1:]))
        alignment = numpy.ndarray(shape)

        idx = 0
        frame = 0
        candicates = []

        while idx < len(path) and frame < target.shape[0]:
            if path[idx][1] > frame:
                # 候補となっているフレームから最も類似度が高いフレームを選ぶ
                candicates.sort(key = lambda x: \
                                    self.distance(source[x], target[frame]))
                alignment[frame] = data[candicates[0]]

                candicates = [path[idx][0]]
                frame += 1
            else:
                candicates.append(path[idx][0])
                idx += 1

        if frame < target.shape[0]:
            candicates.sort(key = lambda x: self.distance(source[x], target[frame]))
            alignment[frame] = data[candicates[0]]

        return alignment
//}

DTWの実装をすべてまとめてモジュールとしたものが以下のソースコードである。
この実装では、ユークリッド距離とコサイン距離のどちらかから類似度計算関数を選択できるようにしている。
また、このモジュールをコマンドライン引数に2つのSTFファイルを与えて実行すると、
DTW前と後での2つのMFCCの第1次係数のズレの違いが確認できるようになっている。

//listnum[dtw][dtw.py][python]{
#!/usr/bin/env python

import numpy
import scipy
import scipy.linalg
import sys

class DTW:
    def __getstate__(self): 
        d = self.__dict__.copy()

        if self.distance == self.cosine:
            d['distance'] = 'cosine'
        elif self.distance == self.euclidean:
            d['distance'] = 'euclidean'

        return d 

    def __setstate__(self, dict):
        self.__dict__ = dict

        if dict['distance'] == 'cosine':
            self.distance = self.cosine
        elif dict['distance'] == 'euclidean':
            self.distance = self.euclidean

    def cosine(self, A, B):
        return scipy.dot(A, B.transpose()) / scipy.linalg.norm(A) \
                                                / scipy.linalg.norm(B)

    def euclidean(self, A, B):
        return scipy.linalg.norm(A - B)

    def __init__(self, source, target, distance = None, window = sys.maxint):
        self.window = window
        self.source = source
        self.target = target

        if distance:
            self.distance = distance
        else:
            self.distance = self.euclidean

        self.dtw()

    def dtw(self):
        <省略>

    def align(self, data, reverse = False):
        <省略>

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: %s <source stf> <target stf>' % sys.argv[0]
        sys.exit()

    from stf import STF
    source, target = STF(), STF()
    source.loadfile(sys.argv[1])
    target.loadfile(sys.argv[2])

    from mfcc import MFCC
    mfcc = MFCC(source.SPEC.shape[1] * 2, source.frequency)
    source_mfcc = numpy.array([mfcc.mfcc(source.SPEC[frame]) \
                        for frame in xrange(source.SPEC.shape[0])])
    mfcc = MFCC(target.SPEC.shape[1] * 2, target.frequency)
    target_mfcc = numpy.array([mfcc.mfcc(target.SPEC[frame]) \
                        for frame in xrange(target.SPEC.shape[0])])

    dtw = DTW(source_mfcc, target_mfcc, \
        window = abs(source_mfcc.shape[0] - target_mfcc.shape[0]) * 2)
    warp_mfcc = dtw.align(source_mfcc)

    import pylab
    pylab.subplot(211)
    pylab.plot(source_mfcc[:, 0])
    pylab.plot(target_mfcc[:, 0])
    pylab.subplot(212)
    pylab.plot(warp_mfcc[:, 0])
    pylab.plot(target_mfcc[:, 0])
    pylab.show()
//}

このモジュールを同じ発話内容を異なる話者が発声したデータを使って実行した結果である。
上のグラフだと、最初から大きくずれが発生し、終了位置も異なるが、
下のグラフだと、波形がマッチしており、長さも等しくなっていることが分かる。

//image[dtw-result][DTWの実行結果]
