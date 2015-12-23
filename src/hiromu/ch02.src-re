= 統計的声質変換の基礎知識

== 統計的声質変換とは

統計的声質変換とは、人間の声から抽出された特徴量を統計的手法によって変換することによって、
ある人の音声データを元に、別の人が同じ内容を話したかのような音声データを生成する仕組みのことである。
基本的には、変換元となる人と変換先となる人が同じ内容を話しているデータ(以降、「パラレルデータ」という)を用いて、
統計的モデルを学習することによって、変換を行っている。

ただし、この統計的モデルが扱えるのはあくまで数値データであり、音声データをそのまま学習に用いることはできない@<fn>{fn1}。
//footnote[fn1][厳密に言えば、音声データをそのまま数値化して扱うことはできるものの、データ量が大きすぎて学習に時間が掛かり過ぎる。]
そこで、音声データを少ないパラメータで表すために、音声分析による特徴量抽出が必要になる。
もちろん、統計的モデルによって変換されたパラメータから、音声合成によって音声データを復元する作業も必要になる。

つまり、統計的声質変換は、以下の様な仕組みとなっている。
//image[frame][統計的声質変換の仕組み][scale=0.3]

== 音声分析による特徴量抽出

音声分析のベースには、ソースフィルタモデルと呼ばれる概念が取り入れられている。
ソースフィルタモデルにおいては、声帯の振動によって生み出された音を声道や口腔の形状によって変化させることで発声しているという考え方に基づき、
声帯からの音源(ソース)を表すパラメータと、声道や口腔での変化(フィルタ)を表すパラメータを用いて音声の特徴を表す。
この際、ソースのパラメータは声の高さや太さ、フィルタのパラメータは「あ」「い」といった発声内容や声の個人性(人それぞれの特徴)などを表すこととなる。
つまり、このフィルタを表すパラメータを統計的モデルによって変換することによって声質変換ができる。
//image[source-filter][ソースフィルタモデル][scale=0.3]

今回は、このパラメータの抽出については、TANDEM-STRAIGHT@<bib>{tandem-straight}というシステムを利用する。
これは、山梨大学の森勢助教が公開している@<fn>{tandem-straight}音声分析・合成システムであり、
音声データを分析し、ソースを表すパラメータである基本周波数、フィルタを表すパラメータであるスペクトル包絡、
そして、音声のかすれや雑音を表すパラメータである非周期成分という3つのパラメータを抽出することができる。
もちろん、この3つのパラメータから音声データを合成することも可能である。
//footnote[tandem-straight][http://ml.cs.yamanashi.ac.jp/straight/]

さらに、メル周波数ケプストラム係数(以降、「MFCC」という)というものを導入する。
これは、スペクトル包絡をより人間の知覚に沿うような形で抽出した特徴量であり、
具体的には、低い音に対してはその音程の細かな違いに気づくが、高い音になるほど音程の違いに気づきにくくなるという
人間の特性を利用している。
MFCCは、スペクトル包絡に対して低い周波数帯がより強調されるようなフィルタ(メルバンクフィルタという)を適用し、
離散コサイン変換を行うことによって得られる。

== EMアルゴリズム

統計的声質変換には、後述する混合ガウスモデルがよく使われるのだが、それだけに限らず様々な場面で使われるのがEMアルゴリズムである。
EMアルゴリズムは、確率モデルのパラメータを推定するときに使われる手法で、ある観測データ@<m>{\boldsymbol{X\}}、その観測データと潜在変数@<m>{\boldsymbol{Z\}}との
同時分布@<m>{P(\boldsymbol{X\}, \boldsymbol{Y\} | \boldsymbol{\theta\})}が与えられた時に、
@<m>{P(\boldsymbol{X\} | \boldsymbol{\theta\}}}を最大化するようなパラメータ@<m>{\boldsymbol{\theta\}}を推定することができる。
式で表すと、以下の通りとなる。

//texequation{
    \begin{split}
        \hat{\boldsymbol{\theta}} = & \ \underset{\boldsymbol{\theta}}{argmax} \ P(\boldsymbol{X} | \boldsymbol{\theta}) \\
                                  = & \ \underset{\boldsymbol{\theta}}{argmax} \ \prod_{all \ \boldsymbol{Z}} P(\boldsymbol{X}, \boldsymbol{Z} | \boldsymbol{\theta})
    \end{split}
//}

ここで、Q関数と呼ばれる関数を導入する。

//texequation{
    \mathcal{Q} (\boldsymbol{\theta}, \boldsymbol{\theta} ^ {old}) = \sum_{all \ \boldsymbol{Z}} P(\boldsymbol{Z} | \boldsymbol{X}, \boldsymbol{\theta} ^ {old}) \log P(\boldsymbol{X}, \boldsymbol{Z} | \boldsymbol{\theta})
//}

すると、以下のステップによって、@<m>{\boldsymbol{\theta\}}を推定できる。

 1. パラメータの初期値@<m>{\boldsymbol{\theta\} ^ {old\}}を適当に定める
 2. Eステップ: @<m>{P(\boldsymbol{Z\} | \boldsymbol{X\}, \boldsymbol{\theta\} ^ {old\})}を計算する。
 3. Mステップ: Eステップで得られた値を元に@<m>{\boldsymbol{\theta\} ^ {new\} = \underset{\boldsymbol{\theta\}\}{argmax\} \mathcal{Q\} (\boldsymbol{\theta\}, \boldsymbol{\theta\} ^ {new\})}を計算する。
 4. @<m>{\boldsymbol{\theta\}}が収束するまで@<m>{\boldsymbol{\theta\} ^ {old\} \leftarrow \boldsymbol{\theta\} ^ {new\}}としてEステップとMステップを繰り返す。

なぜこの方法によって@<m>{\boldsymbol{\theta\}}の最尤推定ができるのかという詳しい説明は専門書に譲るとして、
図を用いて簡単に説明をしておく。
以下の図は、@<m>{\boldsymbol{\theta\}}と対数尤度関数@<m>{\log P(\boldsymbol{X\} | \boldsymbol{\theta\})}のグラフである。

//image[em][EMアルゴリズムのイメージ][scale=0.28]

実は、Eステップは対数尤度関数の@<m>{\boldsymbol{\theta\} ^ {old\}}における下界を求めるというのと対応し、
Mステップはその下界を最大化するパラメータ@<m>{\boldsymbol{\theta\} ^ {new\}}を求めるというのと対応している。
図を見れば、それを繰り返すことによって対数尤度関数を最大化するようにパラメータが動いていることが分かるだろう。

== 混合ガウスモデル

混合ガウスモデル(以降、「GMM」という)は、複数のガウス分布を組み合わせることによって表されるモデルである。
扱うデータを@<m>{D}次元のベクトル@<m>{\boldsymbol{x\}}としたときの定義は以下の通りである。

//texequation{
    P(\boldsymbol{x} | \boldsymbol{\lambda}) = \sum_{m = 1}^{M} w_m \mathcal{N} (\boldsymbol{x}; \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m)
//}

ここで、@<m>{\mathcal{N\} (\boldsymbol{x\}; \boldsymbol{\mu\}_m, \boldsymbol{\Sigma\}_m)}は、平均ベクトルが@<m>{\boldsymbol{\mu\}_m}、分散共分散行列が@<m>{\boldsymbol{\Sigma\}_m}のガウス分布である。

//texequation{
    \mathcal{N} (\boldsymbol{x}; \boldsymbol{\mu}_m, \boldsymbol{\Sigma}_m) = \frac{1}{(2 \pi) ^ \frac{d}{2} \boldsymbol{\Sigma}_m} \exp(- \frac{1}{2} (\boldsymbol{x} - \boldsymbol{\mu}_m) ^ \top \boldsymbol{\Sigma}_m ^ {-1} (\boldsymbol{x} - \boldsymbol{\mu}_m))
//}

すなわち、GMMは@<m>{M}個のガウス分布を線形結合したものであり、確率密度関数となるために以下の制約を持つ。

//texequation{
    w_1, ..., w_m \geq 0, \sum_{m = 1}^{M} w_m = 1
//}

つまり、GMMのパラメータは、それぞれのガウスモデルの重みと平均ベクトル、分散共分散行列からなり、まとめて@<m>{\boldsymbol{\lambda\}}として表される。

=== 混合ガウスモデルの最尤推定

GMMの学習は、学習データのそれぞれに対して推定される確率の積が最大となるようなパラメータを見つけることである。
つまり、@<m>{N}個の学習データに対し、尤度関数@<m>{L(\boldsymbol{\lambda\})}を用いて以下のように定式化される。

//texequation{
    L(\boldsymbol{\lambda}) := \prod_{n = 1}^{N} P(\boldsymbol{x}_n, \boldsymbol{\lambda}) \\\\
//}

//texequation{
    \hat{\boldsymbol{\lambda}} := \underset{\boldsymbol{\lambda}}{argmax} \ L(\boldsymbol{\lambda}) \ \text{subject to}
    \left\{
        \begin{array}{l}
            w_1, ..., w_m \geq 0 \\
            \sum_{m = 1}^{M} w_m = 1
        \end{array}
    \right.
//}

このとき、最尤推定量@<m>{\hat{\boldsymbol{\lambda\}\}}は次式を満たす。

//texequation{
    \left. \frac{\partial}{\partial \boldsymbol{\lambda}} L(\boldsymbol{\lambda}) \right|_{\boldsymbol{\lambda} = \hat{\boldsymbol{\lambda}}} = 0
//}

これは、重み、平均ベクトル、分散共分散行列について、それぞれ以下を満たす。

//texequation{
    \hat{w}_m = \frac{1}{N} \sum_{n = 1}^{N} \hat{\eta}_{n, m}
//}

//texequation{
    \hat{\boldsymbol{\mu}}_m = \frac{ \sum_{n = 1}^{N} \hat{\eta}_{n, m} \boldsymbol{x}_n }{ \sum_{n = 1}^{N} \hat{\eta}_{n, m} }
//}

//texequation{
    \hat{\boldsymbol{\Sigma}}_m = \frac{ \sum_{n = 1}^{N} \hat{\eta}_{n, m} (\boldsymbol{x}_n - \hat{\boldsymbol{\mu}}_m) ^ \top (\boldsymbol{x}_n - \hat{\boldsymbol{\mu}}_m) }{ d \sum_{n = 1}^{N} \hat{\eta}_{n, m} }
//}

ただし、@<m>{\hat{\eta\}_{n, m\}}は次式の通りである。

//texequation{
    \hat{\eta}_{n, m} := \frac{ \hat{w}_m \mathcal{N} (x_n; \hat{\boldsymbol{\mu}}_m, \hat{\boldsymbol{\Sigma}}_m) }{ \sum_{m' = 1}^{M} \hat{w}_m \mathcal{N} (x_n; \hat{\boldsymbol{\mu}}_{m'}, \hat{\boldsymbol{\Sigma}}_{m'}) }
//}

これは、@<m>{\hat{\eta\}_{n, m\}}の定義にそれぞれのパラメータが用いられているため、パラメータを解析的に求めることは難しいので、
EMアルゴリズムを用いることで推定することができる。
この場合は、各ステップは以下のようになる。

 1. パラメータの初期値@<m>{\hat{w\}, \hat{\boldsymbol{\mu\}\}, \hat{\boldsymbol{\Sigma\}\}}を適当に定める。
 2. Eステップ: 現在のパラメータから@<m>{\hat{\eta\}_{n, m\}}を計算する。
 3. Mステップ: 現在の@<m>{\hat{\eta\}_{n, m\}}から@<m>{\hat{w\}, \hat{\boldsymbol{\mu\}\}, \hat{\boldsymbol{\Sigma\}\}}を計算する。
 4. 収束するまでEステップとMステップを繰り返す。

これにより、学習データに対する尤度を最大化するようなパラメータを推定することができる。
その他、EMアルゴリズムの導出やGMMについての具体的な仕組みなどは、C.M. ビショップ著「パターン認識と機械学習」@<bib>{prml}や
杉山 将著「統計的機械学習」@<bib>{stat-ml}などを参照されたい。
