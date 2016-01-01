# SunPro会誌 2016

[![Build Status](https://travis-ci.org/SunProIO/SunPro-C89.svg?branch=master)](https://travis-ci.org/SunProIO/SunPro-C89)

## Build

Ubuntu推奨。

### TL;DR

Ubuntu:

```sh
# apt packages
sudo apt-get install --no-install-recommends install texlive-latex-extra -y
sudo apt-get install dvipng libssl-dev libreadline-dev zlib1g-dev -y

# rbenv
git clone https://github.com/rbenv/rbenv.git ~/.rbenv
echo 'export PATH="$HOME/.rbenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(rbenv init -)"' >> ~/.bashrc
git clone https://github.com/rbenv/ruby-build.git ~/.rbenv/plugins/ruby-build

# nvm
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.30.1/install.sh | bash

source ~/.bashrc

# clone
git clone https://github.com/SunProIO/SunPro-C89.git
cd SunPro-C89

# Ruby
rbenv install
gem install bundler
bundle install

# Node.js
nvm install
npm install

# Build
make all -B
npm run build
```

### Dependencies

* Git
* [rbenv](https://github.com/rbenv/rbenv#installation) (+ [ruby-build](https://github.com/rbenv/ruby-build))
* [nvm](https://github.com/creationix/nvm#installation)
* GNU Make
* LaTeX: `latex`コマンドがちゃんと(?)使えればよい。CJKもとうぶん必要なし。Ubuntuの場合`apt-get --no-install-recommends install texlive-latex-extra`で十分。
* dvipng: Ubuntuなら`apt-get install dvipng`

rbenvやnvmを使用せずにRubyやNode.jsを入れられるならWindows(cygwin)でも動くはず。

### clone

```sh
git clone https://github.com/SunProIO/SunPro-C89.git
```

### Install Dependencies

#### Ruby

```sh
rbenv install
gem install bundler
bundle install
```

**Note**: Ubuntuでは`apt-get install libssl-dev libreadline-dev zlib1g-dev`しないとRubyのビルドに失敗するよ。

#### Node.js

```sh
nvm install
npm install
```

### InDesign用XMLの生成

```sh
make xml
```

### HTMLの生成

```sh
make html
```

### C89特設サイトの生成

上記HTMLの生成を実行したあと、

```sh
npm run build
```

でdistディレクトリが生成される。

## Note

* Gemfile.lockやshrinkwrapはgitignoreしてあるよ。依存パッケージの最新版で正常に動くプロジェクトこそが健全なプロジェクトだよ。
