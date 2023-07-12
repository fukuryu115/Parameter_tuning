# Parameter_tuning
MNISTを用いてAEとVAEを訓練し、双方の潜在空間を比較し考察する

# セットアップ方法

```bash
git clone https://github.com/fukuryu115/Parameter_tuning.git
cd Parameter_tuning
cd docker
source build_docker.sh {任意のパスワード}
source run_docker.sh
```
`http://localhost:62222`にアクセス

`notebook`内のJupyter notebookを実行

# 使用しているポートとその用途
※ホストと仮想環境のIPは全部一緒
|ポート|用途|
|---|---|
|62222|jupyter|
|6001|tensorboard用|
|6002|予備|

# docker内への入り方
`docker exec -it m1tutorial_parameter_tuning bash`

# tokenの確認方法
コンテナ内で
`jupyter server list`
