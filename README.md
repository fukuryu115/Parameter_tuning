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

# tokenの確認方法とjupyterへの入り方
`docker logs m1tutorial_parameter_tuning`  \
を実行したあと、
```bash
== CUDA ==
==========

```
中略
```bash
  _   _          _      _
 | | | |_ __  __| |__ _| |_ ___
 | |_| | '_ \/ _` / _` |  _/ -_)
  \___/| .__/\__,_\__,_|\__\___|
       |_|
                                                                           
Read the migration plan to Notebook 7 to learn about the new features and the actions to take if you are using extensions.

https://jupyter-notebook.readthedocs.io/en/latest/migrate_to_notebook7.html

Please note that updating to Notebook 7 might break some of your extensions.

[I 2023-07-14 12:31:30.473 ServerApp] nbclassic | extension was successfully loaded.
[I 2023-07-14 12:31:30.474 ServerApp] Serving notebooks from local directory: /home/ryuhei-f/Parameter_tuning
[I 2023-07-14 12:31:30.474 ServerApp] Jupyter Server 2.7.0 is running at:
<span style="color:red;">[I 2023-07-14 12:31:30.474 ServerApp] http://localhost:62222/lab?token=969623e83718025e781a0ab218947c1518fc1bdf5b2def75</span>
[I 2023-07-14 12:31:30.474 ServerApp]     http://127.0.0.1:62222/lab?token=969623e83718025e781a0ab218947c1518fc1bdf5b2def75
[I 2023-07-14 12:31:30.474 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2023-07-14 12:31:30.475 ServerApp] 
    
```
赤字で表示されている部分のリンクの`localhost:`の後ろにtokenが記載されている。リンクを`ctrl+右クリ`することでjupyter notebookに入ることができる