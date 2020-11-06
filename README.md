# Pix2PixHDの使い方

## 実行環境
|               |                  |
| ------------- | ---------------- |
| OS            | Ubuntu 18.04 LTS |
| CUDA          | 10.1             |
| GPU           | GeForce GTX 1080 |
| nvidia driver | 435.21           |
| torch         | 1.4.0            |
| torchvision   | 0.5.0            |

- pytorchを導入しておく必要がある
- 未導入なら[ここ](https://pytorch.org/get-started/locally/)からインストール


```bash
pip install dominate
```

```bash
git clone https://github.com/NVIDIA/pix2pixHD.git
cd pix2pixHD
```

## trainingについて
``datasets``内に学習用のフォルダを作る


## 事前準備
以下の2種類の画像が入ったフォルダを作成する（画像サイズは1024x1024で統一する)
- 入力画像を入れるフォルダを``train_A``として作成
- 正解画像を入れるフォルダを``train_B``として作成

```bash
mkdir datasets/mydatasets
```

``mydatasets``フォルダ直下に``train_A``と``train_B``を配置する

## 訓練を実行する
```bash
python train.py --label_nc 0 --name mydatasets --loadSize 1024 --dataroot ./datasets/mydatasets --save_epecho_freq 10
```

- ``train_A`がRGBカラーなら``--label_nc 0``
- ``--name mydatasets``は学習モデルの名前を入力
- ``--batchSize 1``はバッチサイズを指定
- ``--loadSize 1024``は入力画像のサイズを指定
- ``--dataroot ./datasets/mydatasets``は使用するデータセットを指定
- instance map(segmentationを使うときに用いる)を生成しないなら``--no instance``
- ``--save_epoch_freq 10``epochを保存する頻度を指定
- ``--no_html``htmlを出力しない
- オプションの種類は``options``フォルダ内に配置されているファイルから確認できる

## Testing

-``checkpoints``フォルダ内に生成されたファイルNAME(NAMEは任意)を``--name NAME``として指定

## デモを動かす場合
```bash
mkdir -p checkpoints/label2city_1024p
```

``label2city_1024p``直下に[公式READ.ME](https://github.com/NVIDIA/pix2pixHD)の``Testing``の項目の``here``から``latest_net_G.pth``をダウンロードし、``label2city_1024p``直下に配置する

## 実行

```bash
 python test.py --name label2city_1024p --netG local --ngf 32 --resize_or_crop scale_width
```

## エラー処理

```bash
Traceback (most recent call last): File "test.py", line 59, in <module> generated = model.inference(data['label'], data['inst'], data['image']) File "/home/ubuntu/atlas/pix2pixHD/models/pix2pixHD_model.py", line 198, in inference input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True) File "/home/ubuntu/atlas/pix2pixHD/models/pix2pixHD_model.py", line 126, in encode_input edge_map = self.get_edges(inst_map) File "/home/ubuntu/atlas/pix2pixHD/models/pix2pixHD_model.py", line 264, in get_edges edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1]) RuntimeError: Expected object of scalar type Byte but got scalar type Bool for argument #2 'other' 
```

### 解決方法
``models/pix2pixHD_model.py``に以下の行を追加

```py
...
# この行を追加
edge=edge.bool()
edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
...
```
