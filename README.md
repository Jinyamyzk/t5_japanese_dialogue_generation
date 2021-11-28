# T5による会話生成
名大会話コーパスを使って、日本語T5事前学習済みモデルを転移学習しました。  
転移学習は[転移学習のサンプルコード](https://colab.research.google.com/github/sonoisa/t5-japanese/blob/main/t5_japanese_article_generation.ipynb)
を参考にしました。
## 日本語T5事前学習済みモデル
以下のモデルを使用しました。  
[sonoisia](https://huggingface.co/sonoisa/t5-base-japanese)
## 名大会話コーパス
[名大会話コーパス](https://mmsrv.ninjal.ac.jp/nucc/)を
[make-nucc-dataset](https://github.com/Jinyamyzk/make-nucc-dataset)を使ってtsvに整形したものを使用しました。
