# magokoroformer

本リポジトリは競馬データ（2010–2025 など）の前処理・学習・評価を行うスクリプト群を含みます。主要な操作はローカルで仮想環境を作成して依存ライブラリをインストールし、学習スクリプトを実行するだけで始められます。

## 概要
- 学習スクリプト: `magokoroformer2025.py`
- レポート更新スクリプト: `report_csv.py`
- データ（pickle 形式）: `data/` フォルダ内（例: `dump_race2010_2025.pickle`, `dump_list2010_2025.pickle`, `dump_kisyu2010_2025.pickle` など）
- 出力: `report.csv`（レポート）および Excel（例: `kelly基準損益.xlsx`）で性能確認

## 前提条件
- Git, Python（推奨: 3.8 以上）
- OS: macOS / Linux / Windows（コマンドは環境に合わせて読み替えてください）
- 大きなデータファイルは Git LFS で管理されている場合があります（下記参照）。

## クイックスタート（推奨手順）
1. リポジトリをクローン
   - git clone https://github.com/SCSKQ-Okabe/magokoroformer.git
   - cd magokoroformer

2. （必要なら）Git LFS を初期化して LFS 管理ファイルを取得
   - git lfs install
   - git lfs pull

3. 仮想環境を作成・有効化
   - macOS / Linux:
     - python -m venv venv
     - source venv/bin/activate
   - Windows (PowerShell):
     - python -m venv venv
     - .\venv\Scripts\Activate.ps1
   - Windows (cmd.exe):
     - .\venv\Scripts\activate.bat

4. 依存関係のインストール
   - pip install -r requirements.txt

5. 学習を開始
   - python magokoroformer2025.py
   - ※学習スクリプトは長時間動作する可能性があります。学習の設定（エポック数や出力先など）はスクリプト内の引数や設定ファイルで調整してください。

6. 学習後のレポート更新・評価
   - python report_csv.py
   - これにより `report.csv` が更新され、`kelly基準損益.xlsx` に反映して性能確認できる想定です（Excel ファイルは同ディレクトリか、スクリプトで参照するパスに配置してください）。

## データについて
- data/ フォルダにある pickle ファイルは大きなバイナリです。clone 後に中身が空や LFS プレースホルダになっている場合は `git lfs pull` を実行してください。
- データを更新・差し替える場合は、元の形式（pickle）やスクリプトが期待するカラム構成に合わせてください。

## 推奨・注意事項
- Python のバージョンはスクリプトの互換性に依存します。エラーが出る場合は Python 3.8〜3.11 などで試してください。
- GPU を用いる学習（PyTorch / TensorFlow 等）を行う場合、CUDA/ドライバの環境を整えてください（requirements.txt の内容を確認）。
- 学習ログ・モデル出力先はスクリプト内の設定（コマンドライン引数やハードコードされたパス）を確認して必要に応じて変更してください。
- 大きなファイルを扱うため、ディスク容量に余裕を確保してください。

## トラブルシューティング
- requirements.txt にないライブラリの ImportError が出る → requirements.txt を確認・修正し、再インストールしてください。
- pickle 読み込み時にバージョンエラーや不整合が出る → データの作成元（Python バージョンや保存時のライブラリ）を確認してください。
- `git lfs` のオブジェクトが取得できない → `git lfs install` ��� `git lfs pull` を実行し、アクセス権（プライベートリポジトリの場合）を確認してください。

## 出力の確認方法
- `report.csv` を確認し、期待する指標が含まれているかを確認します。
- `kelly基準損益.xlsx` を Excel 等で開いて、`report.csv` の結果が正しく反映されているか確認します。

## 開発・貢献
- 改善や不具合修正は Issue を立てるか、Fork → Pull Request をお願いします。
- 大きなデータを更新する場合は Git LFS を利用し、必要ならデータのメタ情報（更新日、抜粋）を README や別ファイルに追記してください。

## ライセンス / 連絡先
- ライセンス情報がリポジトリに無ければ追記を検討してください（例: MIT License）。
- 問い合わせ・改善提案: リポジトリの Issue を利用してください。
