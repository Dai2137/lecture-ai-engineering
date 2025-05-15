import time
import os
import pytest
import pandas as pd
import pickle

# テスト用データパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
# モデルのパスを定義
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/titanic_model.pkl")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


def test_inference_time(sample_data):
    model = pickle.load(open(MODEL_PATH, "rb"))

    # テストデータの準備
    test_data = sample_data  # テストデータを準備

    # 推論時間の測定
    start_time = time.time()
    model.predict(test_data)
    end_time = time.time()

    inference_time = end_time - start_time

    # 推論時間が許容範囲内かチェック（例：1秒以内）
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time:.2f}秒"
