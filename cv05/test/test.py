import json
import os
import statistics
import unittest
from scipy.stats import sem

from my_utils import wandb_utils

RESULTS = "results"


class TestCV05(unittest.TestCase):
    WandbData = None
    Results = {
        "cv02_sts_val_cz": 0,
        "cv02_sts_val_rob": 0,
        "cv02_sts_val_fer": 0,
        "cv02_sts_pt": 0,

        "cv03_sent_val_cz": 0,
        "cv03_sent_val_rob": 0,
        "cv03_sent_val_fer": 0,
        "cv03_sent_pt": 0,

        "Discussion": "-",
        "Overall": "-",
    }
    N_RUNS = 5

    MANDATORY_HP = ["task", "model_type"]

    @classmethod
    def add_result_value(cls, column: str, val):
        if isinstance(val, float):
            if val < 1:
                val = f"{val:.4f}"
            else:
                val = f"{val:.2f}"
        TestCV05.Results[column] = (column, val)

    def check_best(self, filtered_data, min_acc, top_runs: int, metric="test_acc", col_metric=None,
                   col_pt=None, val_pt=None, best="max"):
        sort_invert = True if best == "max" else False

        test_metric = wandb_utils.best_metric(filtered_data, metric, top_n=top_runs, sort_invert=sort_invert)
        print(test_metric)
        if col_metric is not None:
            acc_top_n_mean = statistics.mean(test_metric)
            conf_interval = sem(test_metric) * 2.7764  # for 5 examples and 95conf

            TestCV05.add_result_value(col_metric, f"{acc_top_n_mean}(±{conf_interval})")

        if not test_metric:
            if col_pt is not None:
                TestCV05.add_result_value(col_pt, "NEE")
            self.fail(f"too little experiments {len(test_metric)} < {top_runs} found: {test_metric}")

        if best == "max" and test_metric[-1] < min_acc:
            if col_pt is not None:
                TestCV05.add_result_value(col_pt, "LOW")
            self.fail(
                f"not found satisfactory results\n all top {top_runs} test_loss:{test_metric} should be more than {min_acc}")

        if best == "min" and test_metric[-1] > min_acc:
            if col_pt is not None:
                TestCV05.add_result_value(col_pt, "HIGH")
            self.fail(
                f"not found satisfactory results\n all top {top_runs} test_loss:{test_metric} should be less than {min_acc}")


        if col_pt is not None:
            TestCV05.add_result_value(col_pt, val_pt)

    @classmethod
    def tearDownClass(cls):
        formatted_results = dict()
        for i, (name, value) in enumerate(TestCV05.Results.items()):
            formatted_results[i] = (name, value)

        os.makedirs(RESULTS, exist_ok=True)
        with open(f"{RESULTS}/cv05.json", 'w', encoding="utf-8") as fd:
            json.dump(formatted_results, fd)

    def test_sts1(self):
        data = wandb_utils.load_runs(["cv05", "best"], mandatory_hp=TestCV05.MANDATORY_HP,
                                     mandatory_m=["test_loss", "train_loss"],
                                     minimum_runtime_s=0, minimum_steps=0)
        filtered = wandb_utils.filter_data(data, {"task": "sts", "model_type": "UWB-AIR/Czert-B-base-cased"})

        self.check_best(filtered, 0.65, TestCV05.N_RUNS, metric="test_loss", col_metric="cv02_sts_val_cz",
                        best="min")

    def test_sts2(self):
        data = wandb_utils.load_runs(["cv05", "best"], mandatory_hp=TestCV05.MANDATORY_HP,
                                     mandatory_m=["test_loss", "train_loss"],
                                     minimum_runtime_s=0, minimum_steps=0)
        filtered = wandb_utils.filter_data(data, {"task": "sts", "model_type": "ufal/robeczech-base"})

        self.check_best(filtered, 0.65, TestCV05.N_RUNS, metric="test_loss", col_metric="cv02_sts_val_rob",
                        best="min")

    def test_sts3(self):
        data = wandb_utils.load_runs(["cv05", "best"], mandatory_hp=TestCV05.MANDATORY_HP,
                                     mandatory_m=["test_loss", "train_loss"],
                                     minimum_runtime_s=0, minimum_steps=0)
        filtered = wandb_utils.filter_data(data, {"task": "sts", "model_type": "fav-kky/FERNET-C5"})

        self.check_best(filtered, 0.65, TestCV05.N_RUNS, metric="test_loss", col_metric="cv02_sts_val_fer",
                            best="min")
    def test_sentiment_1(self):
        data = wandb_utils.load_runs(["cv05", "best"], mandatory_hp=TestCV05.MANDATORY_HP,
                                     mandatory_m=["test_acc", "train_loss", "test_loss"],
                                     minimum_runtime_s=0, minimum_steps=0)
        filtered = wandb_utils.filter_data(data, {"task": "sentiment", "model_type": "UWB-AIR/Czert-B-base-cased"})
        self.check_best(filtered, 0.75, TestCV05.N_RUNS, metric="test_acc", col_metric="cv03_sts_val_cz", best="max")

    def test_sentiment_2(self):
        data = wandb_utils.load_runs(["cv05", "best"], mandatory_hp=TestCV05.MANDATORY_HP,
                                     mandatory_m=["test_acc", "train_loss", "test_loss"],
                                     minimum_runtime_s=0, minimum_steps=0)
        filtered = wandb_utils.filter_data(data, {"task": "sentiment", "model_type": "ufal/robeczech-base"})
        self.check_best(filtered, 0.75, TestCV05.N_RUNS, metric="test_acc", col_metric="cv03_sts_val_rob", best="max")

    def test_sentiment_3(self):
        data = wandb_utils.load_runs(["cv05", "best"], mandatory_hp=TestCV05.MANDATORY_HP,
                                     mandatory_m=["test_acc", "train_loss", "test_loss"],
                                     minimum_runtime_s=0, minimum_steps=0)
        filtered = wandb_utils.filter_data(data, {"task": "sentiment", "model_type": "fav-kky/FERNET-C5"})
        self.check_best(filtered, 0.75, TestCV05.N_RUNS, metric="test_acc", col_metric="cv03_sts_val_fer", best="max")
