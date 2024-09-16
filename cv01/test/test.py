import json
import os
import statistics
from unittest import TestCase

import my_utils.wandb_utils as wandb_utils

ACC_MINIMUM_N = 10
ACC_MINIMUM_VAL = 99

RESULTS = "results"


class TestMnist(TestCase):
    results = None

    @classmethod
    def setUpClass(cls):
        TestMnist.results = dict()

    @classmethod
    def tearDownClass(cls):
        os.makedirs(RESULTS, exist_ok=True)

        with open(f"{RESULTS}/cv01.json", 'w', encoding="utf-8") as fd:
            json.dump(TestMnist.results, fd)

    def __init__(self, *args, **kwargs):
        super(TestMnist, self).__init__(*args, **kwargs)
        mandatory_hp = ["lr", "optimizer", "dp"]
        mandatory_m = ["test_acc", "train_loss", "test_loss"]
        self.wandb_data = wandb_utils.load_runs(["cv01"], mandatory_hp=mandatory_hp, mandatory_m=mandatory_m)

    def test_grid(self):
        grid = {"lr": [0.1, 0.01, 0.001, 0.0001, 0.00001], "model": ["dense", "cnn"], "optimizer": ["sgd", "adam"], "dp": [0, 0.1, 0.3, 0.5]}
        min_n = 2

        grid_status = wandb_utils.grid_status(self.wandb_data, grid)

        ok = True

        for config, num_runs in grid_status.items():
            if num_runs < min_n:
                print(f"not enough experiments with configuration {config} (only {num_runs}<{min_n})")
                ok = False
            else:
                print(f"configuration {config} (runs:{num_runs}) -> OK")
        if not ok:
            TestMnist.results[0] = ("hp experiments", "NEE")

            self.fail("FAILED")
        TestMnist.results[0] = ("hp experiments", "2")

    def test_meta(self):
        TestMnist.results[3] = ("metacentrum", "")

    def test_acc(self):
        test_acc = wandb_utils.best_metric(self.wandb_data, "test_acc")
        acc_top_10_mean = statistics.mean(test_acc[:10])
        if len(test_acc) < ACC_MINIMUM_N:
            TestMnist.results[1] = ("acc", "NEE")
            TestMnist.results[2] = ("acc_val", "0")

            self.fail(f"too little experiments {len(test_acc)} < {ACC_MINIMUM_N}")
        if test_acc[ACC_MINIMUM_N - 1] < ACC_MINIMUM_VAL:
            TestMnist.results[1] = ("acc", f"LOW")
            TestMnist.results[2] = ("acc_val", f"{acc_top_10_mean}")

            self.fail(
                f"not found satisfactory results\ntop {ACC_MINIMUM_N} test_acc:{test_acc[:ACC_MINIMUM_N]} should be > {ACC_MINIMUM_VAL}")
        TestMnist.results[1] = ("acc", f"OK")
        TestMnist.results[2] = ("acc_val", f"{acc_top_10_mean}")


