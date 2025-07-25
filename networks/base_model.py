import pickle
import os
from pathlib import Path
import scipy
import torch
import torch.nn.functional as F
import sys
import numpy as np
import json
import pandas as pd
import shutil
import scipy.stats
from typing import Sequence, Tuple

from tools.plot_time_frequency import TimeFrequencyFigData
from datasets.datasets import Datasets

class BaseModel(object):
    def __init__(self, args, train, test):
        self.args = args
        print("selected gpu id:{}".format(args.gpu_id))
        self.device = torch.device("cuda" if args.use_cuda else "cpu")
        torch.cuda.set_device(args.gpu_id[0]) if args.use_cuda else None
        print(self.device)
        try:
            self.data = Datasets(self.args.dataset).data(self.args)
        except KeyError:
            print('dataset "{}" is not supported'.format(self.args.dataset))
            print("please set another name \n{}".format(Datasets.show_list()))
            raise
        except Exception as e:
            print(f"failed to initialize dataset '{self.args.dataset}': {e}")
            raise
        self.train_loader = self.data.train_loader
        self.valid_loader = self.data.valid_loader
        self.test_loader = self.data.test_loader

        self.epoch = 0
        self.model = self.init_model()

        self.export_dir = f"{self.args.export_dir}" if self.args.export_dir else ""
        self.model_dataset = getattr(self.args, "model_dataset", "") or self.args.dataset
        print(f"model_dataset: {self.model_dataset}")
        if self.model_dataset.endswith("+"):
            self.model_dataset = self.model_dataset[:-1]
        
        self.result_dir = Path(f"{args.result_directory}/dev_data/{self.export_dir}_{args.score}/")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.eval_data_result_dir = Path(f"{args.result_directory}/eval_data/{self.export_dir}_{args.score}/")
        self.eval_data_result_dir.mkdir(parents=True, exist_ok=True)
        self.model_name_suffix = "_"+self.args.model_name_suffix if self.args.model_name_suffix else ""
        self.eval_suffix = "_Eval" if self.args.eval else ""
        base_name = f"{self.export_dir}/{self.args.model}_{self.model_dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}"
        self.checkpoint_dir = f"models/checkpoint/{base_name}"
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = f"{self.checkpoint_dir}/checkpoint.tar"
        Path(f"models/checkpoint/{self.export_dir}").mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path(f"logs/{base_name}")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / "log.csv"

        log_data = self.get_log_header()
        if (train and not self.args.restart):
            print(f"Generate blank log -> {self.log_path}")
            np.savetxt(self.log_path,[log_data],fmt="%s")

        # select using checkpoint file
        if self.args.checkpoint_path:
            checkpoint_path = self.args.checkpoint_path
            saved_log_path = Path(f"logs")
            for dir in Path(os.path.dirname(checkpoint_path)).parts[2:]:
                saved_log_path /= dir
            saved_log_path /= "log.csv"
            if os.path.exists(saved_log_path):
                print(f"copy log: {saved_log_path}\n\t->{self.log_path}")
                shutil.copyfile(saved_log_path, self.log_path)
            else:
                print(f"Generate blank log: {self.log_path}")
                np.savetxt(self.log_path,[log_data],fmt="%s")
        else:
            checkpoint_path = self.checkpoint_path
        
        # load checkpoint
        self.optim_state_dict = None
        if self.args.restart:
            if os.path.exists(checkpoint_path):
                print(f"load checkpoint -> {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path)
                self.load_state_dict(checkpoint=checkpoint)
                self.optim_state_dict = self.load_optim_state_dict(checkpoint=checkpoint)
                if os.path.exists(self.log_path):
                    print(f"log reindex: {self.log_path}")
                    log_data = pd.read_csv(self.log_path).reindex(columns=log_data.split(',')).fillna(0)
                    log_data.to_csv(self.log_path, index=False)
            else :
                print(f"not found -> {checkpoint_path}")
                np.savetxt(self.log_path,[log_data],fmt="%s")
        
        self.model.to(self.device)
        self.model_dir = Path(f"models/saved_model/{self.export_dir}/")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir/f"{self.args.model}_{self.model_dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}.pth"
        self.score_distr_file_path = self.model_dir/f"score_distr_{self.args.model}_{self.model_dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}.pickle"
        self.history_img = self.model_dir/f"history_{self.args.model}_{self.model_dataset}{self.model_name_suffix}{self.eval_suffix}_seed{self.args.seed}.png"
        self.tf_figdata = TimeFrequencyFigData(
            max_imgs=4,
            max_extract=1,
            frames=args.frames,
            frame_hop_length=args.frame_hop_length,
            shape=(self.data.channel, self.data.width, self.data.height)
        )

        self.result_column_dict = {
            "single_domain":["section", "AUC", "pAUC", "precision", "recall", "F1 score"],
            "source_target":["section", "AUC (source)", "AUC (target)", "pAUC", "pAUC (source)", "pAUC (target)",
                            "precision (source)", "precision (target)", "recall (source)", "recall (target)",
                            "F1 score (source)", "F1 score (target)"]
        }

        # output parameter to json
        self.args_path = f"{self.checkpoint_dir}/args.json"
        tf = open(self.args_path, "w")
        json.dump(vars(self.args), tf, indent=2, ensure_ascii=False)
        print(f"save args -> {self.args_path}")
        tf.close()

    def init_model(self):
        pass
    
    def get_log_header(self):
        return "loss,loss_var,time"

    def load_state_dict(self, checkpoint):
        pretrain_net_dict = checkpoint['model_state_dict']
        net_dict = self.model.state_dict()

        for key, val in pretrain_net_dict.items():
            if key not in net_dict:
                continue
            if net_dict[key].shape != val.shape:
                if 'position_embeddings' in key:
                    old = val.permute(0, 2, 1)
                    new_len = net_dict[key].size(1)
                    resized = F.interpolate(old, size=new_len,
                                            mode='linear', align_corners=False)
                    val = resized.permute(0, 2, 1)
                if net_dict[key].shape != val.shape:
                    print(
                        f"skip {key}: checkpoint {tuple(val.shape)} != model {tuple(net_dict[key].shape)}"
                    )
                    continue
            net_dict[key] = val

        self.model.load_state_dict(net_dict)
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']

    def load_optim_state_dict(self, checkpoint, key='optimizer_state_dict'):
        return checkpoint[key]

    def fit_anomaly_score_distribution(
        self,
        y_pred: Sequence[float],
        domain_list: Sequence[str] | None = None,
        score_distr_file_path: str | Path | None = None,
        percentile: float = 0.95,
        *,
        machine_type: str | None = None,
    ) -> float | dict:
        """Fit Gamma distribution(s) to anomaly scores.

        When ``domain_list`` is provided, separate distributions are fitted for
        ``"source"`` and ``"target"`` domains, and the parameters are stored as
        ``gamma_<domain>.pkl`` by default. When ``machine_type`` is given, the
        parameters are additionally stored as ``gamma_<machine_type>_<domain>.pkl``.
        The method returns the corresponding percentile threshold(s).
        """

        if score_distr_file_path is None:
            score_distr_file_path = self.score_distr_file_path
        score_distr_file_path = Path(score_distr_file_path)
        score_distr_file_path.parent.mkdir(parents=True, exist_ok=True)

        if domain_list is None:
            y_pred = np.asarray(y_pred, dtype=np.float64)
            y_min = np.min(y_pred)
            if len(y_pred) < 60:
                thresh = np.quantile(y_pred, percentile)
                with open(score_distr_file_path.with_suffix(".pkl"), "wb") as f:
                    pickle.dump(["percentile", thresh], f, protocol=pickle.HIGHEST_PROTOCOL)
                return float(thresh)
            if np.allclose(y_pred, y_pred[0]):
                shape_hat, loc_hat, scale_hat = 1.0, 0.0, max(y_pred[0], 1e-6)
            else:
                if np.any(y_pred <= 0):
                    y_pred = y_pred - y_pred.min() + 1e-8
                shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(y_pred, floc=0)

            with open(score_distr_file_path.with_suffix(".pkl"), "wb") as f:
                pickle.dump([shape_hat, loc_hat, scale_hat], f, protocol=pickle.HIGHEST_PROTOCOL)

            threshold = scipy.stats.gamma.ppf(percentile, shape_hat, loc=loc_hat, scale=scale_hat)
            threshold += (y_min - 1e-8)          # ← add this line
            return float(threshold)

        # ── per-domain calibration ──────────────────────────────────────────
        thresholds: dict[str, float] = {}
        for domain in ("source", "target"):
            scores = [s for s, d in zip(y_pred, domain_list) if d == domain]
            if not scores:
                continue
            scores = np.asarray(scores, dtype=np.float64)
            scores_min = np.min(scores)
            if len(scores) < 60:
                thresh = np.quantile(scores, percentile)
                filename = f"gamma_{domain}.pkl" if machine_type is None else f"gamma_{machine_type}_{domain}.pkl"
                gamma_path = score_distr_file_path.parent / filename
                with open(gamma_path, "wb") as f:
                    pickle.dump(["percentile", thresh], f, protocol=pickle.HIGHEST_PROTOCOL)
                key = domain if machine_type is None else f"{machine_type}_{domain}"
                thresholds[key] = float(thresh)
                continue
            scores = scores - scores_min + 1e-6
            shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(scores, floc=0)

            filename = f"gamma_{domain}.pkl" if machine_type is None else f"gamma_{machine_type}_{domain}.pkl"
            print("Fitting Gamma distribution for domain:", domain, "with machine type:", machine_type)

            gamma_path = score_distr_file_path.parent / filename
            with open(gamma_path, "wb") as f:
                pickle.dump([shape_hat, loc_hat, scale_hat], f, protocol=pickle.HIGHEST_PROTOCOL)

            key = domain if machine_type is None else f"{machine_type}_{domain}"
            thresholds[key] = float(
                scipy.stats.gamma.ppf(percentile, shape_hat, loc=loc_hat, scale=scale_hat)
                + scores_min - 1e-6
            )

        return thresholds

    
    def calc_decision_threshold(self, score_distr_file_path=None):
        if not score_distr_file_path:
            score_distr_file_path = self.score_distr_file_path
        score_distr_file_path = Path(score_distr_file_path)

        thresholds: dict[str, float] = {}

        for fpath in score_distr_file_path.parent.glob("gamma_*.pkl"):
            name_parts = fpath.stem.split("_")
            if len(name_parts) == 2:
                # domain specific, e.g. gamma_source.pkl
                _, domain = name_parts
                with open(fpath, "rb") as f:
                    params = pickle.load(f)
                if isinstance(params, list) and params[0] == "percentile":
                    thresholds[domain] = float(params[1])
                else:
                    shape_hat, loc_hat, scale_hat = params
                    thresholds[domain] = float(
                        scipy.stats.gamma.ppf(
                            q=self.args.decision_threshold,
                            a=shape_hat,
                            loc=loc_hat,
                            scale=scale_hat,
                        )
                    )
            elif len(name_parts) == 3:
                # machine and domain, e.g. gamma_ToyCar_source.pkl
                _, machine, domain = name_parts
                with open(fpath, "rb") as f:
                    params = pickle.load(f)
                key = f"{machine}_{domain}"
                if isinstance(params, list) and params[0] == "percentile":
                    thresholds[key] = float(params[1])
                else:
                    shape_hat, loc_hat, scale_hat = params
                    thresholds[key] = float(
                        scipy.stats.gamma.ppf(
                            q=self.args.decision_threshold,
                            a=shape_hat,
                            loc=loc_hat,
                            scale=scale_hat,
                        )
                    )

        if thresholds:
            return thresholds

        # fallback to single distribution
        with open(score_distr_file_path.with_suffix(".pkl"), "rb") as f:
            params = pickle.load(f)
        if isinstance(params, list) and params[0] == "percentile":
            return float(params[1])
        shape_hat, loc_hat, scale_hat = params
        return float(
            scipy.stats.gamma.ppf(
                q=self.args.decision_threshold,
                a=shape_hat,
                loc=loc_hat,
                scale=scale_hat,
            )
        )

    def train(self, epoch):
        pass

    def test(self):
        pass

    def copy_eval_data_score(self, decision_result_csv_path, anomaly_score_csv_path):
        eval_data_decision_result_csv_path = self.eval_data_result_dir / os.path.basename(decision_result_csv_path).replace(self.model_name_suffix, "")
        print(f"copy decision result: {decision_result_csv_path}\n\t->{eval_data_decision_result_csv_path}")
        shutil.copyfile(decision_result_csv_path, eval_data_decision_result_csv_path)

        eval_data_anomaly_score_csv_path = self.eval_data_result_dir / os.path.basename(anomaly_score_csv_path).replace(self.model_name_suffix, "")
        print(f"copy anomaly score: {anomaly_score_csv_path}\n\t->{eval_data_anomaly_score_csv_path}")
        shutil.copyfile(anomaly_score_csv_path, eval_data_anomaly_score_csv_path)
