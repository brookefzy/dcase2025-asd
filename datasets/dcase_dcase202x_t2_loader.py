import os
import pickle
import torch
import numpy as np
import fasteners
from pathlib import Path
import time
import datetime
import glob
import librosa

from datasets import loader_common as com
from datasets.augmentations import AugmentationPipeline

class DCASE202XT2Loader(torch.utils.data.Dataset):
    def __init__(self,
                root:str,
                dataset_name,
                section_keyword,
                machine_type:str="ToyCar",
                section_ids=[],
                train=True,
                n_mels=128,
                n_fft=1024,
                hop_length=512,
                fmax=None,
                fmin=None,
                win_length=None,
                power=2.0,
                data_type = "dev",
                source_domain="mix",
                use_id = [],
                is_auto_download=False,
                cfg=None,
    ):
        super().__init__()

        self.use_id = use_id
        self.section_ids = section_ids
        self.machine_type = machine_type
        self.cfg = cfg or {}
        self.augment = AugmentationPipeline(self.cfg) if train else None

        target_dir = os.getcwd()+"/"+root+"raw/"+machine_type
        dir_name = "train" if train else "test"
        dir_names = [dir_name]
        
        self.mode = data_type == "dev"
        if train:
            dir_name = "train"
            if os.path.isdir(os.path.join(target_dir, "supplemental")):
                dir_names.append("supplemental")
        elif os.path.exists("{target_dir}/{dir_name}".format(target_dir=target_dir,dir_name="test_rename")):
            dir_name = "test_rename"
            self.mode = True
        else:
            dir_name = "test"

        self.pickle_dir = os.path.abspath(
            "{dir}/processed/{machine_type}/{dir_name}".format(
                dir=root,
                machine_type=machine_type,
                dir_name=dir_name
            )
        )
        if not (fmax or fmin):
            fmin_max = ""
        else:
            fmin_max = f"_f{fmin}-{fmax}"
        self.log_melspectrogram_dir = os.path.abspath(
            "{dir}/mels{n_mels}_fft{n_fft}_hop{hop_length}{fmin_max}".format(
                dir=self.pickle_dir,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                fmin_max=fmin_max
            )
        )

        target_dir = os.path.abspath("{}/".format(target_dir))
        if is_auto_download:
            # download DCASE2022T2 dataset
            com.download_raw_data(
                target_dir=target_dir,
                dir_name=dir_name,
                machine_type=machine_type,
                data_type=data_type,
                dataset=dataset_name,
                root=root
            )
        elif not os.path.exists(f"{target_dir}/{dir_name}"):
            raise FileNotFoundError(f"{target_dir}/{dir_name} is not directory and do not use auto download. \nplease download dataset or using auto download.")

        print("dataset dir is exists.")

        # get section names from wave file names
        section_names = [f"{section_keyword}_{section_id}" for section_id in section_ids]
        unique_section_names = np.unique(section_names)
        n_sections = len(unique_section_names)
        
        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        if not os.path.exists(self.log_melspectrogram_dir):
            os.makedirs(self.log_melspectrogram_dir, exist_ok=True)
        pickle_name = section_keyword
        for section_id in section_ids:
            pickle_name = f"{pickle_name}_{section_id}"
        pickle_name = f"{pickle_name}_{'+'.join(dir_names)}_{source_domain}_mel{n_fft}-{hop_length}"
        pickle_path = os.path.abspath(f"{self.log_melspectrogram_dir}/{pickle_name}.pickle")

        self.load_pre_process(
            pickle_name=pickle_name,
            target_dir=target_dir,
            pickle_path=pickle_path,
            n_sections=n_sections,
            unique_section_names=unique_section_names,
            dir_names=dir_names,
            train=train,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
            fmax=fmax,
            fmin=fmin,
            win_length=win_length,
        )
        if len(self.use_id) > 0:
            idx_list = [i for i, n in enumerate(np.argmax(self.condition, axis=1)) if int(section_ids[n]) in self.use_id]
        else:
            idx_list = list(range(len(self.condition)))
        self.data = [self.data[i] for i in idx_list]
        if len(self.y_true) > 0:
            self.y_true = [self.y_true[i] for i in idx_list]
        self.condition = [self.condition[i] for i in idx_list]
        self.basenames = [self.basenames[i] for i in idx_list]

        # getitem method setting
        self.dataset_len = len(self.data)
        self.getitem_fn = self.default_item

    def load_pre_process(
            self,
            pickle_name,
            target_dir,
            pickle_path,
            n_sections,
            unique_section_names,
            dir_names,
            train,
            n_mels,
            n_fft,
            hop_length,
            power,
            fmax,
            fmin,
            win_length,
        ):
 
        pickle_lockfile_path = os.path.abspath(f"{self.log_melspectrogram_dir}/{pickle_name}_lockfile")
        if os.path.isfile(pickle_path) and not os.path.isfile(pickle_lockfile_path):
            print("Setup has already been completed.")
            print(f"{pickle_path} is exists.")
            self.load_pickle(pickle_path=pickle_path)
            return
        else:
            pickle_lock = fasteners.InterProcessReaderWriterLock(pickle_lockfile_path)
            dataset_dir_lockfile_path = com.get_lockfile_path(target_dir=target_dir)
            is_enabled_dataset_dir_lock = os.path.isfile(dataset_dir_lockfile_path)
            if is_enabled_dataset_dir_lock:
                dataset_dir_lock = fasteners.InterProcessReaderWriterLock(dataset_dir_lockfile_path)
            try:
                pickle_lock.acquire_write_lock()
            except:
                print(f"{datetime.datetime.now()}\tcan not lock {pickle_lockfile_path}.")
                print(f"{pickle_path} is exists.")
                self.load_pickle(pickle_path=pickle_path)                
                return
            
            if os.path.exists(pickle_path):
                print(f"{pickle_path} is exists.")
                com.release_write_lock(
                    lock=pickle_lock,
                    lock_file_path=pickle_lockfile_path
                )
                self.load_pickle(pickle_path=pickle_path)
                return
            
            if is_enabled_dataset_dir_lock:
                try:
                    dataset_dir_lock.acquire_read_lock()
                except:
                    print(f"can not lock {dataset_dir_lockfile_path}.")
                    is_enabled_dataset_dir_lock = False
            
            self.data = []
            self.y_true = []
            self.condition = []
            self.basenames = []
            for section_idx, section_name in enumerate(unique_section_names):

                all_files = []
                all_labels = np.empty(0, float)
                all_conditions = []
                for _dir in dir_names:
                    if _dir == "supplemental":
                        query = os.path.abspath(f"{target_dir}/supplemental/{section_name}_*.wav")
                        _files = sorted(glob.glob(query))
                        _labels = np.zeros(len(_files))
                        cond_vec = np.eye(n_sections)[section_idx]
                        _cond = [cond_vec for _ in _files]
                    else:
                        _files, _labels, _cond = com.file_list_generator(
                            target_dir=target_dir,
                            section_name=section_name,
                            unique_section_names=unique_section_names,
                            dir_name=_dir,
                            mode=self.mode,
                            train=train,
                        )
                    all_files += list(_files)
                    all_labels = np.append(all_labels, _labels)
                    all_conditions += _cond

                for f, lab, cond in zip(all_files, all_labels, all_conditions):
                    self.basenames.append(os.path.basename(f))
                    y, sr = librosa.load(f, sr=16000, mono=True)
                    log_mel = librosa.feature.melspectrogram(
                        y=y,
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=n_mels,
                        fmin=fmin,
                        fmax=fmax,
                        power=power,
                    )
                    log_mel = librosa.power_to_db(log_mel, ref=np.max).astype(np.float32)
                    log_mel = log_mel[None]
                    self.data.append(log_mel)
                    if self.mode or train:
                        self.y_true.append(lab)
                    self.condition.append(cond)

            if is_enabled_dataset_dir_lock:
                com.release_read_lock(
                    lock=dataset_dir_lock,
                    lock_file_path=dataset_dir_lockfile_path
                )

            with open(pickle_path, 'wb') as f:
                pickle.dump((
                    self.data,
                    self.y_true,
                    self.condition,
                    self.basenames
                ), f, protocol=pickle.HIGHEST_PROTOCOL)

            count = 0
            while not com.is_enabled_pickle(pickle_path=pickle_path):
                assert count < 10
                print(f"{datetime.datetime.now()}\tpickle is not ready yet.")
                time.sleep(count+1)
                count+=1
                print(f"retry check pickle.\ncount : {count}")
            
            com.release_write_lock(
                lock=pickle_lock,
                lock_file_path=pickle_lockfile_path,
            )

    def load_pickle(self, pickle_path, pickle_lock=None, pickle_lock_file=None, retry_count=0, retry_lim=20, retry_delay_time=1):
        assert retry_count < retry_lim
        if pickle_lock:
            try:
                print("wait setting dataset")
                pickle_lock.acquire_read_lock()
            except:
                self.load_pickle(pickle_path=pickle_path)
                return
            com.release_read_lock(
                lock=pickle_lock,
                lock_file_path=pickle_lock_file,
            )
        print(f"load pickle : {pickle_path}")
        try:
            with open(pickle_path, 'rb') as f:
                self.data, self.y_true, self.condition, self.basenames = pickle.load(f)
        except FileNotFoundError:
            print(f"{datetime.datetime.now()}\tFileNotFoundError: can not load pickle.")
            time.sleep(retry_delay_time)
            print(f"retry load pickle : {retry_count+1}/{retry_lim}")
            self.load_pickle(
                pickle_path=pickle_path,
                pickle_lock=pickle_lock,
                pickle_lock_file=pickle_lock_file,
                retry_count=retry_count+1,
                retry_lim=retry_lim,
                retry_delay_time=retry_delay_time+1,
            )

    def __getitem__(self, index):
        """
        Returns:
            Tensor: input data
            Tensor: anomaly label
            Tensor: one-hot vector for conditioning (section id)
            int: start index
            str: file basename
        """
        return self.getitem_fn(index)

    def default_item(self, index):
        mel = self.data[index]
        if self.augment:
            mel = self.augment(mel.squeeze(0))[None]
        y_true = self.y_true[index] if len(self.y_true) > 0 else -1
        condition = self.condition[index]
        basename = self.basenames[index]
        return torch.from_numpy(mel), y_true, condition, basename, 0

    def __len__(self):
        return self.dataset_len

