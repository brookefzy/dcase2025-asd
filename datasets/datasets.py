import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
import sys
# ---- ADD THESE IMPORTS -----------------------------------------------------
import csv
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from datasets.augmentations import AugmentationPipeline
# -----------------------------------------------------------------------------

from datasets.loader_common import get_machine_type_dict
from datasets.dcase_dcase202x_t2_loader import DCASE202XT2Loader
from torch.utils.data import ConcatDataset
import copy
import numpy as np

def pad_collate(batch):
    """Pad variable-length spectrograms along the time axis."""
    first = batch[0]
    if len(first) == 5:  # legacy format from ``DCASE202XT2Loader``
        feats, labels, conds, names, _ = zip(*batch)
        max_T = max(f.shape[-1] for f in feats)
        T_fix = 512
        feats = [F.pad(f, (0, T_fix - f.shape[-1])) for f in feats]
        feats = torch.stack(feats)
        labels = torch.tensor(labels)
        conds = torch.from_numpy(np.stack(conds))
        return feats, labels, conds, list(names)
    else:
        # expected format: (feat, attr_vec, label)
        feats, attrs, labels = zip(*batch)
        max_T = max(f.shape[-1] for f in feats)
        T_fix = 512
        feats = [F.pad(f, (0, T_fix - f.shape[-1])) for f in feats]
        feats = torch.stack(feats)
        attrs = torch.stack(attrs)
        labels = torch.tensor(labels)
        return feats, attrs, labels

class IndustrialDataset(Dataset):
    """Minimal dataset for small industrial examples.

    The dataset directory is expected to contain ``train`` and ``test``
    folders with ``.wav`` files.  When ``use_attribute`` is ``True`` the
    file ``attribute_00.csv`` located under the same directory is used to
    build a simple attribute look‑up table.
    """

    def __init__(self, wav_root: Path, split: str,
                 use_attribute: bool = False,
                 attribute_name: str = "attribute_00.csv",
                 cfg: dict | None = None):
        self.wav_root = Path(wav_root)
        self.split = split
        self.use_attribute = use_attribute
        self.cfg = cfg or {}

        self.files = sorted(self.wav_root.glob(f"{split}/*.wav"))
        # naive label: 0 for normal, 1 for anomaly if "anomaly" in filename
        self.labels = [int("anomaly" in f.name.lower()) for f in self.files]

        if self.use_attribute:
            # ------------------------------------------------------------------
            # Load ``attribute_00.csv`` and build token‑to‑index mapping for each
            # attribute column found in the CSV.  The lookup is stored in
            # ``self.attr_lookup`` and ``self.attr_map`` for one‑hot expansion.
            # ------------------------------------------------------------------
            self.attr_lookup: dict[str, list[str]] = defaultdict(list)
            csv_path = self.wav_root / attribute_name
            if csv_path.exists():
                with open(csv_path, newline="") as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        stem = Path(row["file_name"]).stem
                        tokens = [v for k, v in row.items() if k != "file_name"]
                        self.attr_lookup[stem] = tokens

            # Build vocabularies per attribute position (variable length)
            max_len = max((len(v) for v in self.attr_lookup.values()), default=0)
            vocab: list[set[str]] = [set() for _ in range(max_len)]
            for vals in self.attr_lookup.values():
                for i, token in enumerate(vals):
                    if token:
                        vocab[i].add(token)
            self.attr_vocab = vocab
            self.attr_map = [
                {tok: i for i, tok in enumerate(sorted(col))}
                for col in self.attr_vocab
            ]
        else:
            self.attr_lookup = self.attr_vocab = self.attr_map = None

        # feature parameters
        self.sr = self.cfg.get("sr", 16000)
        self.n_fft = self.cfg.get("n_fft", 1024)
        self.hop_length = self.cfg.get("hop_length", 512)
        self.n_mels = self.cfg.get("n_mels", 128)
        self.win_length = self.cfg.get("win_length", self.n_fft)
        self.fmin = self.cfg.get("fmin", 0.0)
        self.fmax = self.cfg.get("fmax", None)
        self.power = self.cfg.get("power", 2.0)
        self.time_steps = self.cfg.get("time_steps", 512)
        self.augment = AugmentationPipeline(self.cfg) if split == "train" else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_path = self.files[idx]
        rel_path = wav_path.relative_to(self.wav_root)
        import soundfile as sf
        import librosa
        import numpy as np

        signal, sr = sf.read(wav_path)
        if sr != self.sr:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sr)
        log_mel = librosa.feature.melspectrogram(
            y=signal,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=self.power,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        log_mel = librosa.power_to_db(log_mel, ref=np.max).astype(np.float32)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        if log_mel.shape[1] < self.time_steps:
            pad = self.time_steps - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant")
        else:
            log_mel = log_mel[:, : self.time_steps]

        feat = torch.from_numpy(log_mel)[None]
        if self.augment:
            feat = self.augment(feat.squeeze(0))[None]

        label = self.labels[idx]

        if self.use_attribute:
            stem = Path(rel_path).stem
            tokens = self.attr_lookup.get(stem, [])
            one_hot = []
            for i, m in enumerate(self.attr_map):
                vec = torch.zeros(len(m))
                if i < len(tokens) and tokens[i] in m:
                    vec[m[tokens[i]]] = 1.0
                one_hot.append(vec)
            attr_vec = torch.cat(one_hot) if one_hot else torch.empty(0)
        else:
            attr_vec = torch.empty(0)

        return feat, attr_vec, label

class DCASE202XT2(object):
    def __init__(self, args):
        self.width   = args.frames
        self.height  = args.n_mels
        self.channel = 1
        self.input_dim = self.width*self.height*self.channel
        shuffle = args.shuffle
        batch_sampler = None
        batch_size = args.batch_size
        # dev/eval mode flag used by some loaders
        self.mode = args.dev
        print("input dim: %d" % (self.input_dim))

        dataset_name = args.dataset[:11]
        machine_type = args.dataset[11:]
        self.dataset_name = dataset_name
        self.machine_type = machine_type
        self.dataset_str = f"{dataset_name}{machine_type}"
        if args.eval:
            data_path = f'{args.dataset_directory}/{dataset_name.lower()}/eval_data/'
            data_type = "eval"
        elif args.dev:
            data_path = f'{args.dataset_directory}/{dataset_name.lower()}/dev_data/'
            data_type = "dev"
        else:
            print("incorrect argument")
            print("please set option argument '--dev' or '--eval'")
            sys.exit()

        self.machine_type_dict = get_machine_type_dict(dataset_name, mode=args.dev)["machine_type"]
        self.section_id_list = self.machine_type_dict[machine_type][data_type]
        self.num_classes = len(self.section_id_list)
        print("num classes: %d" % (self.num_classes))
        self.id_list = [int(machine_id) for machine_id in self.section_id_list]
        section_keyword = get_machine_type_dict(dataset_name, mode=args.dev)["section_keyword"]
        train_data = DCASE202XT2Loader(
                data_path,
                dataset_name=dataset_name,
                section_keyword=section_keyword,
                machine_type=machine_type,
                train=True,
                section_ids=self.section_id_list,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                power=args.power,
                fmax=args.fmax,
                fmin=args.fmin,
                win_length=args.win_length,
                data_type=data_type,
                use_id=args.use_ids,
                is_auto_download=args.is_auto_download,
                cfg=vars(args),
                )

        train_index, valid_index = train_test_split(range(len(train_data)), test_size=args.validation_split)
        self.train_dataset = Subset(train_data, train_index)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            batch_sampler=batch_sampler,
            collate_fn=pad_collate,
        )
        self.valid_dataset   = Subset(train_data, valid_index)
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            batch_sampler=batch_sampler,
            collate_fn=pad_collate,
        )

        self.test_loader = []
        if args.train_only:
            return
        for id in self.section_id_list:
           _test_loader = DCASE202XT2Loader(
                data_path,
                dataset_name=dataset_name,
                section_keyword=section_keyword,
                machine_type=machine_type,
                train=False,
                section_ids=[id],
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                power=args.power,
                fmax=args.fmax,
                fmin=args.fmin,
                win_length=args.win_length,
                data_type=data_type,
                is_auto_download=args.is_auto_download,
                cfg=vars(args),
           )

           self.test_loader.append(
                torch.utils.data.DataLoader(
                    _test_loader,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=pad_collate,
                )
           )
           self.mode = args.dev or _test_loader.mode


class MultiDCASE202XT2(object):
    """Combine multiple machine types into a single dataset."""

    def __init__(self, args):
        self.datasets = []
        names = args.dataset.split("+")
        for name in names:
            tmp_args = copy.deepcopy(args)
            tmp_args.dataset = name
            self.datasets.append(DCASE202XT2(tmp_args))

        self.width = self.datasets[0].width
        self.height = self.datasets[0].height
        self.channel = 1
        self.input_dim = self.width * self.height * self.channel

        shuffle = args.shuffle
        batch_sampler = None
        batch_size = args.batch_size

        self.train_dataset = ConcatDataset([d.train_dataset for d in self.datasets])
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            batch_sampler=batch_sampler,
            collate_fn=pad_collate,
        )

        self.valid_dataset = ConcatDataset([d.valid_dataset for d in self.datasets])
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            batch_sampler=batch_sampler,
            collate_fn=pad_collate,
        )

        self.test_loader = []
        if not args.train_only:
            for d in self.datasets:
                self.test_loader.extend(d.test_loader)

        self.mode = self.datasets[0].mode
        self.section_id_list = []
        self.id_list = []
        for d in self.datasets:
            self.section_id_list += d.section_id_list
            self.id_list += d.id_list
        self.num_classes = len(self.section_id_list)


class IndustrialData(object):
    """Simple loader for ``IndustrialDataset`` using the same interface as
    ``DCASE202XT2``."""

    def __init__(self, args):
        cfg = vars(args)
        root = Path(args.dataset_directory) / "industrial"

        base_ds = IndustrialDataset(root, "train", use_attribute=cfg.get("use_attribute", False), cfg=cfg)
        train_idx, valid_idx = train_test_split(range(len(base_ds)), test_size=args.validation_split)

        self.width = cfg.get("time_steps", 512)
        self.height = cfg.get("n_mels", 128)
        self.channel = 1
        self.input_dim = self.width * self.height * self.channel

        self.train_dataset = Subset(base_ds, train_idx)
        self.valid_dataset = Subset(base_ds, valid_idx)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            collate_fn=pad_collate,
            num_workers=args.num_workers,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=pad_collate,
            num_workers=args.num_workers,
        )

        self.test_loader = []
        if not args.train_only:
            test_ds = IndustrialDataset(root, "test", use_attribute=cfg.get("use_attribute", False), cfg=cfg)
            self.test_loader.append(
                torch.utils.data.DataLoader(
                    test_ds,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=pad_collate,
                    num_workers=args.num_workers,
                )
            )

        # Compatibility attributes
        self.mode = True
        self.section_id_list = [0]
        self.id_list = [0]
        self.num_classes = 1


class Datasets:
    DatasetsDic = {
        'DCASE2025T2ToyRCCar':DCASE202XT2,
        'DCASE2025T2ToyPet':DCASE202XT2,
        'DCASE2025T2HomeCamera':DCASE202XT2,
        'DCASE2025T2AutoTrash':DCASE202XT2,
        'DCASE2025T2Polisher':DCASE202XT2,
        'DCASE2025T2ScrewFeeder':DCASE202XT2,
        'DCASE2025T2BandSealer':DCASE202XT2,
        'DCASE2025T2CoffeeGrinder':DCASE202XT2,
        'DCASE2025T2ToyCar':DCASE202XT2,
        'DCASE2025T2ToyTrain':DCASE202XT2,
        'DCASE2025T2bearing':DCASE202XT2,
        'DCASE2025T2fan':DCASE202XT2,
        'DCASE2025T2gearbox':DCASE202XT2,
        'DCASE2025T2slider':DCASE202XT2,
        'DCASE2025T2valve':DCASE202XT2,
        'DCASE2024T23DPrinter':DCASE202XT2,
        'DCASE2024T2AirCompressor':DCASE202XT2,
        'DCASE2024T2Scanner':DCASE202XT2,
        'DCASE2024T2ToyCircuit':DCASE202XT2,
        'DCASE2024T2HoveringDrone':DCASE202XT2,
        'DCASE2024T2HairDryer':DCASE202XT2,
        'DCASE2024T2ToothBrush':DCASE202XT2,
        'DCASE2024T2RoboticArm':DCASE202XT2,
        'DCASE2024T2BrushlessMotor':DCASE202XT2,
        'DCASE2024T2ToyCar':DCASE202XT2,
        'DCASE2024T2ToyTrain':DCASE202XT2,
        'DCASE2024T2bearing':DCASE202XT2,
        'DCASE2024T2fan':DCASE202XT2,
        'DCASE2024T2gearbox':DCASE202XT2,
        'DCASE2024T2slider':DCASE202XT2,
        'DCASE2024T2valve':DCASE202XT2,
        'DCASE2023T2bandsaw':DCASE202XT2,
        'DCASE2023T2bearing':DCASE202XT2,
        'DCASE2023T2fan':DCASE202XT2,
        'DCASE2023T2grinder':DCASE202XT2,
        'DCASE2023T2gearbox':DCASE202XT2,
        'DCASE2023T2shaker':DCASE202XT2,
        'DCASE2023T2slider':DCASE202XT2,
        'DCASE2023T2ToyCar':DCASE202XT2,
        'DCASE2023T2ToyDrone':DCASE202XT2,
        'DCASE2023T2ToyNscale':DCASE202XT2,
        'DCASE2023T2ToyTank':DCASE202XT2,
        'DCASE2023T2ToyTrain':DCASE202XT2,
        'DCASE2023T2Vacuum':DCASE202XT2,
        'DCASE2023T2valve':DCASE202XT2,
        'DCASE2022T2bearing':DCASE202XT2,
        'DCASE2022T2fan':DCASE202XT2,
        'DCASE2022T2gearbox':DCASE202XT2,
        'DCASE2022T2slider':DCASE202XT2,
        'DCASE2022T2ToyCar':DCASE202XT2,
        'DCASE2022T2ToyTrain':DCASE202XT2,
        'DCASE2022T2valve':DCASE202XT2,
        'DCASE2021T2fan':DCASE202XT2,
        'DCASE2021T2gearbox':DCASE202XT2,
        'DCASE2021T2pump':DCASE202XT2,
        'DCASE2021T2slider':DCASE202XT2,
        'DCASE2021T2ToyCar':DCASE202XT2,
        'DCASE2021T2ToyTrain':DCASE202XT2,
        'DCASE2021T2valve':DCASE202XT2,
        'DCASE2020T2ToyCar':DCASE202XT2,
        'DCASE2020T2ToyConveyor': DCASE202XT2,
        'DCASE2020T2fan':DCASE202XT2,
        'DCASE2020T2valve':DCASE202XT2,
        'DCASE2020T2slider':DCASE202XT2,
        'DCASE2020T2pump':DCASE202XT2,
        'Industrial': IndustrialData,
        'IndustrialDataset': IndustrialData,
    }

    def __init__(self,datasets_str):
        if '+' in datasets_str:
            self.data = MultiDCASE202XT2
        else:
            self.data = Datasets.DatasetsDic[datasets_str]

    def show_list():
        return Datasets.DatasetsDic.keys()

