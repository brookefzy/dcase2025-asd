import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Subset
from sklearn.model_selection import train_test_split
import sys

from datasets.loader_common import get_machine_type_dict
from datasets.dcase_dcase202x_t2_loader import DCASE202XT2Loader
from torch.utils.data import ConcatDataset
import copy
import numpy as np

def pad_collate(batch):
    """Pad variable-length spectrograms along the time axis."""
    feats, labels, conds, names, _ = zip(*batch)
    max_T = max(f.shape[-1] for f in feats)
    T_fix = 512
    feats = [F.pad(f, (0, T_fix - f.shape[-1])) for f in feats]
    feats = torch.stack(feats)
    labels = torch.tensor(labels)
    conds = torch.from_numpy(np.stack(conds))
    return feats, labels, conds, list(names)

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
    }

    def __init__(self,datasets_str):
        if '+' in datasets_str:
            self.data = MultiDCASE202XT2
        else:
            self.data = Datasets.DatasetsDic[datasets_str]

    def show_list():
        return Datasets.DatasetsDic.keys()

