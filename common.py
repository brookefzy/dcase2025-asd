import yaml
import itertools
import argparse

########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("config.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

def param_to_args_list(params):
    """    Keys in ``params`` can be provided with or without the leading dashes.
    If no dashes are present we add ``--`` (or ``-`` for two character
    options such as ``lr``).  Values that are lists are expanded so that
    ``{"ids": [1, 2]}`` becomes ``['--ids', '1', '2']``.
    """
    # params = list(itertools.chain.from_iterable(zip(params.keys(), params.values())))
    args_list = []
    for key, value in params.items():
        if key.startswith('-'):
            opt = key
        else:
            opt = '--' + key

        args_list.append(opt)

        if isinstance(value, list):
            args_list.extend([str(v) for v in value])
        else:
            args_list.append(str(value))
    return args_list

########################################################################
# argparse setting
########################################################################
def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def float_or_None(v):
    if v.lower() in ["none", "null"]:
        return None
    return float(v)

def get_argparse():
    parser = argparse.ArgumentParser(
            description='Main function to call training for different AutoEncoders')
    parser.add_argument('--model', type=str, default='DCASE2023T2-AE', metavar='N',
                        help='train model name')
    parser.add_argument('--score', type=str, default="MSE", choices=["MSE", "MAHALA"])
    parser.add_argument('--seed', type=int, default=39876401, metavar='S',
                        help='random seed (default: 39876401)')
    
    parser.add_argument('--use_cuda', type=str2bool, default=True,
                        help='enables CUDA training')
    parser.add_argument('--gpu_id',type=int, nargs='*', default=[0,],
                        help='Specify GPU id')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--decision_threshold', type=float, default=0.9)
    parser.add_argument('--max_fpr', type=float, default=0.1)
    # multi-branch specific feature parameters
    parser.add_argument('--time_steps', type=int, default=512)

    # transformer AE parameters
    parser.add_argument('--transformer_hidden', type=int, default=512)
    parser.add_argument('--transformer_nhead', type=int, default=8)
    parser.add_argument('--transformer_ff', type=int, default=2048)
    parser.add_argument('--transformer_layers', type=int, default=6)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--decoder_layers', type=int, default=4)

    # embedding/flow parameters
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--diffusion_unet_dim', type=int, default=64)
    parser.add_argument('--diffusion_mults', type=int, nargs='*', default=[1,2,4])
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--diffusion_loss_type', type=str, default='l2')
    parser.add_argument('--flow_dim', type=int, default=384)
    parser.add_argument('--tau', type=float, default=0.07)

    # feature
    parser.add_argument('--n_mels',type=int, default=128, 
                        help='Length of the melfilter bank')
    parser.add_argument('--frames',type=int, default=5, 
                        help='Number of frames in a feature vector')
    parser.add_argument('--frame_hop_length',type=int, default=1, 
                        help='number of frames between successive feature')
    parser.add_argument('--n_fft',type=int, default=1024, 
                        help='length of the FFT window')
    parser.add_argument('--hop_length',type=int, default=512, 
                        help='number of samples between successive frames')
    parser.add_argument('--power', type=float, default=2.0)
    parser.add_argument('--fmin', type=float, default=0.0)
    parser.add_argument('--fmax', type=float_or_None, default=None)
    parser.add_argument('--win_length', type=float_or_None, default=None)

    # fit
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', '--learning_rate', default=0.03, type=float,
                        help='learning rate (default: 0.03)')
    parser.add_argument('--shuffle', type=str, default="full",
                        help='shuffle type (full , simple)')
    parser.add_argument('--validation_split', type=float, default=0.1)
    
    parser.add_argument('--w2', type=float, default=1.0)
    parser.add_argument('--w3', type=float, default=1.0)
    parser.add_argument('--w4', type=float, default=1.0)
    parser.add_argument('--w5', type=float, default=1.0)
    parser.add_argument('--maml_lr', type=float, default=1e-2)
    parser.add_argument('--maml_shots', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--specaug_p', type=float, default=1.0)
    parser.add_argument('--specaug_num', type=int, default=2)
    parser.add_argument('--specaug_freq', type=float, default=0.15)
    parser.add_argument('--specaug_time', type=float, default=0.15)
    parser.add_argument('--attr_input_dim', type=int, default=128)
    parser.add_argument('--attr_hidden', type=int, default=64)
    parser.add_argument('--attr_latent', type=int, default=10)

    # dataset
    parser.add_argument('--dataset_directory', type=str, default='data',
                        help='Where to parent dataset dir')
    parser.add_argument('--dataset', type=str, default='DCASE2023T2ToyCar', metavar='N',
                        help='dataset to use')
    parser.add_argument('-d', '--dev', action='store_true',
                        help='Use Development dataset')
    parser.add_argument('-e', '--eval', action='store_true',
                        help='Use Evaluation dataset')
    parser.add_argument('--use_ids', type=int, nargs='*', default=[],
                    help='Machine ID to be treated as nml data')
    parser.add_argument('--is_auto_download', type=str2bool, default=False,
                        help="Download dataset if not exist")

    # save data
    parser.add_argument('--result_directory', type=str, default='results/', metavar='N',
                        help='Where to store images')
    parser.add_argument('--export_dir',type=str, default='', 
                        help='Name of the directory to be generated under the Result directory.')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--ast_model', type=str, default='MIT/ast-finetuned-audioset-10-10-0.4593')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('-tag','--model_name_suffix',type=str, default='', 
                        help='Add a word to file name')

    # resume learning
    parser.add_argument('--restart',action='store_true', 
                        help='Resume learning with checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default="",
                        help="Using checkpoint file path. default: this checkpoint")
    parser.add_argument('--train_only', action='store_true', default=False,
                        help='Run train only')
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='Run test only')
    
    return parser