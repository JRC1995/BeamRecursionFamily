import argparse
from argparse import ArgumentParser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser(description="LanguageProcessors Arguments")
    parser.add_argument('--model', type=str, default="HEBT_GRC",
                        choices=["CRvNN", "OM", "BT_GRC", "BT_GRC_OS",
                                 "GT_GRC", "EGT_GRC",
                                 "EBT_GRC", "EBT_GRC512", "EBT_GRC_noslice", "EBT_GRC512_noslice",
                                 "GAU_IN", "EGT_GAU_IN", "EBT_GAU_IN",
                                 "S4DStack", "BalancedTreeGRC", "HGRC", "HCRvNN", "HOM",
                                 "HEBT_GRC", "HEBT_GRC_noSSM",
                                 "HEBT_GRC_noRBA", "HEBT_GRC_small", "HEBT_GRC_random",
                                 "HEBT_GRC_chunk20", "HEBT_GRC_chunk10",
                                 "CRvNN_nohalt","MEGA"])
    parser.add_argument('--no_display', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--display_params', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--test', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--chunk_mode_inference', type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument('--model_type', type=str, default="classifier",
                        choices=["sentence_pair", "classifier", "seq_label", "sentence_pair2", "classifier2"])
    parser.add_argument('--dataset', type=str, default="cifar10_lra",
                        choices=["proplogic", "proplogic_C", "listopsc", "listopsd",
                                 "SST2", "SST5", "MNLIdev", "IMDB", "IMDB_lra", "SNLI",
                                 "AAN_lra", "BoolQ", "listops_lra", "QQP", "listopsmix", "cifar10_lra",
                                 "pathfinder_lra",
                                 "listops200speed",
                                 "listops500speed",
                                 "listops900speed",
                                 "listops_lra_speed", "IMDB_lra_speed4000"])
    parser.add_argument('--times', type=int, default=3)
    parser.add_argument('--initial_time', type=int, default=0)
    parser.add_argument('--truncate_k', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--example_display_step', type=int, default=500)
    parser.add_argument('--load_checkpoint', type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument('--reproducible', type=str2bool, default=True, const=True, nargs='?')
    return parser
