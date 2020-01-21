import argparse

# Parameter Configuration Function

def get_cl_arguments():
    parser = argparse.ArgumentParser(description='Conducts federated learning experiment.')

    group = parser.add_argument_group('run settings')
    group.add_argument('--cuda', type=int, default=0,
                       help='Number of cuda device to use.')


    group = parser.add_argument_group('server settings')
    group.add_argument('--avg_method', type=str, default='fed_avg',
                       help='Averaging algorithm used by server. Options are "fed_avg", "1".')

    group.add_argument('--rounds', type=int, default=1000,
                       help='Total number of communication rounds.')

    group = parser.add_argument_group('client settings')

    group.add_argument('--split', type=str, default='random',
                       help='How the trainset is divided. Options are "random", "class".')

    group.add_argument('--speeds', type=str, default='same',
                       help='Relative speed of client hardware. Options are "same", "bimodal".')
    
    group.add_argument('--batchsize', type=int, default=32,
                       help='Batch size used by clients during local training.')
    args = parser.parse_args()
    print(args)
    return args