import argparse
import matplotlib.pyplot as plt
import numpy as np
from src.hmm_discretization import Discrete_Model_Optimization
from src.data_collection import generate_parameters, paper_experiments
from parameters_data.means_and_covars import means_opts, covars_opts


def main():
    # Adding parsers
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_hidden',
        type=int,
        help='Choose number of hidden state, Implemented: 2 or 3',
        default=2,
        required=False
    )
    parser.add_argument(
        '--n_dim',
        type=int,
        help='Choose number of dimensions',
        default=2,
        required=False
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        help='Choose sample size of observations',
        default=300,
        required=False
    )
    parser.add_argument(
        '--grid_size',
        type=int,
        help='Choose grid size',
        default=6,
        required=False
    )
    parser.add_argument(
        '--lr',
        type=int,
        help='Choose learning rate',
        default=0.01,
        required=False
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        help='Choose number of epochs for training',
        default=200,
        required=False
    )
    parser.add_argument(
        '--discr_method',
        type=str,
        help='Choose discretization method',
        default='random',
        required=False
    )
    parser.add_argument(
        '--loss_function',
        type=str,
        help='Choose Loss Function',
        default='KLD',
        required=False
    )
    parser.add_argument(
        '--random_params',
        action=argparse.BooleanOptionalAction,
        help='Choose if initializing random parameters',
        default=False,
        required=False
    )
    parser.add_argument(
        '--compare_discr',
        action=argparse.BooleanOptionalAction,
        help='Comparing discretization techniques',
        default=False,
        required=False
    )

    args = parser.parse_args()

    means_rand, covars_rand, STARTPROB, TRANSMAT = generate_parameters(
        args.n_dim,
        args.n_hidden)
    if args.random_params:
        MEANS = means_rand
        COVARS = covars_rand
    else:
        MEANS = means_opts[f'means{args.n_hidden}_{args.n_dim}']
        COVARS = covars_opts[f'covars{args.n_hidden}_{args.n_dim}']
    N_DIMENSIONS = args.n_dim
    N_COMPONENTS = args.n_hidden
    N_FEATURES = N_COMPONENTS
    SAMPLE_SIZE = args.sample_size
    GRID_SIZE = args.grid_size
    DISCR_METHOD = args.discr_method
    LR = args.lr
    N_EPOCHS = args.n_epochs
    RANDOM_INIT = True
    LOSS_FUNCTION = args.loss_function
    if args.compare_discr:
        _, _, _, _, _ = paper_experiments(
            SAMPLE_SIZE,
            GRID_SIZE,
            N_DIMENSIONS,
            N_COMPONENTS,
            MEANS,
            COVARS,
            STARTPROB,
            TRANSMAT,
            it_experiment=1
        )
        print('a')
    else:
        mod_opt = Discrete_Model_Optimization(
            N_COMPONENTS,
            N_FEATURES,
            N_DIMENSIONS,
            SAMPLE_SIZE,
            2 ** GRID_SIZE,
            DISCR_METHOD,
            LR,
            N_EPOCHS,
            STARTPROB,
            TRANSMAT,
            MEANS,
            COVARS,
            RANDOM_INIT
        )
        PARAMS, LOSSES = mod_opt.training(LOSS_FUNCTION)
        plt.plot(np.arange(len(LOSSES)), LOSSES)
        title = f"{LOSS_FUNCTION} Loss. "\
            f"Sample Size={SAMPLE_SIZE}. Grid=2^{GRID_SIZE}"
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss")
        plt.title(title)
        plt.show()
        mod_opt.plot_initial()


if __name__ == '__main__':
    main()
