import argparse
import numpy as np
import torch
from hmm_discretization import Discrete_Model_Optimization
from torch.distributions import MultivariateNormal


def eval_estimates(n_components, obs, m, c):
    dists = [MultivariateNormal(torch.tensor(m[i]), torch.tensor(c[i]))
             for i in range(n_components)]
    avg = np.mean(
            [torch.mean(torch.exp(dist.log_prob(obs))).item()
             for dist in dists])
    return avg


def main():

    # Adding parsers
    parser = argparse.ArgumentParser()
    # Number of Hidden states
    parser.add_argument(
        '--n_hidden',
        type=int,
        help='Choose number of hidden state',
        default=2,
        required=False
    )
    # Sample size
    parser.add_argument(
        '--sample_size',
        type=int,
        help='Choose sample size of observations',
        default=5000,
        required=False
    )
    parser.add_argument(
        '--grid_size',
        type=int,
        help='Choose grid size',
        default=2 ** 6,
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
        default=501,
        required=False
    )
    parser.add_argument(
        '--discretization_method',
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
        '--random_init',
        action=argparse.BooleanOptionalAction,
        help='Choose if initialization will be random or not',
        default=False,
        required=False
    )
    parser.add_argument(
        '--train_model',
        action=argparse.BooleanOptionalAction,
        help='Training the model to estimate parameters',
        default=False,
        required=False
    )

    args = parser.parse_args()

    # 2 Hidden States
    means2 = np.array([
        [1.0, 4.0], [5.0, -3.0]
        ])
    covars2 = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.5], [0.5, 1.0]]
        ])
    startprob2 = np.array([0.5, 0.5])
    transmat2 = np.array([[0.2, 0.8], [0.5, 0.5]])

    # 3 Hidden States
    means3 = np.array([
        [1.0, 4.0], [5.0, -3.0], [-3.0, -4.0]
        ])
    covars3 = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.5], [0.5, 1.0]],
        [[1.5, 1.0], [1.0, 1.5]]
        ])
    startprob3 = np.array([0.5, 0.2, 0.3])
    transmat3 = np.array([[0.2, 0.6, 0.2], [0.5, 0.4, 0.1], [0.15, 0.35, 0.5]])

    if args.n_hidden == 2:
        STARTPROB = startprob2
        TRANSMAT = transmat2
        MEANS = means2
        COVARS = covars2

    elif args.n_hidden == 3:
        STARTPROB = startprob3
        TRANSMAT = transmat3
        MEANS = means3
        COVARS = covars3

    else:
        print("Please choose another number of hidden states")
        return

    N_DIMENSIONS = 2
    N_COMPONENTS = len(MEANS)
    N_FEATURES = N_COMPONENTS
    SAMPLE_SIZE = args.sample_size
    GRID_SIZE = args.grid_size
    DISCRETIZATION_METHOD = args.discretization_method
    LR = args.lr
    N_EPOCHS = args.n_epochs
    RANDOM_INIT = args.random_init
    LOSS_FUNCTION = args.loss_function

    mod_opt = Discrete_Model_Optimization(N_COMPONENTS,
                                          N_FEATURES,
                                          N_DIMENSIONS,
                                          SAMPLE_SIZE,
                                          GRID_SIZE,
                                          DISCRETIZATION_METHOD,
                                          LR,
                                          N_EPOCHS,
                                          STARTPROB,
                                          TRANSMAT,
                                          MEANS,
                                          COVARS,
                                          RANDOM_INIT)

    if args.train_model:
        params, losses = mod_opt.training(
            LOSS_FUNCTION, verbose=False)
        eval_estimates(
            N_COMPONENTS,
            torch.tensor(mod_opt.observations_test),
            params[0],
            params[1])
        print(losses)

    else:
        mod_opt.plot_initial()


if __name__ == '__main__':
    main()
