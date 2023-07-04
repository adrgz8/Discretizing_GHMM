"""
Python code for discretization techique utilized as an
alternative to Baum Welch Algorithm to optimize
Gaussian Hidden Markov Model parameters

By: Adrian Rodriguez Gonzalez
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter
from hmmlearn import hmm
from IPython.display import clear_output
from itertools import permutations
from torch import nn, optim
from torch.distributions import MultivariateNormal
from scipy.stats import qmc
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams["legend.markerscale"] = 0.3
torch.set_printoptions(sci_mode=False)


class Discrete_Model(nn.Module):
    def __init__(self,
                 n_components,
                 n_features,
                 n_dimensions,
                 sample_size,
                 startprob,
                 transmat,
                 means,
                 covars):
        super(Discrete_Model, self).__init__()

        self.n_components = n_components
        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.sample_size = sample_size

        est_startprob = startprob
        est_transmat = transmat
        est_means = means
        est_covars = np.linalg.cholesky(covars)

        # Converting Means, Covariances and S matrix to Tensors
        self.t_means = nn.Parameter(
            torch.tensor(est_means),
            requires_grad=True)
        self.t_covars = nn.Parameter(
            torch.tensor(est_covars),
            requires_grad=True)
        self.t_S_matrix = nn.Parameter(
            torch.tensor(np.log(np.dot(np.diag(est_startprob), est_transmat))),
            requires_grad=True)

    def forward(self, obs):
        lower_cov = torch.tril(self.t_covars)
        cov_matrix = lower_cov @ torch.transpose(lower_cov, dim0=1, dim1=2)
        ns = [MultivariateNormal(self.t_means[i], cov_matrix[i])
              for i in range(self.n_components)]
        P = torch.stack([torch.exp(n.log_prob(obs)) for n in ns]).T
        P = P / P.sum(axis=0, keepdims=True)
        S = torch.exp(self.t_S_matrix) / torch.exp(self.t_S_matrix).sum()
        Q_matrix = P @ S @ P.T
        return Q_matrix

    def retrieve_parameters(self):
        mean_retrieve = self.t_means
        covars_retrieve = self.t_covars @ torch.transpose(
            self.t_covars,
            dim0=1,
            dim1=2
            )
        S_mat = torch.exp(self.t_S_matrix) / torch.exp(self.t_S_matrix).sum()
        startprob_retrieve = torch.sum(S_mat, dim=1)
        transmat_retrieve = S_mat / startprob_retrieve.unsqueeze(dim=1)
        return (
            mean_retrieve.detach().numpy(),
            covars_retrieve.detach().numpy(),
            startprob_retrieve.detach().numpy(),
            transmat_retrieve.detach().numpy()
            )


class Discrete_Model_Optimization(hmm.GaussianHMM):
    def __init__(self,
                 n_components,
                 n_features,
                 n_dimensions,
                 sample_size,
                 grid_size,
                 random_method,
                 lr,
                 epochs,
                 startprob,
                 transmat,
                 means,
                 covars,
                 random_init=True,
                 test_obs=None):
        super(Discrete_Model_Optimization, self).__init__(
            n_components=n_components,
            covariance_type="full"
            )
        self.n_components = n_components
        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.sample_size = sample_size
        self.grid_size = grid_size
        self.lr = lr
        self.epochs = epochs
        self.startprob_ = startprob
        self.transmat_ = transmat
        self.means_ = means
        self.covars_ = covars
        if test_obs is None:
            self.observations, _ = self.sample(self.sample_size)
            self.observations_test, _ = self.sample(int(self.sample_size*0.2))
        else:
            self.observations = test_obs[0]
            self.observations_test = test_obs[1]

        if random_init:
            # Random Start Probability
            r_startprob = np.random.random(size=self.n_components)
            r_startprob = r_startprob / sum(r_startprob)

            # Using KMeans for Means initialization
            kmeans = KMeans(n_clusters=self.n_components, n_init='auto')
            kmeans.fit(self.observations)
            r_means = kmeans.cluster_centers_
            r_means = self.ordering_vals(means, r_means)

            # Using Empirical Covariance for Covariance matrix initialization
            obs_sep = [self.observations[np.where(kmeans.labels_ == i)]
                       for i in range(self.n_components)]
            emp_cov = EmpiricalCovariance()
            r_covars = [emp_cov.fit(i).covariance_ for i in obs_sep]
            r_covars = np.stack(r_covars, axis=0)
            r_covars = self.ordering_vals(covars, r_covars)

            # Initializing transition matrix based on the observations
            mat = np.zeros((self.n_components, self.n_components))
            labs = kmeans.labels_
            for i in range(len(self.observations)-2):
                idx1, idx2 = labs[i:i+2]
                mat[idx1][idx2] += 1
            r_transmat = np.array([mat[i] / mat.sum(axis=1)[i]
                                   for i in range(self.n_components)])
            r_transmat = self.ordering_vals(transmat, r_transmat)

            self.model = Discrete_Model(self.n_components,
                                        self.n_features,
                                        self.n_dimensions,
                                        self.sample_size,
                                        r_startprob,
                                        r_transmat,
                                        r_means,
                                        r_covars)

        else:
            self.model = Discrete_Model(self.n_components,
                                        self.n_features,
                                        self.n_dimensions,
                                        self.sample_size,
                                        startprob,
                                        transmat,
                                        means,
                                        covars)

        # Ranges for random variables
        self.x_min = self.observations[:, 0].min()
        self.x_max = self.observations[:, 0].max()
        self.y_min = self.observations[:, 1].min()
        self.y_max = self.observations[:, 1].max()
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min

        # Discrete sequence of pseudo-random & KNN indices
        if random_method == 'random':
            self.discrete_seq = self.get_rand_seq()
        elif random_method == 'grid':
            self.discrete_seq = self.get_grid_seq()
        elif random_method == 'LH':
            self.discrete_seq = self.get_LatinHypercube_seq()
        elif random_method == 'Sobol':
            self.discrete_seq = self.get_Sobol_seq()
        elif random_method == 'Halton':
            self.discrete_seq = self.get_Halton_seq()
        else:
            print('Choose a different random method')
            return
        self.knn_indices = self.get_knn_indices()

    def ordering_vals(self, val_real, val_est):
        permuts = list(permutations(list(range(self.n_components))))
        idx_min = np.argmin([np.abs(val_est[list(i)] - val_real).sum()
                             for i in permuts])
        ord_vals = val_est[list(permuts[idx_min])]
        return ord_vals

    # Functions with quasi-random sequences
    def get_rand_seq(self):
        lw = (self.x_min-(self.x_range*0.05), self.y_min-(self.y_range*0.05))
        hg = (self.x_max+(self.x_range*0.05), self.y_max+(self.x_range*0.05))
        rand_seq = np.random.default_rng().uniform(
            low=lw,
            high=hg,
            size=(self.grid_size, self.n_dimensions)
        )
        return rand_seq

    def get_grid_seq(self):
        n_cols = int(np.ceil(np.sqrt(self.grid_size)))
        n_rows = int(np.ceil(self.grid_size / n_cols))
        x = np.linspace(self.x_min, self.x_max, n_cols)
        y = np.linspace(self.y_min, self.y_max, n_rows)
        x, y = np.meshgrid(x, y)
        grid_seq = np.column_stack((x.ravel(), y.ravel()))
        return grid_seq[:self.grid_size]

    def get_LatinHypercube_seq(self):
        LH_vals = qmc.LatinHypercube(d=self.n_dimensions)
        LH_seq = LH_vals.random(int(self.grid_size))
        LH_seq[:, 0] = self.x_min + self.x_range * LH_seq[:, 0]
        LH_seq[:, 1] = self.y_min + self.y_range * LH_seq[:, 1]
        return LH_seq

    def get_Sobol_seq(self):
        Sob_seq = qmc.Sobol(d=self.n_dimensions).random(self.grid_size)
        Sob_seq[:, 0] = self.x_min + self.x_range * Sob_seq[:, 0]
        Sob_seq[:, 1] = self.y_min + self.y_range * Sob_seq[:, 1]
        return Sob_seq

    def get_Halton_seq(self):
        Hal_seq = qmc.Halton(d=self.n_dimensions).random(self.grid_size)
        Hal_seq[:, 0] = self.x_min + self.x_range * Hal_seq[:, 0]
        Hal_seq[:, 1] = self.y_min + self.y_range * Hal_seq[:, 1]
        return Hal_seq

    def get_knn_indices(self):
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(self.discrete_seq, range(self.grid_size))
        return knn.predict(self.observations)

    def get_discrete_obs(self, to_torch=False):
        discrete_obs = self.discrete_seq[self.knn_indices]
        if to_torch:
            return torch.Tensor(discrete_obs)
        return discrete_obs

    def plot_grid_with_values(self):
        # Ordering the number of times the Discrete observation was selected
        counts = Counter(self.knn_indices)
        for i in range(self.sample_size):
            counts[i] += 1
        ord_counts = dict(sorted(counts.items(), key=lambda x: x[0]))
        sizes = np.array(list(ord_counts.values()))
        sizes[np.where(sizes <= 1)] = 1
        sizes[np.where((sizes > 1) & (sizes <= 100))] = 16
        sizes[np.where((sizes > 100) & (sizes <= 500))] = 64
        sizes[np.where((sizes > 500) & (sizes <= 10000))] = 256
        sizes[np.where(sizes > 10000)] = 1024
        discrete_obs = self.get_discrete_obs()
        plt.scatter(
            self.discrete_seq[:, 0],
            self.discrete_seq[:, 1],
            color='gray',
            alpha=0.3,
            marker='+')
        plt.scatter(
            self.observations[:, 0],
            self.observations[:, 1],
            color='blue',
            alpha=0.3,
            marker='*',
            label='Continuous')
        plt.scatter(
            discrete_obs[:, 0],
            discrete_obs[:, 1],
            color='red',
            alpha=0.5,
            s=sizes,
            label='Discrete')
        plt.legend()
        plt.show()

    def plot_initial(self):
        Q_emp = self.compute_empirical_Q()
        Q_est = self.model(torch.tensor(self.discrete_seq)).detach().numpy()
        _, axis = plt.subplots(ncols=3, figsize=(15, 5))
        axis[0].imshow(Q_emp)
        axis[1].imshow(Q_est)
        axis[2] = self.plot_grid_with_values()
        plt.show()

    # Computes Empirical Q based on discretized values (Co-occurance matrix)
    def compute_empirical_Q(self, to_torch=False):
        Q_emp = np.zeros((self.grid_size, self.grid_size))
        for i in range(len(self.knn_indices)-2):
            idx1, idx2 = self.knn_indices[i:i+2]
            Q_emp[idx1, idx2] += 1
        Q_emp /= Q_emp.sum()
        if to_torch:
            return torch.tensor(Q_emp)
        return Q_emp

    # Fitting the model
    def training(self, loss_function='MSE', verbose=True):
        loss_acum = list()
        Q_emp = self.compute_empirical_Q(to_torch=True)
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        t_discrete_seq = torch.tensor(self.discrete_seq)
        if loss_function == 'MSE':
            loss_fun = nn.MSELoss(reduction='sum')
        elif loss_function == "KLD":
            loss_fun = nn.KLDivLoss(reduction='sum', log_target=True)
        else:
            print('Choose different Loss function')
            return
        for i in range(self.epochs):
            try:
                optimizer.zero_grad()
                if loss_function == 'MSE':
                    loss = loss_fun(self.model(t_discrete_seq), Q_emp)
                else:
                    loss = loss_fun(
                        nn.functional.log_softmax(
                            self.model(t_discrete_seq),
                            dim=1),
                        nn.functional.log_softmax(
                            Q_emp,
                            dim=1)
                        )
                loss.backward()
                optimizer.step()
                val_loss = np.round(loss.detach().numpy(), 6)
                if i % int(self.epochs / 10) == 0:
                    # clear_output()
                    (self.means_,
                     self.covars_,
                     self.startprob_,
                     self.transmat_,) = self.model.retrieve_parameters()
                    # val_loss = np.round(loss.detach().numpy(), 6)
                    if verbose:
                        print_S_matrix = np.dot(
                            np.diag(self.startprob_),
                            self.transmat_
                        )
                        print(f"Epoch: {i} / {self.epochs} | Loss: {val_loss}")
                        print(
                            "\n" "Means: " "\n", self.means_, "\n"
                            "\n" "Covars: " "\n", self.covars_, "\n"
                            "\n" "S_matrix: " "\n", print_S_matrix, "\n"
                            "\n" "Start prob: " "\n", self.startprob_, "\n"
                            "\n" "Trans mat: " "\n", self.transmat_)
                        print('-' * 50)
                    self.lr *= 0.9
                    optimizer = optim.Adam(
                        params=self.model.parameters(),
                        lr=self.lr
                    )
                if loss_function == 'MSE':
                    loss_acum.append(val_loss)
                else:
                    loss_acum.append(np.exp(val_loss))
            except KeyboardInterrupt:
                clear_output()
                print_S_matrix = np.dot(
                    np.diag(self.startprob_),
                    self.transmat_
                )
                print(f"After {i} epochs, values are: ")
                print(
                    "Means: " "\n", self.means_, "\n"
                    "Covars: " "\n", self.covars_, "\n"
                    "S_matrix: " "\n", print_S_matrix
                )
                print('-' * 50)
                plt.plot(np.arange(len(loss_acum)), loss_acum)
                plt.show()
                return self.model.retrieve_parameters(), loss_acum

        if verbose:
            plt.plot(np.arange(len(loss_acum)), loss_acum)
            plt.show()
        return self.model.retrieve_parameters(), loss_acum
