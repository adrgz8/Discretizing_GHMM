import pandas as pd
import matplotlib.pyplot as plt


def values_to_df(row_names,
                 col_names,
                 file_path,
                 n_hidd,
                 change_format=False):
    df = pd.read_csv(file_path)
    df.pop(df.columns[0])
    df = df.T
    df.columns = col_names
    df_loss = df.iloc[list(range(0, 40, 4))]
    df_score = df.iloc[list(range(1, 40, 4))]
    df_acc = df.iloc[list(range(2, 40, 4))]
    df_discr = df.iloc[list(range(3, 40, 4))]
    df_all = [df_loss, df_score, df_acc, df_discr]
    if change_format:
        df_names = ['Loss', 'Score', 'Acc', 'Discrepancy']
        for df_sel, name in zip(df_all, df_names):
            df_sel = df_sel.round(decimals=6)
            df_sel = df_sel.T
            df_sel.columns = row_names
            df_sel.to_csv(f'{name}_{n_hidd}.csv')
    return [df_loss, df_score, df_acc, df_discr]


def plotting_results(metric, name, n_hidden):
    disc_methods = ['Random', 'Grid', 'LH', 'Sobol', 'Halton']
    n_dims = ['2', '3', '4', '6', '8', '10']
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(8, 15))
    title_text = f"{name} for Discretization methods. " \
                 f"{n_hidden} hidden states"
    fig.suptitle(title_text, fontsize=16)
    for c, i in enumerate(n_dims):
        axes1 = axes[c, 0].twinx()
        axes2 = axes[c, 1].twinx()
        # axes[c, 0].bar(disc_methods, metric[f'{i}D with 2^6'][:5], color='y')
        # axes[c, 1].bar(disc_methods, metric[f'{i}D with 2^8'][:5], color='y')
        if c == 5:
            axes[c, 0].plot(disc_methods,
                            metric[f'{i}D with 2^6'][:5],
                            color='y',
                            marker='X',
                            linestyle=(0, (3, 1, 1, 1, 1, 1)),
                            label='MSE')
            axes[c, 1].plot(disc_methods,
                            metric[f'{i}D with 2^8'][:5],
                            color='y',
                            marker='X',
                            linestyle=(0, (3, 1, 1, 1, 1, 1)))
            axes1.plot(disc_methods,
                       metric[f'{i}D with 2^6'][5:],
                       color='b',
                       marker='D',
                       linestyle=(0, (1, 10)),
                       label='KLD')
            axes2.plot(disc_methods,
                       metric[f'{i}D with 2^8'][5:],
                       color='b',
                       marker='D',
                       linestyle=(0, (1, 10)))
        else:
            axes[c, 0].plot(disc_methods,
                            metric[f'{i}D with 2^6'][:5],
                            color='y',
                            marker='X',
                            linestyle=(0, (3, 1, 1, 1, 1, 1)))
            axes[c, 1].plot(disc_methods,
                            metric[f'{i}D with 2^8'][:5],
                            color='y',
                            marker='X',
                            linestyle=(0, (3, 1, 1, 1, 1, 1)))
            axes1.plot(disc_methods,
                       metric[f'{i}D with 2^6'][5:],
                       color='b',
                       marker='D',
                       linestyle=(0, (1, 10)))
            axes2.plot(disc_methods,
                       metric[f'{i}D with 2^8'][5:],
                       color='b',
                       marker='D',
                       linestyle=(0, (1, 10)))
        axes[c, 0].set_ylabel(f'MSE {name}', c='y')
        axes[c, 1].set_ylabel(f'MSE {name}', c='y')
        axes1.set_ylabel(f'KLD {name}', c='b')
        axes2.set_ylabel(f'KLD {name}', c='b')
        axes[c, 0].set_title(f'{i} Dimensions, 2^6 Grid Size')
        axes[c, 1].set_title(f'{i} Dimensions, 2^8 Grid Size')
        axes1.set_xlabel('Discretization Method')
        axes2.set_xlabel('Discretization Method')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig.legend()
    plt.show()


if __name__ == '__main__':
    col_names = ['2D with 2^6',
                 '2D with 2^8',
                 '3D with 2^6',
                 '3D with 2^8',
                 '4D with 2^6',
                 '4D with 2^8',
                 '6D with 2^6',
                 '6D with 2^8',
                 '8D with 2^6',
                 '8D with 2^8',
                 '10D with 2^6',
                 '10D with 2^8']
    row_names = ['Random MSE',
                 'Grid MSE',
                 'Latin Hypercube MSE',
                 'Sobol MSE',
                 'Halton MSE',
                 'Random KLD',
                 'Grid KLD',
                 'Latin Hypercube KLD',
                 'Sobol KLD',
                 'Halton KLD']
    metric_names = ['Loss', 'Log-Prob', 'Accuracy', 'Discrepancy']
    n_hiddens = [2, 3]
    for n_hidden in n_hiddens:
        mets = values_to_df(row_names,
                            col_names,
                            f'test_results/metrics/Results_ND_{n_hidden}h.csv',
                            n_hidden)
        for metric, metric_name in zip(mets, metric_names):
            plotting_results(metric, metric_name, n_hidden)
