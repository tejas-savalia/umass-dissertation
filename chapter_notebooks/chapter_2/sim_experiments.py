if __name__ == '__main__':
    import numpy as np
    import bambi as bmb
    import pymc as pm
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from simulation_utils import *
    from power_sim_utils import *
    import arviz as az
    import multiprocessing, itertools
    
    p = multiprocessing.Pool()
    
    #modular_graph = create_adjacency_matrix_for_modular_graph(15, 3, [5, 5, 5], [(0, 14), (4, 5), (9, 10)], [0, 4, 5, 9, 10, 14])
    sample_sizes = [20, 40, 60, 80, 100, 120, 140, 160]
    num_experiments = 500
    noise = 0.1
    # Uncomment to recreate experiments with rt differences
    #rt_diff_comp_df_temp = []
    #for s in sample_sizes:
    #    params = itertools.product([s], np.arange(num_experiments), [noise])
    #    rt_diff = np.array(p.map(run_first_experiments, params))
    #    walk_lengths = np.repeat([1, 3, 6, 1400], s/4)

    
    #    rt_diff_comp_df_temp.append(pd.DataFrame({'Sample Size': np.repeat(s, num_experiments*s*2),
     #                                        'Model': np.tile(np.repeat(['SR', 'TCM'], s), num_experiments),
      #                                       'walk_lengths': np.tile(walk_lengths, 2*num_experiments),
       #                                       'rt_diff': np.ravel(rt_diff)
        #                                    }))
    #rt_diff_comp_df = pd.concat(rt_diff_comp_df_temp).reset_index(drop=True)
    #rt_diff_comp_df.to_csv('simulation_results/exp1_rtdiff_power_exp.csv')

    #Uncomment to fit models on simulated exp1 data data
    BF = np.array(p.map(compute_BF, sample_sizes))
    exp1_BF = pd.DataFrame({'Sample Size': np.repeat(sample_sizes, num_experiments),
                            'Experiment': np.tile(np.arange(num_experiments), len(sample_sizes)),
                            'BF': np.ravel(BF)                           
                           })
    exp1_BF.to_csv('simulation_results/exp1_BF.csv')
