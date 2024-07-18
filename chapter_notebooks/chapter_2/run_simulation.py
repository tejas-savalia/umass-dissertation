if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import seaborn as sns
    from simulation_utils import *
    from sklearn.manifold import MDS
    from sklearn.metrics import euclidean_distances
    import scipy.stats as stat
    from scipy.optimize import minimize
    import itertools
    from multiprocessing import Pool

    p = Pool()

    
    graph = create_adjacency_matrix_for_modular_graph(15, 3, [5, 5, 5], [(0, 14), (4, 5), (9, 10)], [0, 4, 5, 9, 10, 14])
    #graph = create_adjacency_matrix_for_modular_graph(10, 2, [5, 5], [(4, 5)], [4, 5], 
    #                                                              [(0, 4), (4, 1), (0, 2), (0, 3), 
    #                                                              (9, 5), (5, 8), (6, 9), (7, 9)                                                             
     #                                                             ])

    
    a = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    b = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    walk_length = [1, 3, 6, 1000]
    # walk_length = [1000]
    iterations = np.arange(100)
    models = ['SR', 'TCM']
    # models = ['SR']
    graphs = [graph]
    
    node_entropy_params = itertools.product(iterations, a, b, walk_length, graphs, models)
    # node_distance_params = itertools.product(iterations, a, b, walk_length, graphs, models)


    # result = np.array(p.map(compute_node_entropies, node_entropy_params))
    # result = np.array(p.map(compute_node_distances, node_distance_params))
    # result = np.array(p.map(compute_node_distances_entropyboost, node_distance_params))
    
    # entropy_df = pd.DataFrame({'iteration': np.repeat(iterations, len(a)*len(b)*len(walk_length)*len(models)*graph.shape[0]),
    #                                 'param_a': np.tile(np.repeat(a, len(b)*len(walk_length)*len(models)*graph.shape[0]), len(iterations)),
    #                                  'param_b': np.tile(np.repeat(b, len(walk_length)*len(models)*graph.shape[0]), len(a)*len(iterations)),
    #                                  'walk_length': np.tile(np.repeat(walk_length, len(models)*graph.shape[0]), len(a)*len(b)*len(iterations)),
    #                                  'model': np.tile(np.repeat(models, graph.shape[0]), len(walk_length)*len(a)*len(b)*len(iterations)),
    #                                  'node': np.tile(np.arange(graph.shape[0]), len(models)*len(walk_length)*len(a)*len(b)*len(iterations)),
    #                                  'entropy': np.ravel(result)
    #                                  })
    # entropy_df.to_csv('simulation_results/3module_SRTCM_comp_simplertcm.csv')
    result = np.array(p.map(compute_node_surprisal, node_entropy_params))

    entropy_df = pd.DataFrame({'iteration': np.repeat(iterations, len(a)*len(b)*len(walk_length)*len(models)*3),
                                    'param_a': np.tile(np.repeat(a, len(b)*len(walk_length)*len(models)*3), len(iterations)),
                                     'param_b': np.tile(np.repeat(b, len(walk_length)*len(models)*3), len(a)*len(iterations)),
                                     'walk_length': np.tile(np.repeat(walk_length, len(models)*3), len(a)*len(b)*len(iterations)),
                                     'model': np.tile(np.repeat(models, 3), len(walk_length)*len(a)*len(b)*len(iterations)),
                                     'node comp': np.tile(['nbnb', 'nbb', 'bb'], len(models)*len(walk_length)*len(a)*len(b)*len(iterations)),
                                     'entropy': np.ravel(result)
                                     })
    entropy_df.to_csv('simulation_results/3module_SRTCM_comp_simplertcm_surprisal.csv')


    # entropy_df = pd.DataFrame({'iteration': np.repeat(iterations, len(a)*len(b)*len(walk_length)*len(models)*len(graphs)),
    #                             'param_a': np.tile(np.repeat(a, len(b)*len(walk_length)*len(models)*len(graphs)), len(iterations)),
    #                              'param_b': np.tile(np.repeat(b, len(walk_length)*len(models)*len(graphs)), len(a)*len(iterations)),
    #                              'walk_length': np.tile(np.repeat(walk_length, len(models)*len(graphs)), len(a)*len(b)*len(iterations)),
    #                              'model': np.tile(np.repeat(models, len(graphs)), len(walk_length)*len(a)*len(b)*len(iterations)),
    #                              'node': np.tile(np.arange(len(graphs)), len(models)*len(walk_length)*len(a)*len(b)*len(iterations)),
    #                              'entropy distance (2)': np.ravel(result[:, 0]),
    #                              'entropy distance (3)': np.ravel(result[:, 1]),
    #                              'entropy distance (1)': np.ravel(result[:, 2]),
    #                              })

    # entropy_df.to_csv('simulation_results/2moduledist_SR_distances_entropyboost.csv')


    # Memory Simulations
    # iterations = np.arange(100)
    # alpha = [0.01, 0.1, 0.25, 0.5, 0.75, 0.99]
    # gamma = [0.01, 0.1, 0.25, 0.5, 0.75, 0.99]
    # context_boost = ["Boost", "No Boost"]
    # node_type = ["Boundary", "Non Boundary"]
    # graphs = [graph]
    
    # params = itertools.product(iterations, alpha, gamma, graphs)
    # aucs = np.array(p.map(run_mem_boost, params))

    # auc_df = pd.DataFrame({'iteration': np.repeat(iterations, len(alpha)*len(gamma)*2*2),
    #               'alpha': np.tile(np.repeat(alpha, len(gamma)*2*2), len(iterations)),
    #               'gamma': np.tile(np.repeat(gamma, 2*2), len(iterations)*len(alpha)),
    #               'context boost': np.tile(np.repeat(['Boost', 'No Boost'], 2), len(iterations)*len(alpha)*len(gamma)),
    #               'Node Type': np.tile(['Boundary', 'Non Boundary'], len(iterations)*len(alpha)*len(gamma)*2),
    #               'aucs': np.ravel(aucs)
    #              })
    # auc_df.to_csv('simulation_results/3module_entropyboost.csv')

    # Categorization simulations
    # graph = create_adjacency_matrix_for_modular_graph(10, 2, [5, 5], [(4, 5)], [4, 5], 
    #                                                               [(0, 4), (4, 1), (2, 3),
    #                                                                (9, 5), (5, 8), (6, 7)
    #                                                               ])
    # iterations = np.arange(100)
    # num_diag_features = [1, 2, 3]
    # walk_length = [1, 1000]
    # alpha = [0.01, 0.1, 0.25, 0.5, 0.75, 0.99]
    # gamma = [0.01, 0.1, 0.25, 0.5, 0.75, 0.99]
    # graphs = [graph]
    # params = itertools.product(iterations, num_diag_features, walk_length, alpha, gamma, graphs)
    # f1_scores = np.array(p.map(categorization_sim, params))


    # cat_df = pd.DataFrame({'iteration': np.repeat(iterations, len(alpha)*len(gamma)*len(num_diag_features)*len(walk_length)),
    #                        'num_diag_features': np.tile(np.repeat(num_diag_features, len(alpha)*len(gamma)*len(walk_length)), len(iterations)),
    #                        'walk length': np.tile(np.repeat(walk_length, len(alpha)*len(gamma)), len(iterations)*len(num_diag_features)),
    #                        'alpha': np.tile(np.repeat(alpha, len(gamma)), len(iterations)*len(num_diag_features)*len(walk_length)),
    #                        'gamma': np.tile(gamma, len(iterations)*len(num_diag_features)*len(walk_length)*len(alpha)),
    #                        'f1 scores': np.ravel(f1_scores)
    #              })
    # cat_df.to_csv('simulation_results/2module_cat.csv')
