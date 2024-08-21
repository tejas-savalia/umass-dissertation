import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, f1_score, euclidean_distances
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import jensenshannon
import itertools

def create_adjacency_matrix_for_modular_graph(num_nodes, num_modules, module_sizes, inter_module_edges, boundary_nodes,  edges_to_remove = None):
    """
    Creates an adjacency matrix for a graph with modular structure.
    
    Args:
    num_nodes: The total number of nodes in the graph.
    num_modules: The number of modules in the graph.
    module_sizes: A list of the sizes of each module.
    inter_module_edges: A list of edges between modules.
    
    Returns:
    An adjacency matrix for the graph.
    """

  # Create an empty adjacency matrix.
    adj_matrix = np.zeros((num_nodes, num_nodes))

  # Add edges within each module.
    for module_index in range(num_modules):
        module_start_index = sum(module_sizes[:module_index])
        module_end_index = module_start_index + module_sizes[module_index]
    
        for node_index in range(module_start_index, module_end_index):
            for other_node_index in range(module_start_index, module_end_index):
                if node_index != other_node_index:
                    adj_matrix[node_index, other_node_index] = 1
        
    for node_i in boundary_nodes:
        for node_j in boundary_nodes:
            adj_matrix[node_i][node_j] = 0
    
    if edges_to_remove is not None:
        for edge in edges_to_remove:
            node_index_1, node_index_2 = edge
            adj_matrix[node_index_1, node_index_2] = 0
            adj_matrix[node_index_2, node_index_1] = 0

    # Add edges between modules.
    for edge in inter_module_edges:
        node_index_1, node_index_2 = edge
        adj_matrix[node_index_1, node_index_2] = 1
        adj_matrix[node_index_2, node_index_1] = 1
    
    return adj_matrix



def plot_graph(graph, savefig_path = None, boundary_color = True):
    G = nx.Graph() 
    color_map = []
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i][j]:
                G.add_edge(i, j)
    if boundary_color:
        for node in G:
            if node in [0, 4, 5, 9, 10, 14]:
                color_map.append('C1')
            else:
                color_map.append('C0')
    else:
        color_map = ['C0']*len(G)
    nx.draw(G, node_color = color_map, with_labels = True)
    if savefig_path:
        plt.savefig(savefig_path, dpi = 300, transparent = True)
    plt.show()

def random_walk(graph, hop_step = 1000, path_length = 1000, start_state = None, p = None):
    #Random Walk
    if start_state == None:
        start_state = np.random.choice(range(graph.shape[0]))
        current_state = start_state
    else:
        start_state = start_state
        current_state = start_state
    path = np.zeros(path_length)

    for i in range(path_length):

        if (i+1)%hop_step == 0:
            start_state = np.random.choice(range(graph.shape[0]))
            current_state = start_state

        neighbour_states = np.where(graph[current_state])[0]
        # print(neighbour_states)
        if p == None:
            next_state = np.random.choice(neighbour_states)
        else:
            # print(graph[current_state][neighbour_states])
            next_state = np.random.choice(neighbour_states, p = graph[current_state][neighbour_states])
            
        path[i] = current_state
        current_state = next_state

    return path


def single_hamilton(G, start_point):
    F = [(G,[start_point])]
    n = G.number_of_nodes()
    while F:
        graph,path = F.pop()
        confs = []
        neighbors = (node for node in graph.neighbors(path[-1]) 
                     if node != path[-1]) #exclude self loops
        for neighbor in neighbors:
            conf_p = path[:]
            conf_p.append(neighbor)
            conf_g = nx.Graph(graph)
            conf_g.remove_node(path[-1])
            confs.append((conf_g,conf_p))
        for g,p in confs:
            if len(p)==n:
                return p
            else:
                F.append((g,p))
    return None
    
def hamiltonian_path(graph, path_length):
    G = nx.Graph() 
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i][j]:
                G.add_edge(i, j)
    complete_path = []
    while len(complete_path) < path_length:
        if len(complete_path) == 0:
            hamiltonian_path = single_hamilton(G, np.random.choice(15))
        else:
            hamiltonian_path = single_hamilton(G, complete_path[-1])
        complete_path.extend(hamiltonian_path)
    return complete_path

def find_single_euler_path(graph, startpoint):
    path = []
     
    # Loop will run until there is element in the
    # stack or current edge has some neighbour.
    cur = startpoint
    graph_copy = graph.copy()
    stack = []
    while (len(stack) > 0 or sum(graph_copy[cur])!= 0):
        
        # If current node has not any neighbour
        # add it to path and pop stack set new 
        # current to the popped element
        if (sum(graph_copy[cur]) == 0):
            path.append(cur)
            cur = stack[-1]
            del stack[-1]

        # If the current vertex has at least one
        # neighbour add the current vertex to stack,
        # remove the edge between them and set the
        # current to its neighbour.
        else:
            nonzero_vertices = np.nonzero(graph_copy[cur])[0]
            np.random.shuffle(nonzero_vertices)
            stack.append(cur)
            graph_copy[cur][nonzero_vertices[0]] = 0
            graph_copy[nonzero_vertices[0]][cur] = 0
            cur = nonzero_vertices[0]

    return path

def find_euler(graph, walk_length=1000):
    path = []
    while len(path) < walk_length:
        if len(path) == 0:
            start_point = np.random.choice(graph.shape[0])
        else:
            start_point = path[-1]
        path.extend(find_single_euler_path(graph, start_point))
    return path[:walk_length]

def run_SR(path, graph, alpha = 0.1, gamma = 0.05, plot = True, snapshot_step = 1000):
    SR = np.random.uniform(0, 1, size=graph.shape)
    num_nodes = graph.shape[0]
    start_state = np.random.choice(np.arange(num_nodes))
    current_state = start_state    

    snapshot_SR = np.zeros((int(len(path)/snapshot_step), graph.shape[0], graph.shape[1]))
    for i, observed_state in enumerate(path):

        expected_probs = SR[current_state]
        one_hot_obs = np.zeros(num_nodes)
        one_hot_obs[int(observed_state)] = 1

        SR_delta = one_hot_obs + gamma*SR[int(observed_state), :] - expected_probs

        SR[current_state, :] = SR[current_state, :] + alpha*SR_delta
        SR[current_state, :] = SR[current_state, :]/sum(SR[current_state])
        current_state = int(observed_state)

        if plot and i%snapshot_step == 0:
            snapshot_SR[int(i/snapshot_step)] = SR
    return SR, snapshot_SR


# def run_tcm(path, graph, alpha=0.1, beta=0.1, plot = True):
#     num_nodes = graph.shape[0]
#     items = np.identity(num_nodes)
#     context = np.random.uniform(0, 1, (num_nodes, num_nodes))
#     presented_items = items[path]
#     m_fc = np.random.uniform(0, 1, (num_nodes, num_nodes))
    
#     for i in range(1, len(path)):

#         c_IN = np.dot(m_fc, presented_items[i])
#         c_IN = c_IN/np.linalg.norm(c_IN)
#         rho = np.sqrt(1+(beta**2)*((np.dot(context[path[i-1]] , c_IN)**2)-1)) - beta*np.dot(context[path[i-1]],c_IN);
        
#         context[path[i]] = rho*context[path[i-1]] + beta*c_IN
        
#         m_fc[path[i]] = m_fc[path[i]] + alpha*np.dot(presented_items[path[i]], context[path[i-1]])
        
        
#     # for i in np.arange(m_fc.shape[0]):
#     #     m_fc[i] = m_fc[i]/np.linalg.norm(m_fc[i])
#     if plot:
#         g = sns.heatmap(m_fc[i])

#     return context

def run_tcm(path, graph, alpha=0.1, beta=0.1, plot = True):
    num_nodes = graph.shape[0]
    items = np.identity(num_nodes)
    # context = np.random.uniform(0, 1, (num_nodes, num_nodes))
    presented_items = items[path]
    # m_fc = np.random.uniform(0, 1, (num_nodes, num_nodes))
    M = np.random.uniform(0, 1, (num_nodes, num_nodes))
    for i in range(num_nodes):
        M[i] = M[i]/np.sum(M[i])
    context = np.random.uniform(0, 1, num_nodes)
    
    for i in range(1, len(path)):
        
        context = beta*context + presented_items[i-1]
        context = context/np.sum(context)
        M[path[i-1]][path[i]] = M[path[i-1]][path[i]] + alpha*context[path[i-1]]
        M[path[i-1]] = M[path[i-1]]/np.sum(M[path[i-1]])
        
        # c_IN = np.dot(m_fc, presented_items[i])
        # c_IN = c_IN/np.linalg.norm(c_IN)
        # rho = np.sqrt(1+(beta**2)*((np.dot(context[path[i-1]] , c_IN)**2)-1)) - beta*np.dot(context[path[i-1]],c_IN);
        
        # context[path[i]] = rho*context[path[i-1]] + beta*c_IN
        
        # m_fc[path[i]] = m_fc[path[i]] + alpha*np.dot(presented_items[path[i]], context[path[i-1]])
        
        
    # for i in np.arange(m_fc.shape[0]):
    #     m_fc[i] = m_fc[i]/np.linalg.norm(m_fc[i])
    if plot:
        # g = sns.heatmap(m_fc[i])
        g = sns.heatmap(M)

    return M

def compute_node_entropies(params):
    #params is three parameters. a (alpha_SR, alpha_tcm), b(gamma, beta), hop_step
    if params[0]%10 == 0:
        print("iteration: ", params[0])
    a = params[1]
    b = params[2]
    graph = params[4]
    model = params[5]
    node_entropy = np.zeros(graph.shape[0])
    if model == 'SR':
        context_matrix = run_SR(random_walk(graph, hop_step=params[3]).astype(int), graph, a, b, plot=False)[0]
    else:
        context_matrix = run_tcm(random_walk(graph, hop_step=params[3]).astype(int), graph, a, b, plot=False)

            
    # graph_entropy = -np.sum(SR*np.log(SR))
    for node in range(graph.shape[0]):
        node_entropy[node] = -np.sum(context_matrix[node]*np.log(context_matrix[node]))

    return node_entropy

def compute_node_jsdist(params):
    #params is three parameters. a (alpha_SR, alpha_tcm), b(gamma, beta), hop_step
    if params[0]%10 == 0:
        print("iteration: ", params[0])
    a = params[1]
    b = params[2]
    graph = params[4]
    model = params[5]
    node_entropy = np.zeros(graph.shape[0])
    if model == 'SR':
        context_matrix = run_SR(random_walk(graph, hop_step=params[3]), graph, a, b, plot=False)
    else:
        context_matrix = run_tcm(random_walk(graph, hop_step=params[3]).astype(int), graph, a, b, plot=False)

    node_jsdistances = np.array([[jensenshannon(i, j) for i in context_matrix] for j in context_matrix]).reshape(15, 15)            
    # graph_entropy = -np.sum(SR*np.log(SR))
    # for node in range(graph.shape[0]):
    #     node_entropy[node] = -np.sum(context_matrix[node]*np.log(context_matrix[node]))
    nonb_nonb_idx = [i for i in itertools.combinations([1, 2, 3], 2)] + [i for i in itertools.combinations([6, 7, 8], 2)] + [i for i in itertools.combinations([11, 12, 13], 2)]
    nonb_b_idx = [i for i in itertools.product([0, 4], [1, 2, 3])] + [i for i in itertools.product([5, 9], [6, 7, 8])] + [i for i in itertools.product([10, 14], [11, 12, 13])]
    
    nonb_nonb_js = np.mean([node_jsdistances[i] for i in nonb_nonb_idx])
    nonb_b_js = np.mean([node_jsdistances[i] for i in nonb_b_idx])
    b_b_js = np.mean([node_jsdistances[i] for i in [(0, 14), (4, 5), (9, 10)]])

    return np.array([nonb_nonb_js, nonb_b_js, b_b_js])

def compute_node_surprisal(params):
    #params is three parameters. a (alpha_SR, alpha_tcm), b(gamma, beta), hop_step
    if params[0]%10 == 0:
        print("iteration: ", params[0])
    a = params[1]
    b = params[2]
    graph = params[4]
    model = params[5]
    node_entropy = np.zeros(graph.shape[0])
    if model == 'SR':
        context_matrix, c = run_SR(random_walk(graph, hop_step=params[3]), graph, a, b, plot=False)
    else:
        context_matrix = run_tcm(random_walk(graph, hop_step=params[3]).astype(int), graph, a, b, plot=False)

    # node_surprisal = np.array([[jensenshannon(i, j) for i in context_matrix] for j in context_matrix]).reshape(15, 15)            
    # graph_entropy = -np.sum(SR*np.log(SR))
    # for node in range(graph.shape[0]):
    #     node_entropy[node] = -np.sum(context_matrix[node]*np.log(context_matrix[node]))
    nonb_nonb_idx = [i for i in itertools.combinations([1, 2, 3], 2)] + [i for i in itertools.combinations([6, 7, 8], 2)] + [i for i in itertools.combinations([11, 12, 13], 2)]
    nonb_b_idx = [i for i in itertools.product([0, 4], [1, 2, 3])] + [i for i in itertools.product([5, 9], [6, 7, 8])] + [i for i in itertools.product([10, 14], [11, 12, 13])]
    # print(context_matrix)
    nonb_nonb_js = np.mean([-np.log(context_matrix[i[0], i[1]]) for i in nonb_nonb_idx])
    nonb_b_js = np.mean([-np.log(context_matrix[i[0], i[1]]) for i in nonb_b_idx])
    b_b_js = np.mean([-np.log(context_matrix[i[0], i[1]]) for i in [(0, 14), (4, 5), (9, 10)]])

    return np.array([nonb_nonb_js, nonb_b_js, b_b_js])


def compute_node_distances(params):
    iteration = params[0]
    graph = params[4]
    alpha = params[1]
    gamma = params[2]
    if iteration%10 == 0:
        print("iteration: ", iteration)
    
    across_nodes_to_compare = pd.DataFrame({ "node a": [7, 6, 2, 3, 4, 5, 5, 5, 4, 4],
                        "node b": [4, 4, 5, 5, 8, 1, 4, 4, 5, 5],
                        "distance": [2, 2, 2, 2, 3, 3, 1, 1, 1, 1],
                       })
    within_nodes_to_compare = pd.DataFrame({ "node a": [7, 6, 2, 3, 4, 5, 5, 5, 4, 4],
                        "node b": [9, 9, 0, 0, 0, 9, 6, 7, 2, 3],
                       })
    
    nodes_to_compare = across_nodes_to_compare.merge(within_nodes_to_compare, on = ["node a"])
    
    walk = random_walk(graph).astype(int)
    SR = run_SR(walk, graph=graph, alpha=alpha, gamma=gamma, plot = False)
    node_entropy = np.zeros(graph.shape[0])
    for node in range(node_entropy.shape[0]):
        node_entropy[node] = -np.sum(SR[node]*np.log(SR[node]))
    
    node_distances = [SR[nodes_to_compare.loc[i, "node a"], nodes_to_compare.loc[i, "node b_x"]] - 
                             SR[nodes_to_compare.loc[i, "node a"], nodes_to_compare.loc[i, "node b_y"]] for i in range(len(nodes_to_compare))]
    
    return np.mean(node_distances[:4]), np.mean(node_distances[4:8]), np.mean(node_distances[8:])   

def compute_node_distances_entropyboost(params):
    iteration = params[0]
    graph = params[4]
    alpha = params[1]
    gamma = params[2]
    if iteration%10 == 0:
        print("iteration: ", iteration)
    
    across_nodes_to_compare = pd.DataFrame({ "node a": [7, 6, 2, 3, 4, 5, 5, 5, 4, 4],
                        "node b": [4, 4, 5, 5, 8, 1, 4, 4, 5, 5],
                        "distance": [2, 2, 2, 2, 3, 3, 1, 1, 1, 1],
                       })
    within_nodes_to_compare = pd.DataFrame({ "node a": [7, 6, 2, 3, 4, 5, 5, 5, 4, 4],
                        "node b": [9, 9, 0, 0, 0, 9, 6, 7, 2, 3],
                       })
    
    nodes_to_compare = across_nodes_to_compare.merge(within_nodes_to_compare, on = ["node a"])
    
    walk = random_walk(graph).astype(int)
    SR = run_SR(walk, graph=graph, alpha=alpha, gamma=gamma, plot = False)
    node_entropy = np.zeros(graph.shape[0])
    for node in range(node_entropy.shape[0]):
        node_entropy[node] = -np.sum(SR[node]*np.log(SR[node]))
    boundary_entropy = np.mean(node_entropy[[4, 5]])
    non_boundary_entropy = np.mean(node_entropy[[0, 1, 2, 3, 6, 7, 8, 9]])    
    
    node_distances = [SR[nodes_to_compare.loc[i, "node a"], nodes_to_compare.loc[i, "node b_x"]]*boundary_entropy - 
                             SR[nodes_to_compare.loc[i, "node a"], nodes_to_compare.loc[i, "node b_y"]]*non_boundary_entropy for i in range(len(nodes_to_compare))]
    
    return np.mean(node_distances[:4]), np.mean(node_distances[4:8]), np.mean(node_distances[8:])   



## Recognition Memory Functions

def get_stimset(means, cov_scale = 0.1, n = 1000):
    
    cov = np.identity(means.shape[0]) * cov_scale
    return np.random.multivariate_normal(means, cov, size=n)
    


def get_encoded(store_prob_alpha, store_prob_beta, encode_noise, stim_stream, obj_cat_feature_matrix, context_boost = False, node_entropy = None):
    encoder = np.zeros((len(stim_stream), obj_cat_feature_matrix.shape[1]))
    # forgetting_rate = 0.05
    # encoding_rate = 0.9
    if not context_boost:
        for i, stim in enumerate(stim_stream):
            for feature in range(obj_cat_feature_matrix.shape[1]):
                store_prob = np.random.beta(store_prob_alpha, store_prob_beta)
                acc_prob = np.random.beta(store_prob_alpha, store_prob_beta)

                if np.random.binomial(1, store_prob):
                    if np.random.binomial(1, acc_prob):
                        encoder[i][feature] = obj_cat_feature_matrix[stim][feature] 
                    else:
                        encoder[i][feature] = obj_cat_feature_matrix[stim][feature] + np.random.normal(obj_cat_feature_matrix[stim][feature], encode_noise)
                else:
                    encoder[i][feature] = 0
    else:
        for i, stim in enumerate(stim_stream):
            for feature in range(obj_cat_feature_matrix.shape[1]):
                store_prob = np.random.beta(store_prob_alpha*node_entropy[stim], store_prob_beta/node_entropy[stim])
                acc_prob = np.random.beta(store_prob_alpha*node_entropy[stim], store_prob_beta/node_entropy[stim])

                if np.random.binomial(1, store_prob):
                    if np.random.binomial(1, acc_prob):
                        encoder[i][feature] = obj_cat_feature_matrix[stim][feature] 
                    else:
                        encoder[i][feature] = obj_cat_feature_matrix[stim][feature] + np.random.normal(obj_cat_feature_matrix[stim][feature], encode_noise)
                else:
                    encoder[i][feature] = 0
        

    return encoder
    
def recognition(test_item, encoder, c = 50, match_sd = .05, weights = np.array([.5, .5]), criterion = 1):

    memory_match_trace = 0
    
    for e in encoder:
        dist = c*(np.sqrt(weights[0]*(test_item[0] - e[0])**2 + weights[1]*(test_item[1] - e[1])**2))
        memory_match_trace += np.exp(-dist) #+ np.random.normal(0, match_sd)
    # print(memory_match_trace/len(encoder))
    if memory_match_trace/len(encoder) > criterion:
        return True
    else:
        return False        

def get_roc(stim_stream, new_test_objects, study_objects, cb = False, node_entropy = None):
    c = 1
    encoder = np.array(get_encoded(1, 1, 0.01, stim_stream, study_objects, context_boost=cb, node_entropy=node_entropy))
    old_boundary_objects = study_objects[[0, 4, 5, 9, 10, 14]]
    old_nonboundary_objects = study_objects[[1, 2, 3, 6, 7, 8, 11, 12, 13]]
    

    criteria = np.linspace(0, 1, 100)
    fa_rates = np.zeros(100)
    
    hit_rates_boundary = np.zeros(100)
    hit_rates_nonboundary = np.zeros(100)

    for i, criterion in enumerate(criteria):
        
        fa_rates[i] = np.mean([recognition(new_test_item, encoder, criterion=criterion, c = c) for new_test_item in new_test_objects])
        hit_rates_boundary[i] = np.mean([recognition(old_test_item, encoder, criterion=criterion, c = c) for old_test_item in old_boundary_objects])
        hit_rates_nonboundary[i] = np.mean([recognition(old_test_item, encoder, criterion=criterion, c = c) for old_test_item in old_boundary_objects])
            
    return np.array([fa_rates, hit_rates_boundary, hit_rates_nonboundary])

def run_mem_boost(params):
    iteration = params[0]
    if iteration%10 == 0:
        print('iteration', iteration)
    alpha = params[1]
    gamma = params[2]
    graph = params[3]
    d_ = 0
    signal_cov_scale = 0.1
    stim_set = get_stimset(np.array([0, 0]), cov_scale = signal_cov_scale, n = 15)
    test_items = get_stimset(np.array([signal_cov_scale*d_, signal_cov_scale*d_]), cov_scale = signal_cov_scale, n = 15)
    walk = random_walk(graph).astype(int)
    SR = run_SR(alpha=0.1, gamma=0.1, path=walk, graph=graph, plot=False)
    node_entropies = np.array([-np.sum(SR[i]*np.log(SR[i])) for i in np.arange(SR.shape[0])])
    
    cb_roc = get_roc(walk, test_items, stim_set, cb=True, node_entropy=node_entropies)
    noncb_roc = get_roc(walk, test_items, stim_set, cb=False)
    
    return auc(cb_roc[0, :], cb_roc[1, :]), auc(cb_roc[0, :], cb_roc[2, :]), auc(noncb_roc[0, :], noncb_roc[1, :]), auc(noncb_roc[0, :], noncb_roc[2, :])

def categorization_sim(params):
    iteration = params[0]
    num_diag_features = params[1]
    walk_length = params[2]
    alpha = params[3]
    gamma = params[4]
    graph = params[5]
    if iteration%20 == 0:
        print("iteration", iteration)
    
    walk = random_walk(graph, hop_step=walk_length)
    SR = run_SR(alpha=alpha, gamma=gamma, path=walk, graph=graph, plot = False)
    SR_distances = euclidean_distances(SR)
    
    stims = np.zeros((10, 9))
    stims[:5, :num_diag_features] = 1
    stims[5:, num_diag_features:num_diag_features*2] = 1
    stims[:, num_diag_features*2:] = np.random.binomial(1, 0.5, 10*(9-num_diag_features*2)).reshape(10, 9-num_diag_features*2)
    
    for i in range(stims.shape[0]):
        stims[i] = stims[i]/np.sum(stims[i])
    
    stim_context_ass = np.dot(1-SR_distances, stims)
    gm = GaussianMixture(n_components=2, random_state=0, init_params='k-means++').fit(stim_context_ass)        
    y_ones = np.concatenate([np.ones(5), np.zeros(5)])
    y_zeros = np.concatenate([np.zeros(5), np.ones(5)])
    return np.max([f1_score(gm.predict(stim_context_ass), y_ones), f1_score(gm.predict(stim_context_ass), y_zeros)])

