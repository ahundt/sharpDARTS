'''
This file contains graph operations for multichannelne. For graph generation see MultiChannelNet in cnn/model_search.py

'''
import matplotlib.pyplot as plt
import networkx as nx
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydotplus
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("Needs PyGraphviz or PyDotPlus to generate graph visualization")

import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser("Common Argument Parser")
parser.add_argument('--graph', type=str, default='../network_test.graph', help='location of the graph file')
parser.add_argument('--generate_path', type=str, default='greedy', help='specify one or more strategy\
  for generate path/sub-graph from the main graph, options include greedy, dag_longest, beam_search, flow_with_demand, iterative_cost_search')
parser.add_argument('--flow_demand', type=int, default=2, help='Set the demand limit for the iterative cost search or the demand for flow with demand')
parser.add_argument('--demand_step', type=int, default=1, help='Set the demand step size for the iterative cost search for best model')
parser.add_argument('--flow_cut', type=float, default=0, help='Set the threshold for the edges to be included in iterative cost search for best model.')
parser.add_argument('--beam_width', type=int, default=2, help='Set the beam width for beam search')

args = parser.parse_args()


def gen_greedy_path(G, strategy="top_down"):
    '''
    Generates a single path in a greedy way.
    # Arguments
        G: Graph
        strategy: top_down or bottom_up
    '''
    if strategy == "top_down":
        start_ = "Source"
        current_node = "Source"
        end_node = "Linear"
        new_G = G
    elif strategy == "bottom_up":
        start_ = "Linear"
        current_node = "Linear"
        end_node = "Source"
        new_G = G.reverse(copy=True)
    wt = 0
    node_list = []
    while current_node != end_node:
        neighbors = [n for n in new_G.neighbors(start_)]
        for nodes in neighbors:
            weight_ = new_G.get_edge_data(start_, nodes, "weight")
            # print(weight_)
            if len(weight_):
                weight_ = weight_["weight"]
            else:
                weight_ = 0
    #         print(weight_)
            if weight_ > wt:
                wt = weight_
                current_node = nodes
        node_list.append(current_node)
        # print("start",start_)
        # print(node)
        start_ = current_node
        wt = -1
    # print(node_list)
    if strategy == "bottom_up":
        node_list = node_list[::-1]
        node_list.append("Linear")
    return node_list


def generate_path(G, operations, directory):
    '''
    Generates optimal path/sub_graph using different strategies
    '''
    if 'greedy' in operations:
        bottom_up_greedy = gen_greedy_path(G, strategy="bottom_up")
        top_down_greedy = gen_greedy_path(G, strategy="top_down")

    if 'dag_longest' in operations:
        dag_optimal_path = nx.algorithms.dag.dag_longest_path(G)

    if 'beam_search' in operations:
        # TODO test and save output
        weighted_degree = dict(list(G.degree(weight='weight')))
        beam_edges = nx.algorithms.traversal.beamsearch.bfs_beam_edges(G, 'Source', weighted_degree.get, width=args.beam_width)

    if 'flow_with_demand' in operations:
        G.nodes["Source"]['demand'] = -args.flow_demand
        G.nodes["Linear"]['demand'] = args.flow_demand
        fl_cost, fl_dict = nx.capacity_scaling(G, demand='demand', weight='capacity', capacity='weight_int')
        new_g = nx.DiGraph()
        new_path = new_g.add_edges_from(min_cost_flow_edge)
        nx.write_pickle(new_path, os.path.join(directory_name, "capacity_scaled.graph"))

    elif 'iterative_cost_search' in operations:
        demand_cost_list = []
        for demand in range(1, args.flow_demand, args.demand_step):
            G.nodes["Source"]['demand'] = -args.flow_demand
            G.nodes["Linear"]['demand'] = args.flow_demand
            try:
                flow_cost, flow_dict = nx.network_simplex(cnn_model.G, weight='capacity', capacity='weight_int')
            except nx.exception.NetworkXUnfeasible:
                print('There is no flow satisfying the demand ', demand)
                continue
            min_cost_flow_edges = [(u, v) for u in flow_dict for v in flow_dict[u] if flow_dict[u][v] > args.flow_cut]
            if len(min_cost_flow_edges) != 0:
                demand_cost_list.append(demand, fl_cost)

        plt.xlabel('Demand')
        plt.ylabel('Cost')
        plt.scatter(*zip(*demand_cost_list))
        plt.savefig(os.path.join(directory_name, "iterative_cost_search_graph.png"))


def main():
    G = nx.read_pickle(args.graph)
    operations = args.generate_path.split(" ")
    date_str = datetime.now().strftime('_%Y-%m-%d_%H:%M:%S')
    directory_name = "./graph_operations"+date_str+'/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    generate_path(G, operations, directory)


if __name__ == '__main__':
    main()
