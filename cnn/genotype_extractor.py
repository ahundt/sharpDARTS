'''
This file includes code from the DARTS and sharpDARTS https://arxiv.org/abs/1903.09900 papers.
'''

import numpy as np
import genotypes


def parse_cell(weights, primitives, steps=4, skip_primitive='none'):
    """ Take a weight array and turn it into a list of pairs (primitive_string, node_index).
    """
    gene = []
    n = 2
    start = 0
    for add_node_index in range(steps):
        # Each step is a separate "add node" in the graph, so i is the integer index of the current node.
        # A better name for i might be add_node_index.
        end = start + n
        # Only look at the weights relevant to this node.
        # "Nodes" 0 and 1 will always be the output of the previous cells.
        #
        # All other nodes will be add nodes which need edges connecting back to the previous nodes:
        # add node 0 will need 2: rows 0, 1
        # add node 1 will need 3: rows 2, 3, 4
        # add node 2 will need 4: rows 5, 6, 7, 8
        # add node 3 will need 5: rows 9, 10, 11, 12, 13
        # ...and so on if there are more than 4 nodes.
        W = weights[start:end].copy()
        # print('add_node_index: ' + str(add_node_index) + ' start: ' + str(start) + ' end: ' + str(end) + ' W shape: ' + str(W.shape))
        # Each row in the weights is a separate edge, and each column are the possible primitives that edge might use.
        # The first "add node" can connect back to the two previous cells, which is why the edges are i + 2.
        # The sorted function orders lists from lowest to highest, so we use -max in the lambda function to sort from highest to lowest.
        # We currently say there will only be two edges connecting to each node, which is why there is [:2], to select the two highest score edges.
        # Each later nodes can connect back to the previous cells or an internal node, so the range(i+2) of possible connections increases.
        pre_edges = sorted(range(add_node_index + 2),
                           key=lambda x: -max(W[x][k] for k in range(len(W[x])) if skip_primitive is None or k != primitives.index(skip_primitive)))
        edges = pre_edges[:2]
        # print('edges: ' + str(edges))
        # We've now selected the two edges we will use for this node, so next let's select the layer primitives.
        # Each edge needs a particular primitive, so go through all the edges and compare all the possible primitives.
        for j in edges:
            k_best = None
            # note: This probably could be simpler via argmax...
            # Loop through all the columns to find the highest score primitive for the chosen edge, excluding none.
            for k in range(len(W[j])):
                if skip_primitive is None or k != primitives.index(skip_primitive):
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
            # Once the best primitive is chosen, create the new gene element, which is the
            # string for the name of the primitive, and the index of the previous node to connect to,
            gene.append((primitives[k_best], j))
        start = end
        n += 1
    # Return the full list of (node, primitive) pairs for this set of weights.
    return gene

def genotype_cell(alphas_normal, alphas_reduce, primitives, steps=4, multiplier=4, skip_primitive='none'):
    # skip_primitive = 'none' is a hack in original DARTS, which removes a no-op primitive.
    # skip_primitive = None means no hack is applied
    # note the printed weights from a call to Network::arch_weights() in model_search.py
    # are already post-softmax, so we don't need to apply softmax again
    # alphas_normal = torch.FloatTensor(genotypes.SHARPER_SCALAR_WEIGHTS.normal)
    # alphas_reduce = torch.FloatTensor(genotypes.SHARPER_SCALAR_WEIGHTS.reduce)
    # F.softmax(alphas_normal, dim=-1).data.cpu().numpy()
    # F.softmax(alphas_reduce, dim=-1).data.cpu().numpy()
    gene_normal = parse_cell(alphas_normal, primitives, steps, skip_primitive=skip_primitive)
    gene_reduce = parse_cell(alphas_reduce, primitives, steps, skip_primitive=skip_primitive)

    concat = range(2+steps-multiplier, steps+2)
    genotype = genotypes.Genotype(
        normal=gene_normal, normal_concat=concat,
        reduce=gene_reduce, reduce_concat=concat,
        layout='cell',
    )
    return genotype


def main():
    '''
    Parse raw weights with the hack that excludes the 'none' primitive, and without that hack.
    Then print out the final genotypes.
    '''
    # Set these variables to the ones you're using in this case from genotypes.py
    skip_primitive = None
    raw_weights_genotype = genotypes.SHARPER_SCALAR_WEIGHTS
    primitives = genotypes.SHARPER_PRIMITIVES
    # get the normal and reduce weights as a numpy array
    alphas_normal = np.array(raw_weights_genotype.normal)
    alphas_reduce = np.array(raw_weights_genotype.reduce)

    # for steps, see layers_in_cells in train_search.py
    steps = 4
    # for multiplier, see multiplier for Network class in model_search.py
    multiplier = 4
    # note the printed weights from a call to Network::arch_weights() in model_search.py
    # are already post-softmax, so we don't need to apply softmax again
    # alphas_normal = torch.FloatTensor(genotypes.SHARPER_SCALAR_WEIGHTS.normal)
    # alphas_reduce = torch.FloatTensor(genotypes.SHARPER_SCALAR_WEIGHTS.reduce)
    # F.softmax(alphas_normal, dim=-1).data.cpu().numpy()
    # F.softmax(alphas_reduce, dim=-1).data.cpu().numpy()

    # skip_primitive = 'none' is a hack in original DARTS, which removes a no-op primitive.
    # skip_primitive = None means no hack is applied
    print('#################')
    genotype = genotype_cell(alphas_normal, alphas_reduce, primitives, steps=4, multiplier=4, skip_primitive='none')
    print('SHARPER_SCALAR_genotype_skip_none = ' + str(genotype))
    genotype = genotype_cell(alphas_normal, alphas_reduce, primitives, steps=4, multiplier=4, skip_primitive=None)
    print('SHARPER_SCALAR_genotype_no_hack = ' + str(genotype))
    # Set these variables to the ones you're using in this case from genotypes.py
    skip_primitive = None
    raw_weights_genotype = genotypes.SHARPER_MAX_W_WEIGHTS
    primitives = genotypes.SHARPER_PRIMITIVES
    # get the normal and reduce weights as a numpy array
    alphas_normal = np.array(raw_weights_genotype.normal)
    alphas_reduce = np.array(raw_weights_genotype.reduce)

    # for steps, see layers_in_cells in train_search.py
    steps = 4
    # for multiplier, see multiplier for Network class in model_search.py
    multiplier = 4
    # note the printed weights from a call to Network::arch_weights() in model_search.py
    # are already post-softmax, so we don't need to apply softmax again
    # alphas_normal = torch.FloatTensor(genotypes.SHARPER_SCALAR_WEIGHTS.normal)
    # alphas_reduce = torch.FloatTensor(genotypes.SHARPER_SCALAR_WEIGHTS.reduce)
    # F.softmax(alphas_normal, dim=-1).data.cpu().numpy()
    # F.softmax(alphas_reduce, dim=-1).data.cpu().numpy()

    # skip_primitive = 'none' is a hack in original DARTS, which removes a no-op primitive.
    # skip_primitive = None means no hack is applied
    print('#################')
    genotype = genotype_cell(alphas_normal, alphas_reduce, primitives, steps=4, multiplier=4, skip_primitive='none')
    print('SHARPER_MAX_W_genotype_skip_none = ' + str(genotype))
    genotype = genotype_cell(alphas_normal, alphas_reduce, primitives, steps=4, multiplier=4, skip_primitive=None)
    print('SHARPER_MAX_W_genotype_no_hack = ' + str(genotype))

if __name__ == '__main__':
    main()
