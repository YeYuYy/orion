import math
import warnings
import numpy as np
import networkx as nx
from copy import deepcopy
from orion.core.network_dag import NetworkDAG
from orion.nn.linear import LinearTransform
from orion.nn.activation import Chebyshev, Quad
from orion.nn.operations import Cat, Mult, Add, LogSum

class TTSPSolver:
    """
    Algorithm to find the approximately maximum TTSP subgraph from a given DAG
    for subsequent bootstrapping placement. It is inclined to preserve edges
    that with larger ciphertext number and "uncertainty" of bootstrapping.
    """
    def __init__(self, raw_dag: NetworkDAG, slots: int, l_eff: int):
        self.slots = slots
        self.l_eff = l_eff
        self.raw_dag = raw_dag
        self.reachability_matrix = np.ones((raw_dag.number_of_nodes(), raw_dag.number_of_nodes()))
        self.sorted_nodes = list(nx.topological_sort(raw_dag))
        self.node_to_idx = {node: i for i, node in enumerate(self.sorted_nodes)}
        self._process_nodes()
        self._assign_weight()

        self.ttsp_dag = NetworkDAG(None)
        source = self.sorted_nodes[0]
        sink = self.sorted_nodes[-1]
        self.ttsp_dag.add_node(source, **raw_dag.nodes[source])
        self.ttsp_dag.add_node(sink, **raw_dag.nodes[sink])

    def solve_max_ttsp_subgraph(self):
        """
        Main function to solve the maximum TTSP subgraph.
        Greedily add "ear" shaped paths with largest weights until no more edges can be added
        without violating the serial-parallel constraint.
        Returns:
            NetworkDAG: The TTSP subgraph.
        """
        while True:
            update_flag = False
            ear_paths = self._sort_ear_path()

            for path in ear_paths:
                u, v, p = path['u'], path['v'], path['path']
                u_idx = self.node_to_idx[u]
                v_idx = self.node_to_idx[v]
                if self.reachability_matrix[u_idx, v_idx] == 1:
                    is_sp = False
                    if self.ttsp_dag.has_edge(u, v):
                        is_sp = not (len(p) == 1) # Filter out edges that are already in the TTSP dag
                    else:
                        self.ttsp_dag.add_edge(u, v)
                        is_sp = self.ttsp_dag.is_serial_parallel()
                        self.ttsp_dag.remove_edge(u, v)
                    if is_sp:
                        update_flag = True
                        last_node = u
                        for node in p[:-1]:
                            self.ttsp_dag.add_node(node, **self.raw_dag.nodes[node])
                            self.ttsp_dag.add_edge(last_node, node, **self.raw_dag.edges[last_node, node])
                            last_node = node
                        self.ttsp_dag.add_edge(last_node, v, **self.raw_dag.edges[last_node, v])
                        print(f"├── Added {len(p)+1} edges from {u} to {v} into TTSP subgraph.")
                        break
                    else:
                        self.reachability_matrix[u_idx, v_idx] = 0
            if not update_flag:
                break

        return self.ttsp_dag, self.raw_dag
    
    def _process_nodes(self):
        for node in list(self.sorted_nodes):
            attrs = self.raw_dag.nodes[node]
            if attrs.get('op') == 'call_module':
                module = attrs.get('module')
                in_shape = module.fhe_input_shape[0] if isinstance(module.fhe_input_shape[0], tuple) else module.fhe_input_shape
                out_shape = module.fhe_output_shape[0] if isinstance(module.fhe_output_shape[0], tuple) else module.fhe_output_shape
                attrs['in_ct_num'] = math.ceil(math.prod(in_shape) / self.slots)
                attrs['out_ct_num'] = math.ceil(math.prod(out_shape) / self.slots)
                if isinstance(module, (LinearTransform, Mult, Quad)):
                    attrs['depth'] = 1
                elif isinstance(module, Chebyshev):
                    attrs['depth'] = module.depth
                elif isinstance(module, (Cat, Add, LogSum)):
                    attrs['depth'] = 0
                else:
                    attrs['depth'] = 0
                    warnings.warn(f"Depth for module type {type(module)} is not defined, set to 0 by default.")

            # call_function always have one predecessor even before insertion of fork/join. 
            # It does not change the number of ciphertexts. Thus, we can safely copy the shape from its predecessor.
            elif attrs.get('op') == 'call_function':
                pred = self.raw_dag.nodes[next(self.raw_dag.predecessors(node))]
                attrs['in_ct_num'] = pred['out_ct_num']
                attrs['out_ct_num'] = attrs['in_ct_num']
                if 'mul' in attrs.get('label'):
                    attrs['depth'] = 1
                elif 'add' in attrs.get('label') or 'sub' in attrs.get('label'):
                    attrs['depth'] = 0
                else:
                    attrs['depth'] = 0
                    warnings.warn(f"Depth for function {attrs.get('label')} is not defined, set to 0 by default.")

            elif attrs.get('op') == 'fork' or attrs.get('op') == 'join' or attrs.get('op') == 'placeholder':
                attrs['in_ct_num'], attrs['out_ct_num'], attrs['depth'] = 0, 0, 0
            else:
                raise TypeError(f"Node operation {attrs.get('op')} is not supported.")

    def _assign_weight(self):
        for u, v, attrs in self.raw_dag.edges(data=True):
            u_attrs = self.raw_dag.nodes[u]
            v_attrs = self.raw_dag.nodes[v]

            if u_attrs['op'] == 'placeholder':
                weight = 0
            elif u_attrs['op'] in ['call_function', 'call_module'] and v_attrs['op'] in ['call_function', 'call_module']:
                weight = u_attrs['out_ct_num'] * max(min(u_attrs['depth'] + v_attrs['depth'],\
                                                          self.l_eff - u_attrs['depth'] - v_attrs['depth']), 0)
            elif u_attrs['op'] == 'fork' or u_attrs['op'] == 'join':
                weight = 0
            elif v_attrs['op'] == 'fork' or v_attrs['op'] == 'join':
                weight = u_attrs['out_ct_num'] * u_attrs['depth']
            else:
                raise TypeError(f"Edge from {u_attrs['op']} to {v_attrs['op']} is not recognized.")
            attrs['weight'] = weight

    def _sort_ear_path(self):
        """
        Compute the "ear" shaped paths with largest weights in raw_dag between any two nodes
        in ttsp_dag. An "ear" shaped path is defined as a path that starts and ends at nodes
        in ttsp_dag, with no intermediate nodes belonging to ttsp_dag.
        
        Returns:
            list of dict: [{'u': start, 'v': end, 'weight': val, 'path': [...]}, ...]
            Sorted in descending order of weight.
        """
        ttsp_nodes = set(self.ttsp_dag.nodes())
        node_to_idx = {node: i for i, node in enumerate(self.sorted_nodes)}
        all_paths = []

        for source in ttsp_nodes:
            dist = {node: -math.inf for node in self.raw_dag.nodes()}
            parent = {node: None for node in self.raw_dag.nodes()}
            dist[source] = 0
            start_idx = node_to_idx[source]
            
            for i in range(start_idx, len(self.sorted_nodes)):
                curr_node = self.sorted_nodes[i]
                if dist[curr_node] == -math.inf:
                    continue
                
                # Store the path from source to curr_node.
                # We do not propagate further from this ttsp node, so that paths are "ear" shaped.
                if curr_node in ttsp_nodes and curr_node != source:
                    path_list = self._reconstruct_path(parent, source, curr_node)
                    all_paths.append({
                        'u': source,
                        'v': curr_node,
                        'weight': dist[curr_node],
                        'path': path_list
                    })
                    continue 

                # Standard DP propagation
                for neighbor in self.raw_dag.successors(curr_node):
                    weight = self.raw_dag[curr_node][neighbor].get('weight', 1)
                    if dist[curr_node] + weight > dist[neighbor]:
                        dist[neighbor] = dist[curr_node] + weight
                        parent[neighbor] = curr_node

        all_paths.sort(key=lambda x: x['weight'], reverse=True)        
        return all_paths

    def _reconstruct_path(self, parent_map, source, target):
        path = []
        curr = target
        while curr is not None:
            if curr == source:
                break
            path.append(curr)
            curr = parent_map[curr]
        return path[::-1]
