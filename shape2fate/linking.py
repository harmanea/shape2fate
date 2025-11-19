import itertools
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from scipy import spatial, optimize


class Node:
    def __init__(self, name: str):
        self.name = name

        self.in_edges: list['Edge'] = []
        self.out_edges: list['Edge'] = []

    def __str__(self):
        return f'{[e.u for e in self.in_edges]} -> {self.name} -> {[e.v for e in self.out_edges]}'

    def __repr__(self):
        return self.name

    def create_flow_constraint(self, solver: pywraplp.Solver):
        solver.Add(sum([edge.flow for edge in self.in_edges]) == sum([edge.flow for edge in self.out_edges]))

    def create_assignment_constraint(self, solver: pywraplp.Solver, value: int = 1):
        solver.Add(sum([edge.flow for edge in self.in_edges]) == value)

    def create_source_constraint(self, solver: pywraplp.Solver, value: int):
        solver.Add(sum([edge.flow for edge in self.out_edges]) == value)


class DetectionNode(Node):
    def __init__(self, frame: int, index: int):
        super().__init__(f'o{index},{frame}')
        self.frame = frame
        self.index = index


class MergeNode(Node):
    def __init__(self, into_node: DetectionNode):
        super().__init__(f'm{into_node.index},{into_node.frame - 1}')
        self.into_node = into_node

    # noinspection DuplicatedCode
    def create_merge_constraint(self, solver: pywraplp.Solver):
        assert len(self.out_edges) == 2, 'merge nodes must have exactly two outgoing edges'

        edge_to_death_node = [edge for edge in self.out_edges if isinstance(edge.v, DeathNode)][0]
        edge_to_detection_node = [edge for edge in self.out_edges if edge.v == self.into_node][0]

        solver.Add(edge_to_detection_node.flow - edge_to_death_node.flow <= 0)
        for edge in self.in_edges:
            solver.Add(edge.flow - edge_to_detection_node.flow <= 0)


class SkipNode(Node):
    def __init__(self, from_node: DetectionNode, skipped_frames: int):
        super().__init__(f'x{from_node.index},{from_node.frame + skipped_frames - 1}')
        self.from_node = from_node
        self.skipped_frames = skipped_frames


class SplitNode(Node):
    def __init__(self, from_node: DetectionNode | SkipNode):
        if isinstance(from_node, DetectionNode):
            index, frame = from_node.index, from_node.frame + 1
        elif isinstance(from_node, SkipNode):
            index, frame = from_node.from_node.index, from_node.from_node.frame + from_node.skipped_frames
        else:
            raise TypeError(f'Unsupported node type: {type(from_node)}')

        super().__init__(f's{index},{frame}')

        self.from_node = from_node
        self.frame = frame
        self.index = index

    # noinspection DuplicatedCode
    def create_split_constraint(self, solver: pywraplp.Solver):
        assert len(self.in_edges) == 2, 'split nodes must have exactly two incoming edges'

        edge_from_birth_node = [edge for edge in self.in_edges if isinstance(edge.u, BirthNode)][0]
        edge_from_detection_node = [edge for edge in self.in_edges if edge.u == self.from_node][0]

        solver.Add(edge_from_detection_node.flow - edge_from_birth_node.flow <= 0)
        for edge in self.out_edges:
            solver.Add(edge.flow - edge_from_detection_node.flow <= 0)


class BirthNode(Node):
    def __init__(self, frame: int):
        super().__init__(f'b{frame}')
        self.frame = frame


class DeathNode(Node):
    def __init__(self, frame: int):
        super().__init__(f'd{frame}')
        self.frame = frame


class Edge:
    def __init__(self, u: Node, v: Node, unit_cost: float, flow: pywraplp.Solver.IntVar):
        self.u = u
        self.v = v
        self.unit_cost = unit_cost
        self.flow = flow

    def __str__(self):
        return f'{self.u.name} -> {self.v.name}'


class Tracklet:
    def __init__(self, index: int):
        self.index = index

        self.predecessors = set()
        self.successors = set()
        self.detections = {}

    def t_start(self) -> int:
        return min(self.detections.keys())

    def t_end(self) -> int:
        return max(self.detections.keys())

    def has_successors(self) -> bool:
        return len(self.successors) > 0

    def has_predecessors(self) -> bool:
        return len(self.predecessors) > 0

    def __str__(self):
        return f'#{self.index} : {list(self.predecessors)} -> {self.detections} -> {list(self.successors)}'

    def __repr__(self):
        return str(self.index)

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        if not isinstance(other, Tracklet):
            return NotImplemented
        return self.index == other.index

    def __lt__(self, other):
        if not isinstance(other, Tracklet):
            return NotImplemented
        return self.index < other.index

    def __len__(self):
        return len(self.detections)


class LinkingGraph:
    def __init__(self, detections: pd.DataFrame,
                 dist_function: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
                 max_linking_distance: float,
                 birth_death_cost: Callable[[pd.Series], float] | float,
                 max_skipped_frames: int = 1):

        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        self.objective = []

        n_frames = detections.frame.max() + 1
        n_detections = np.bincount(detections.frame.values, minlength=n_frames).tolist()

        source = Node('source')
        sink = Node('sink')

        birth_nodes = [BirthNode(frame) for frame in range(n_frames - 1)]
        death_nodes = [DeathNode(frame) for frame in range(1, n_frames)]

        self.nodes = [[] for _ in range(n_frames)]
        for detection in detections.itertuples():
            self.nodes[detection.frame].append(DetectionNode(detection.frame, detection.Index))

        for frame in range(n_frames - 1):
            distances = dist_function(detections[detections.frame == frame],
                                      detections[detections.frame == frame + 1])
            for ia, ib in np.argwhere(distances < max_linking_distance):
                self._create_edge(self.nodes[frame][ia], self.nodes[frame + 1][ib], distances[ia, ib], 1)

        skip_nodes = []
        for skipped_frames in range(2, max_skipped_frames + 2):
            for frame in range(0, n_frames - skipped_frames):
                distances = dist_function(detections[detections.frame == frame],
                                          detections[detections.frame == frame + skipped_frames])
                for ia, row in enumerate(distances):
                    indices = np.argwhere(row < max_linking_distance)
                    if indices.size == 0:
                        continue

                    skip_node = SkipNode(self.nodes[frame][ia], skipped_frames)
                    skip_nodes.append(skip_node)

                    self._create_edge(self.nodes[frame][ia], skip_node, 0, 1)

                    for ib in np.squeeze(indices, 1):
                        self._create_edge(skip_node, self.nodes[frame + skipped_frames][ib], distances[ia, ib] * skipped_frames, 1)

        merge_nodes, split_nodes = [], []
        for node in itertools.chain.from_iterable(self.nodes):
            if len(node.in_edges) > 1:
                merge_nodes.append(MergeNode(node))
            if len(node.out_edges) > 1:
                split_nodes.append(SplitNode(node))

        # Merge edges
        for merge_node in merge_nodes:
            from_nodes = [(edge.u, edge.unit_cost) for edge in merge_node.into_node.in_edges]
            self._create_edge(merge_node, death_nodes[merge_node.into_node.frame - 1], 0, len(from_nodes))
            self._create_edge(merge_node, merge_node.into_node, 0, 1)
            for from_node, cost in from_nodes:
                self._create_edge(from_node, merge_node, cost, 1)

        # Split edges
        for split_node in split_nodes:
            into_nodes = [(edge.v, edge.unit_cost) for edge in split_node.from_node.out_edges if isinstance(edge.v, DetectionNode)]
            self._create_edge(birth_nodes[split_node.frame - 1], split_node, 0, len(into_nodes))
            self._create_edge(split_node.from_node, split_node, 0, 1)
            for into_node, cost in into_nodes:
                self._create_edge(split_node, into_node, cost, 1)

        # Source edges
        for frame in range(n_frames - 1):
            self._create_edge(source, birth_nodes[frame], 0, n_detections[frame + 1])
        for node in self.nodes[0]:
            self._create_edge(source, node, 0, 1)

        # Sink edges
        for frame in range(n_frames - 1):
            self._create_edge(death_nodes[frame], sink, 0, sum([n_detections[frame - i] for i in range(-1, max_skipped_frames + 1)]))
        for node in self.nodes[-1]:
            self._create_edge(node, sink, 0, 1)

        # Birth edges
        for frame in range(n_frames - 1):
            for node in self.nodes[frame + 1]:
                self._create_edge(birth_nodes[frame], node, birth_death_cost(detections.iloc[node.index]) if callable(birth_death_cost) else birth_death_cost, 1)

            self._create_edge(birth_nodes[frame], death_nodes[frame], 0, n_detections[frame + 1])

        # Death edges
        for frame in range(n_frames - 1):
            for node in self.nodes[frame]:
                self._create_edge(node, death_nodes[frame], birth_death_cost(detections.iloc[node.index]) if callable(birth_death_cost) else birth_death_cost, 1)

        # Constraints
        for node in itertools.chain.from_iterable(self.nodes):
            node.create_flow_constraint(self.solver)
            node.create_assignment_constraint(self.solver)

        for birth_node in birth_nodes:
            birth_node.create_flow_constraint(self.solver)
            birth_node.create_assignment_constraint(self.solver, n_detections[birth_node.frame + 1])

        for death_node in death_nodes:
            death_node.create_flow_constraint(self.solver)

        for skip_node in skip_nodes:
            skip_node.create_flow_constraint(self.solver)

        for merge_node in merge_nodes:
            merge_node.create_flow_constraint(self.solver)
            merge_node.create_merge_constraint(self.solver)

        for split_node in split_nodes:
            split_node.create_flow_constraint(self.solver)
            split_node.create_split_constraint(self.solver)

        source.create_source_constraint(self.solver, sum(n_detections))

        # Define objective
        objective = self.solver.Sum(self.objective)
        self.solver.Minimize(objective)

    def _create_edge(self, u: Node, v: Node, unit_cost: float, capacity: int):
        flow = self.solver.IntVar(0, capacity, self.__str__())

        e = Edge(u, v, unit_cost, flow)
        u.out_edges.append(e)
        v.in_edges.append(e)

        if unit_cost > 0:
            self.objective.append(flow * unit_cost)

    def solve(self) -> int:
        return self.solver.Solve()

    def get_result(self) -> list[Tracklet]:
        tracklets = []
        node_tracklets = {}

        for node in itertools.chain.from_iterable(self.nodes):
            if node.frame == 0:
                t = Tracklet(len(tracklets))
                t.detections[0] = [node.index]

                node_tracklets[node.index] = t
                tracklets.append(t)

            else:
                edge = [edge for edge in node.in_edges if edge.flow.solution_value() > 0][0]
                match edge.u:
                    case BirthNode():
                        t = Tracklet(len(tracklets))
                        t.detections[node.frame] = [node.index]

                        node_tracklets[node.index] = t
                        tracklets.append(t)

                    case SkipNode() as in_node:
                        t = node_tracklets[in_node.from_node.index]

                        for frame in range(in_node.from_node.frame + 1, node.frame):
                            t.detections[frame] = [in_node.from_node.index, node.index]
                        t.detections[node.frame] = [node.index]

                        node_tracklets[node.index] = t

                    case SplitNode() as in_node:
                        t = Tracklet(len(tracklets))
                        t.detections[node.frame] = [node.index]

                        predecessor_tracklet = node_tracklets[in_node.index]
                        t.predecessors.add(predecessor_tracklet)
                        predecessor_tracklet.successors.add(t)

                        node_tracklets[node.index] = t
                        tracklets.append(t)

                    case MergeNode() as in_node:
                        t = Tracklet(len(tracklets))
                        t.detections[node.frame] = [node.index]

                        for edge in in_node.in_edges:
                            if edge.flow.solution_value() > 0:
                                if isinstance(edge.u, SkipNode):
                                    predecessor_tracklet = node_tracklets[edge.u.from_node.index]

                                    for frame in range(edge.u.from_node.frame + 1, node.frame):
                                        predecessor_tracklet.detections[frame] = [edge.u.from_node.index, node.index]

                                else:
                                    predecessor_tracklet = node_tracklets[edge.u.index]

                                t.predecessors.add(predecessor_tracklet)
                                predecessor_tracklet.successors.add(t)

                        node_tracklets[node.index] = t
                        tracklets.append(t)

                    case in_node:
                        t = node_tracklets[in_node.index]
                        t.detections[node.frame] = [node.index]
                        node_tracklets[node.index] = t

        return tracklets


class UntanglingGraph:
    def __init__(self, tracklets: list[Tracklet], gamma: float):
        self.solver = pywraplp.Solver.CreateSolver('SCIP')

        self.tracklets = tracklets

        # Variables
        self.edge_removals: dict[tuple[Tracklet, Tracklet], pywraplp.Solver.BoolVar] = {}
        self.tracklet_splits: dict[Tracklet, pywraplp.Solver.IntVar] = {}
        self.tracklet_merges: dict[tuple[Tracklet, ...], pywraplp.Solver.BoolVar] = {}

        # Maps tracklets to their merge sets
        self.merge_sets: dict[Tracklet, list[tuple[Tracklet, ...]]] = defaultdict(list)

        # Find merge sets
        tracklet_sets = set()
        for t in tracklets:
            for p in t.predecessors:
                self.edge_removals[(p, t)] = self.solver.BoolVar(f'e_{p.index}_{t.index}')

            self.tracklet_splits[t] = self.solver.IntVar(0, self.solver.infinity(), f's_{t.index}')

            predecessor_successors = [s for p in t.predecessors for s in p.successors]
            successor_predecessors = [p for s in t.successors for p in s.predecessors]
            parallel_tracklets = set(predecessor_successors + successor_predecessors) - {t}

            for r in range(1, len(parallel_tracklets) + 1):
                for tracklet_set in itertools.combinations(parallel_tracklets, r):
                    tracklet_set = tracklet_set + (t,)

                    unique_predecessors = {tuple(sorted(tracklet.predecessors))
                                           for tracklet in tracklet_set
                                           if tracklet.has_predecessors()}

                    if len(unique_predecessors) == 1:
                        t_end = max([p.t_end() for p in next(iter(unique_predecessors))])
                        same_predecessors = all([tracklet.t_start() > t_end for tracklet in tracklet_set])
                    else:
                        same_predecessors = len(unique_predecessors) == 0

                    unique_successors = {tuple(sorted(tracklet.successors))
                                         for tracklet in tracklet_set
                                         if tracklet.has_successors()}

                    if len(unique_successors) == 1:
                        t_start = min([p.t_start() for p in next(iter(unique_successors))])
                        same_successors = all([tracklet.t_end() < t_start for tracklet in tracklet_set])
                    else:
                        same_successors = len(unique_successors) == 0

                    if same_predecessors and same_successors:
                        tracklet_sets.add(tuple(sorted(tracklet_set)))

        for tracklet_set in tracklet_sets:
            self.tracklet_merges[tracklet_set] = self.solver.BoolVar(f'm_{"_".join(map(repr, tracklet_set))}')
            for t in tracklet_set:
                self.merge_sets[t].append(tracklet_set)

        # Predecessor constraints
        for t in tracklets:
            if t.has_predecessors():
                remove_edges_to_predecessors = sum([self.edge_removals[(p, t)] for p in t.predecessors])

                merge_predecessors = sum(
                    [(len([tracklet for tracklet in tracklet_set if tracklet in t.predecessors]) - 1)
                     * self.tracklet_merges[tracklet_set]
                     for tracklet_set
                     in {tracklet_set for p in t.predecessors for tracklet_set in self.merge_sets[p]}])

                split_self = self.tracklet_splits[t]

                number_of_predecessors = len(t.predecessors)

                self.solver.Add(-remove_edges_to_predecessors - merge_predecessors - split_self <= 1 - number_of_predecessors)

        # Successor constraints
        for t in tracklets:
            if t.has_successors():
                remove_edges_to_successors = sum([self.edge_removals[(t, s)] for s in t.successors])

                merge_successors = sum([(len([tracklet for tracklet in tracklet_set if tracklet in t.successors]) - 1)
                                        * self.tracklet_merges[tracklet_set]
                                        for tracklet_set
                                        in {tracklet_set for s in t.successors for tracklet_set in self.merge_sets[s]}])

                split_self = self.tracklet_splits[t]

                number_of_successors = len(t.successors)

                self.solver.Add(-remove_edges_to_successors - merge_successors - split_self <= 1 - number_of_successors)

        # A tracklet can be merged with at most one set of tracklets, or it can be split
        for t in tracklets:
            if self.merge_sets[t]:
                for merge_set in self.merge_sets[t]:
                    self.solver.Add((1 - self.tracklet_merges[merge_set]) * len(tracklets) >= self.tracklet_splits[t])

                if len(self.merge_sets[t]) > 1:
                    self.solver.Add(sum([self.tracklet_merges[merge_set] for merge_set in self.merge_sets[t]]) <= 1)

        # Enforce only a single edge cut in merging sets
        for merge_set, track_merge in self.tracklet_merges.items():
            predecessors = {s for t in merge_set for s in t.predecessors}
            for p in predecessors:
                removals = [self.edge_removals[(p, t)] for t in merge_set if p in t.predecessors]
                if len(removals) > 1:
                    self.solver.Add(1 + len(removals) * (1 - track_merge) >= sum(removals))

            successors = {s for t in merge_set for s in t.successors}
            for s in successors:
                removals = [self.edge_removals[(t, s)] for t in merge_set if s in t.successors]
                if len(removals) > 1:
                    self.solver.Add(1 + len(removals) * (1 - track_merge) >= sum(removals))

            successor_merge_sets = {ms for s in successors for ms in self.merge_sets[s]}
            for other_merge_set in successor_merge_sets:
                removals = [self.edge_removals[(p, s)] for p in merge_set for s in other_merge_set if s in p.successors]
                if len(removals) > 1:
                    other_track_merge = self.tracklet_merges[other_merge_set]
                    self.solver.Add(1 + len(removals) * (1 - track_merge) + len(removals) * (1 - other_track_merge) >= sum(removals))

        # Define objective
        objective = sum(self.edge_removals.values()) * gamma \
                    + sum([len(tracklet) * tracklet_split
                           for tracklet, tracklet_split in self.tracklet_splits.items()]) \
                    + sum([sum(sorted([len(tracklet) for tracklet in merge_set])[:-1]) * tracklet_merge
                           for merge_set, tracklet_merge in self.tracklet_merges.items()])

        self.solver.Minimize(objective)

    def solve(self) -> int:
        return self.solver.Solve()

    def get_result(self, detections: pd.DataFrame) -> pd.DataFrame:
        tracklets_to_remove = set()

        # Cut edges
        for (p, s), e in self.edge_removals.items():
            if e.solution_value() > 0:
                ps = [p]
                for tracklet_set in self.merge_sets[p]:
                    if self.tracklet_merges[tracklet_set].solution_value() > 0:
                        ps = tracklet_set

                ss = [s]
                for tracklet_set in self.merge_sets[s]:
                    if self.tracklet_merges[tracklet_set].solution_value() > 0:
                        ss = tracklet_set

                for p in ps:
                    for s in ss:
                        if s in p.successors:
                            p.successors.remove(s)
                            s.predecessors.remove(p)

        # Merge tracks
        for tracklet_set, tracklet_merge in self.tracklet_merges.items():
            if tracklet_merge.solution_value() > 0:
                for t in tracklet_set[1:]:
                    tracklets_to_remove.add(t)

                    for p in t.predecessors:
                        p.successors.remove(t)

                    for s in t.successors:
                        s.predecessors.remove(t)

                merged = defaultdict(list)
                for t in tracklet_set:
                    for f, d in t.detections.items():
                        merged[f].extend(d)

                tracklet_set[0].detections = merged

        # Split tracks
        for t, tracklet_split in self.tracklet_splits.items():
            if tracklet_split.solution_value() > 0:
                tracklets_to_remove.add(t)

                n_split = int(tracklet_split.solution_value()) + 1

                new_tracklets = []
                for _ in range(n_split):
                    new_tracklet = Tracklet(len(self.tracklets))
                    new_tracklet.detections = t.detections.copy()
                    new_tracklets.append(new_tracklet)
                    self.tracklets.append(new_tracklet)

                for p in t.predecessors:
                    p.successors.remove(t)

                for s in t.successors:
                    s.predecessors.remove(t)

                if t.predecessors and t.successors:
                    predecessors = list(t.predecessors)
                    successors = list(t.successors)

                    # Find optimal assignment
                    p_xy = [tuple(detections.iloc[p.detections[p.t_end()]][['x', 'y']].mean()) for p in predecessors]
                    s_xy = [tuple(detections.iloc[s.detections[s.t_start()]][['x', 'y']].mean()) for s in successors]
                    distances = spatial.distance.cdist(np.array(p_xy, dtype=np.float32),
                                                       np.array(s_xy, dtype=np.float32), metric='euclidean')
                    indices = optimize.linear_sum_assignment(distances)

                    for p_i, s_i, new_tracklet in zip(*indices, new_tracklets):
                        p = predecessors[p_i]
                        p.successors.add(new_tracklet)
                        new_tracklet.predecessors.add(p)

                        s = successors[s_i]
                        s.predecessors.add(new_tracklet)
                        new_tracklet.successors.add(s)

                        last_p_detection = p.detections[p.t_end()]
                        first_s_detection = s.detections[s.t_start()]
                        for frame in new_tracklet.detections.keys():
                            new_tracklet.detections[frame] = new_tracklet.detections[frame] + last_p_detection + first_s_detection

                n_diff = len(t.predecessors) - len(t.successors)
                if n_diff > 0:
                    for p, new_tracklet in zip([p for p in t.predecessors if not p.has_successors()], new_tracklets[-n_diff:]):
                        p.successors.add(new_tracklet)
                        new_tracklet.predecessors.add(p)

                        last_p_detection = p.detections[p.t_end()]
                        for frame in new_tracklet.detections.keys():
                            new_tracklet.detections[frame] = new_tracklet.detections[frame] + last_p_detection

                elif n_diff < 0:
                    for s, new_tracklet in zip([s for s in t.successors if not s.has_predecessors()], new_tracklets[n_diff:]):
                        s.predecessors.add(new_tracklet)
                        new_tracklet.successors.add(s)

                        first_s_detection = s.detections[s.t_start()]
                        for frame in new_tracklet.detections.keys():
                            new_tracklet.detections[frame] = new_tracklet.detections[frame] + first_s_detection

        # Connect tracks with 1:1 mappings
        for t in self.tracklets:
            if t in tracklets_to_remove:
                continue

            while t.has_successors():
                successor = next(iter(t.successors))

                tracklets_to_remove.add(successor)

                t.detections |= successor.detections
                t.successors = successor.successors

        # Generate result
        particles, x_coords, y_coords, classes, frames = [], [], [], [], []

        for t in self.tracklets:
            if t in tracklets_to_remove:
                continue

            for f, detection in t.detections.items():
                d = detections.iloc[detection]
                x_coords.append(np.mean(d.x))
                y_coords.append(np.mean(d.y))
                classes.append(np.mean(d.cls))
                frames.append(f)
                particles.append(t.index)

        data = {
            'particle': np.array(particles),
            'x': np.array(x_coords),
            'y': np.array(y_coords),
            'cls': np.array(classes),
            'frame': np.array(frames)
        }

        return pd.DataFrame(data=data)


def print_solver_status(status, solver):
    match status:
        case solver.OPTIMAL:
            print(f'Optimal solution with objective value {solver.Objective().Value():.2f}'
                  f' found in {solver.wall_time() / 1000:.1f}s and {solver.iterations()} iterations')
        case solver.FEASIBLE: print('Problem is feasible, or stopped by time limit')
        case solver.INFEASIBLE: print('Problem is infeasible')
        case solver.UNBOUNDED: print('Problem is unbounded')
        case solver.ABNORMAL: print('Error of some kind')
        case _: print(f'Unknown status {status}')
