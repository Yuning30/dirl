"""
Constructing task automaton and abstract reachability graph from spec.
"""

from copy import copy
from spectrl.main.spec_compiler import Cons, land, TaskSpec
from spectrl.hierarchy.reachability import AbstractEdge, AbstractReachability


class TaskAutomaton:
    """
    Task Automaton without registers.

    Parameters:
        delta: list of list of (int, predicate) pairs.
               predicate: state, resource -> float.
        final_states: set of int (final monitor states).

    Initial state is assumed to be 0.
    """

    def __init__(self, delta, final_states):
        self.delta = delta
        self.final_states = final_states
        self.num_states = len(self.delta)


def automaton_graph_from_spec(spec: TaskSpec):
    """
    Constructs task automaton and abstract reachability graph from the specification.

    Parameters:
        spec: TaskSpec

    Returns:
        (automaton, abstract_reach): TaskAutomaton * AbstractReachability
    """
    automaton = None
    if spec.cons == Cons.ev:
        # Case 1: Objectives

        # Step 1b: Construct abstract graph
        abstract_graph = [
            [AbstractEdge(1, spec.predicate, [true_pred])],
            [],
        ]
        abstract_reach = AbstractReachability(abstract_graph, set([1]))

    elif spec.cons == Cons.alw:
        # Case 2: Constraints

        # Step 2a: Get automaton and graph for subtask
        _, r1 = automaton_graph_from_spec(spec.subtasks[0])

        # Step 2c: Construct abstract graph
        abstract_graph = []
        for edges in r1.abstract_graph:
            new_edges = []
            for edge in edges:
                if edge.predicate is not None:
                    new_predicate = land(edge.predicate, spec.predicate)
                else:
                    new_predicate = None
                new_constraints = [land(b, spec.predicate) for b in edge.constraints]
                new_edges.append(
                    AbstractEdge(edge.target, new_predicate, new_constraints)
                )
            abstract_graph.append(new_edges)
        abstract_reach = AbstractReachability(abstract_graph, set(r1.final_vertices))

    elif spec.cons == Cons.seq:
        # Case 3: Sequencing
        # import pdb
        # pdb.set_trace()

        # Step 3a: Get automaton and graph for subtasks
        _, r1 = automaton_graph_from_spec(spec.subtasks[0])
        _, r2 = automaton_graph_from_spec(spec.subtasks[1])

        # Step 3c: Construct abstract graph
        abstract_graph = [[copy_edge(e) for e in edges] for edges in r1.abstract_graph]

        for edges in r2.abstract_graph[1:]:
            new_edges = []
            for e in edges:
                new_target = e.target + r1.num_vertices - 1
                new_edges.append(
                    AbstractEdge(new_target, e.predicate, copy(e.constraints))
                )
            abstract_graph.append(new_edges)

        for v in r1.final_vertices:
            # abstract_graph[v] = []
            for e in r2.abstract_graph[0]:
                new_target = e.target + r1.num_vertices - 1
                new_constraints = e.constraints
                abstract_graph[v].append(
                    AbstractEdge(new_target, e.predicate, new_constraints)
                )

        final_vertices = set([t + r1.num_vertices - 1 for t in r2.final_vertices])
        abstract_reach = AbstractReachability(abstract_graph, final_vertices)

    elif spec.cons == Cons.choose:
        # Case 4: Choice

        # Step 4a: Get automaton and graph for subtasks
        _, r1 = automaton_graph_from_spec(spec.subtasks[0])
        _, r2 = automaton_graph_from_spec(spec.subtasks[1])

        # Step 4c: Construct abstract graph
        abstract_graph = [[]]
        for e in r1.abstract_graph[0]:
            abstract_graph[0].append(copy_edge(e))
        for e in r2.abstract_graph[0]:
            new_target = e.target + r1.num_vertices - 1
            abstract_graph[0].append(
                AbstractEdge(new_target, e.predicate, copy(e.constraints))
            )

        for edges in r1.abstract_graph[1:]:
            new_edges = [copy_edge(e) for e in edges]
            abstract_graph.append(new_edges)

        for edges in r2.abstract_graph[1:]:
            new_edges = []
            for e in edges:
                new_target = e.target + r1.num_vertices - 1
                new_edges.append(
                    AbstractEdge(new_target, e.predicate, copy(e.constraints))
                )
            abstract_graph.append(new_edges)

        final_vertices = r1.final_vertices.union(
            set([t + r1.num_vertices - 1 for t in r2.final_vertices])
        )
        abstract_reach = AbstractReachability(abstract_graph, final_vertices)

    elif spec.cons == Cons.repeat:
        if spec.subtasks[0].cons == Cons.ev:
            abstract_graph = automaton_graph_from_spec(spec.subtasks[0])
        elif spec.subtasks[0].cons == Cons.alw:
            _, r = automaton_graph_from_spec(spec.subtasks[0])

            neighbor_of_initial = [e.target for e in r.abstract_graph[0]]
            # add an edge from every final vertex to every neighbor of the initial vertex
            for final_vertex in r.final_vertices:
                for neighbor in neighbor_of_initial:
                    r.abstract_graph[final_vertex].append(
                        AbstractEdge(neighbor, None, [true_pred])
                    )

            # add the constraint
            abstract_graph = []
            for edges in r.abstract_graph:
                new_edges = []
                for edge in edges:
                    if edge.predicate is not None:
                        new_predicate = land(edge.predicate, spec.predicate)
                    else:
                        new_predicate = None
                    new_constraints = [
                        land(b, spec.predicate) for b in edge.constraints
                    ]
                    new_edges.append(
                        AbstractEdge(edge.target, new_predicate, new_constraints)
                    )
                abstract_graph.append(new_edges)
            abstract_reach = AbstractReachability(abstract_graph, set(r.final_vertices))
        elif spec.subtasks[0].cons == Cons.seq:
            _, r = automaton_graph_from_spec(spec.subtasks[0])
            neighbor_of_initial = [e.target for e in r.abstract_graph[0]]
            # add an edge from every final vertex to every neighbor of the initial vertex
            for final_vertex in r.final_vertices:
                for neighbor in neighbor_of_initial:
                    r.abstract_graph[final_vertex].append(
                        AbstractEdge(neighbor, None, [true_pred])
                    )

            abstract_reach = r
        elif spec.subtasks[0].cons == Cons.choose:
            _, r = automaton_graph_from_spec(spec.subtasks[0])
            neighbor_of_initial = [e.target for e in r.abstract_graph[0]]
            # add an edge from every final vertex to every neighbor of the initial vertex
            for final_vertex in r.final_vertices:
                for neighbor in neighbor_of_initial:
                    r.abstract_graph[final_vertex].append(
                        AbstractEdge(neighbor, None, [true_pred])
                    )

            abstract_reach = r
        elif spec.subtasks[0].cons == Cons.repeat:
            # ignore repeat over repeat
            abstract_graph = automaton_graph_from_spec(spec.subtasks[0])
        else:
            assert False

    return automaton, abstract_reach


def true_pred(sys_state):
    return 0


def copy_edge(edge):
    return AbstractEdge(edge.target, edge.predicate, copy(edge.constraints))
