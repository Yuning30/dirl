from spectrl.hierarchy.construction import automaton_graph_from_spec
from spectrl.hierarchy.reachability import HierarchicalPolicy, ConstrainedEnv
from spectrl.main.spec_compiler import ev, seq, choose, alw, repeat
from spectrl.util.io import parse_command_line_options, save_log_info
from spectrl.util.rl import print_performance, get_rollout
from spectrl.rl.ars import HyperParams

from spectrl.examples.rooms_envs import (
    GRID_PARAMS_LIST,
    MAX_TIMESTEPS,
    START_ROOM,
    FINAL_ROOM,
    VEL_GRID_PARAMS_LIST
)
from spectrl.envs.rooms import RoomsEnv

import os
import copy

num_iters = [50, 100, 200, 300, 400, 500]

# Construct Product MDP and learn policy
if __name__ == "__main__":
    flags = parse_command_line_options()
    render = flags["render"]
    env_num = flags["env_num"]
    folder = flags["folder"]
    itno = flags["itno"]
    spec_num = flags["spec_num"]

    log_info = []

    for i in num_iters:

        grid_params = VEL_GRID_PARAMS_LIST[0]

        hyperparams = HyperParams(30, i, 30, 15, 0.05, 0.3, 0.15)

        print(
            "\n**** Learning Policy for Spec #{} in Env #{} ****".format(
                spec_num, env_num
            )
        )

        # Step 1: initialize system environment
        spec3 = seq(
            ev(grid_params.in_room(4)),
            seq(ev(grid_params.in_room(7)),
            seq(ev(grid_params.in_room(8)),
            ev(grid_params.in_room(5))))
        )

        spec11 = seq(ev(grid_params.in_room(1)), repeat(spec3))
        # import pdb

        # pdb.set_trace()
        print(spec11)
        _, abstract_reach = automaton_graph_from_spec(spec11)
        print("\n**** Abstract Graph ****")
        abstract_reach.pretty_print()
        abstract_reach.plot()
        # exit()
        import pdb

        pdb.set_trace()

        # Step 5: Learn policy
        abstract_policy, nn_policies, stats = abstract_reach.learn_dijkstra_policy(
            grid_params,
            hyperparams,
            algo="vel",
            res_model=None,
            max_steps=100,
            render=render,
            neg_inf=-100,
            safety_penalty=-1,
            num_samples=500,
        )

        # Test policy
        hierarchical_policy = HierarchicalPolicy(
            abstract_policy, nn_policies, abstract_reach.abstract_graph, 2
        )
        final_env = ConstrainedEnv(
            system,
            abstract_reach,
            abstract_policy,
            res_model=None,
            max_steps=MAX_TIMESTEPS[env_num],
        )

        # Print statements
        _, prob = print_performance(
            final_env, hierarchical_policy, stateful_policy=True
        )
        print("\nTotal Sample Steps: {}".format(stats[0]))
        print("Total Time Taken: {} mins".format(stats[1]))
        print("Total Edges Learned: {}".format(stats[2]))

        # Render
        if render:
            print("\nSimulation with learned policy...")
            get_rollout(final_env, hierarchical_policy, True, stateful_policy=True)

        logdir = os.path.join(folder, "spec{}".format(spec_num), "hierarchy")
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        log_info.append([stats[0], stats[1], prob])

    save_log_info(log_info, itno, logdir)
