import torch
import torch.nn as nn
import numpy as np
import pdb

from spectrl.training.control.rl.utils.gym_env import GymEnv
from spectrl.training.control.rl.policies.gaussian_prog import *
from spectrl.training.control.rl.utils.fc_network import FCNetwork
from spectrl.training.control.rl.baselines.mlp_baseline import MLPBaseline
from spectrl.training.control.rl.algos.npg_cg import NPG
from spectrl.training.control.rl.algos.trpo import TRPO

# from spectrl.training.control.rl.algos.ppo_clip import PPO
from spectrl.training.control.rl.utils.train_agent import train_agent

from gym.utils import seeding
from timeit import default_timer as timer
from os import path
import dill as pickle

# import marvelgym as gym

# from marvelgym.spec_env import SpecEnv
import cv2


def createProgNetwork(prog_type, observation_dim, action_dim):
    if prog_type == "Linear":
        prog = LinearProgNetwork(observation_dim, action_dim)
    elif prog_type == "ITELinear":
        prog = ITELinearProgNetwork(observation_dim, action_dim)
    elif prog_type == "NestITELinear":
        prog = NestITELinearProgNetwork(observation_dim, action_dim)
    elif prog_type == "Nest2ITELinear":
        prog = Nest2ITELinearProgNetwork(observation_dim, action_dim)
    elif prog_type == "ITEConstant":
        prog = ITEConstantProgNetwork(observation_dim, action_dim)
    elif prog_type == "NestITEConstant":
        prog = NestITEConstantProgNetwork(observation_dim, action_dim)
    elif prog_type == "Nest2ITEConstant":
        prog = Nest2ITEConstantProgNetwork(observation_dim, action_dim)
    elif prog_type == "PendulumPID":
        prog = PendulumPIDProgNetwork(observation_dim, action_dim)
    elif prog_type == "LunarLanderPD":
        prog = LunarLanderPDProgNetwork(observation_dim, action_dim)
    elif prog_type == "MLP":
        prog = FCNetwork(
            observation_dim,
            action_dim,
            hidden_sizes=(64, 64),
            nonlinearity="tanh",
            bias=True,
        )
    else:
        assert False

    return prog


def train_and_verify(
    seed,
    env,
    verify_prog,
    prog_type,
    trainsteps,
    init_distribution,
    goal_region,
    constraints,
):
    # pdb.set_trace()
    trained_policy = train_policy(env, prog_type, trainsteps, seed)
    # pdb.set_trace()
    end_states = estimate_reach_region(env, trained_policy)
    # pdb.set_trace()
    VEL_goal_region = compute_VEL_goal_region(goal_region, end_states)
    # pdb.set_trace()
    # verified_policy = verify(
        # verify_prog, trained_policy, init_distribution, VEL_goal_region, constraints
    # )
    success = True
    if success:
        return trained_policy, VEL_goal_region
    return None, goal_region


def verify(verify_prog, policy, init_distribution, goal_region, constraints):
    # TODO
    verified_policy = None
    return verified_policy


def estimate_reach_region(env, policy, rollouts=100):
    end_states = []
    for i in range(0, rollouts):
        state, done = env.reset(), False
        while not done:
            action = policy.get_action(state)[1]["mean"]
            next_state, rwd, done, _ = env.step(action)
            state = next_state

        end_states.append(state)

    return np.stack(end_states)


def compute_VEL_goal_region(gt_goal_region, estimated_end_states):
    # now assume that gt_goal_region and estimated_end_state have same dimension
    # partial goal region will be considered later in cartpole
    estimated_lower = np.min(estimated_end_states, axis=0)
    estimated_upper = np.max(estimated_end_states, axis=0)

    inter_lower = np.where(estimated_lower > gt_goal_region[0], estimated_lower, gt_goal_region[0])
    inter_upper = np.where(estimated_upper < gt_goal_region[1], estimated_upper, gt_goal_region[1])
    intersect = np.all(inter_lower < inter_upper)

    if intersect:
        return np.array([inter_lower, inter_upper])
    return gt_goal_region


def train_policy(env, prog_type, trainsteps, seed):
    prog = createProgNetwork(prog_type, env.observation_dim, env.action_dim)
    policy = ProgPolicy(
        env.observation_dim, env.action_dim, prog=prog, seed=seed
    )  # LinearPolicy(e.spec, seed=SEED)
    baseline = MLPBaseline(
        env.observation_dim, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3
    )
    agent = TRPO(
        env,
        policy,
        baseline,
        normalized_step_size=0.1,
        kl_dist=None,
        seed=seed,
        save_logs=True,
    )
    agent = train_agent(
        job_name="9rooms",  # No. Train with rewards
        agent=agent,
        seed=seed,
        niter=trainsteps,
        gamma=0.995,
        gae_lambda=0.97,
        num_cpu=4,
        sample_mode="trajectories",
        num_traj=100,
        save_freq=5,
        evaluation_rollouts=5,
        out_dir="data/",
    )
    return agent


def train(
    env_name, prog_type, trainsteps, SEED, phaselearning=False, phase_trainiter_inc=0
):
    # import pdb
    # pdb.set_trace()
    try:
        e = GymEnv(gym.make(env_name).gym_env)
    except:
        e = GymEnv(env_name)

    if phaselearning:  # learning in different phases
        time_limit = e.env._max_episode_steps / e.env.env.specs_size()
        print(f"episode_steps for each learning phase {time_limit}")
        e.env._max_episode_steps = time_limit
        e.env.env.set_timelimit(time_limit)
    else:
        prog = createProgNetwork(prog_type, e.spec.observation_dim, e.spec.action_dim)

    e.set_seed(SEED)  # random search uses a fixed seed.
    # state_shape = e.spec.observation_dim
    # action_shape = e.spec.action_dim  # number of actions
    # step, noise, ndr, bdr = .1, .3, 16, 8
    # rl_agent = ARS_V1(alpha=step, noise=noise, N_of_directions=ndr, b=bdr, training_length=50)
    # rl_agent.train(e, policy, Normalizer(state_shape))
    # agent = NPG(e, policy, baseline, normalized_step_size=0.1, seed=SEED, save_logs=True)
    # agent = PPO(e, policy, baseline, clip_coef=0.2, epochs=10, mb_size=64, learn_rate=3e-4, seed=seed, save_logs=True)

    if not phaselearning:  # do we have specs to solve? No.
        policy = ProgPolicy(
            e.spec, prog=prog, seed=SEED
        )  # LinearPolicy(e.spec, seed=SEED)
        baseline = MLPBaseline(
            e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3
        )
        agent = TRPO(
            e,
            policy,
            baseline,
            normalized_step_size=0.1,
            kl_dist=None,
            seed=seed,
            save_logs=True,
        )
        train_agent(
            job_name=env_name,  # No. Train with rewards
            agent=agent,
            seed=SEED,
            niter=trainsteps,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=4,
            sample_mode="trajectories",
            num_traj=100,
            save_freq=5,
            evaluation_rollouts=5,
            out_dir="data/",
        )
    else:  # Yes. Train with the specs
        for i in range(e.env.env.specs_size()):
            prog = createProgNetwork(
                prog_type, e.spec.observation_dim, e.spec.action_dim
            )
            policy = ProgPolicy(e.spec, prog=prog, seed=SEED)
            baseline = MLPBaseline(
                e.spec, reg_coef=1e-3, batch_size=64, epochs=2, learn_rate=1e-3
            )
            agent = TRPO(
                e,
                policy,
                baseline,
                normalized_step_size=0.1,
                kl_dist=None,
                seed=seed,
                save_logs=True,
            )
            print(f"------------- Training phase {i} -------------")
            env_name_ph = env_name + "_" + str(i)
            best_policy_at, best_perf = train_agent(
                job_name=env_name_ph,
                agent=agent,
                seed=SEED,
                niter=trainsteps + (i * phase_trainiter_inc),
                gamma=0.995,
                gae_lambda=0.97,
                num_cpu=1,
                sample_mode="trajectories",
                num_traj=100,
                save_freq=5,
                evaluation_rollouts=5,
                out_dir="data/",
            )
            pi = "data/" + env_name_ph + "/iterations/best_policy.pickle"
            policy = pickle.load(open(pi, "rb"))
            e.env.env.advance(policy)


def visualize_policy(
    env_name, num_episodes=1, mode="exploration", discrete=False, render=True
):
    try:
        e = GymEnv(gym.make(env_name).gym_env)
    except:
        e = GymEnv(env_name)
    horizon = e._horizon

    if False:
        for i in range(e.env.env.specs_size()):
            env_name_ph = env_name + "_" + str(i)
            pi = "data/" + env_name_ph + "/iterations/best_policy.pickle"
            policy = pickle.load(open(pi, "rb"))
            e.env.env.advance(policy)

        time_limit = e.env._max_episode_steps / e.env.env.specs_size()
        print(f"episode_steps for each learning phase {time_limit}")
        e.env.env.set_timelimit(time_limit)
        e.env.env.eval(horizon, num_episodes, mode, discrete)
    else:
        pi = "data/" + env_name + "/iterations/best_policy.pickle"
        policy = pickle.load(open(pi, "rb"))
        print(f"policy type = {type(policy.model)}")
        total_score = 0.0
        f = 0
        frame = 0
        for ep in range(num_episodes):
            o = e.reset()
            d = False
            t = 0
            score = 0.0
            while t < horizon and d == False:
                if render:
                    # import pdb
                    # pdb.set_trace()
                    rst = e.render()
                    # cv2.imwrite(f"images/{frame:04d}.png", rst)

                    frame += 1
                a = (
                    policy.get_action(o, discrete=discrete)[0]
                    if mode == "exploration"
                    else policy.get_action(o, discrete=discrete)[1]["evaluation"]
                )
                o, r, d, _ = e.step(a)
                t = t + 1
                score = score + r
            print("Episode score = %f" % score)
            total_score += score
            f = f + 1 if score != 500 else f
        print(f"averaged score: {total_score / num_episodes}")
        print(f"succ rate {(num_episodes - f) / num_episodes}")
    del e


def interpret_policy(env_name, phaselearning=False):
    if phaselearning:
        e = GymEnv(gym.make(env_name).gym_env)
        for i in range(e.env.env.specs_size()):
            env_name_ph = env_name + "_" + str(i)
            pi = "data/" + env_name_ph + "/iterations/best_policy.pickle"
            policy = pickle.load(open(pi, "rb"))
            print(policy.model.interpret())
            print(f"---> {e.env.env.specs[i]}")
        del e
    else:
        pi = "data/" + env_name + "/iterations/best_policy.pickle"
        policy = pickle.load(open(pi, "rb"))
        print(policy.model.interpret())


def save_policy(env_name, phaselearning=False):
    if phaselearning:
        e = GymEnv(gym.make(env_name).gym_env)
        for i in range(e.env.env.specs_size()):
            env_name_ph = env_name + "_" + str(i)
            pi = "data/" + env_name_ph + "/iterations/best_policy.pickle"
            policy = pickle.load(open(pi, "rb"))
            print(policy.model.interpret())
            print(f"---> {e.env.env.specs[i]}")
        del e
    else:
        pi = "data/" + env_name + "/iterations/best_policy.pickle"
        policy = pickle.load(open(pi, "rb"))
        policy.model.save_model()


if __name__ == "__main__":
    start = timer()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", action="store", dest="env_name", default="Pendulum")
    parser.add_argument("--eval", action="store_true", dest="eval")
    parser.set_defaults(eval=False)
    parser.add_argument("--interpret", action="store_true", dest="interpret")
    parser.set_defaults(interpret=False)
    parser.add_argument("--save_model", action="store_true", dest="save_model")
    parser.set_defaults(sace_model=False)
    parser.add_argument("--seed", action="store", dest="seed", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    print(f"seed is {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    benchmark = args.env_name
    eval = args.eval
    interpret = args.interpret
    save_model = args.save_model

    # Default is that we do not have phase-based training
    phaselearning = False
    phase_trainiter_inc = 0

    # Experiments for now
    if benchmark in ["Pendulum", "Pendulum-v0", "InvertedPendulum-v2", "Swimmer-v3"]:
        prog_type = "ITELinear"
        trainsteps = 50
    elif benchmark in [
        "Acc",
        "Bicycle",
        "Hopper-v2",
        "HalfCheetah-v2",
        "BipedalWalker-v3",
        "Acc2",
    ]:
        # For Bicycle (obstacle), need at least 300 iterations to converge to the target. Better to be 500.
        prog_type = "ITELinear"
        trainsteps = 300
    elif benchmark in [
        "CartPole",
        "MountainCarContinuous-v0",
        "MountainCar",
        "Acrobot",
        "MountainCarSpeed",
    ]:
        prog_type = "Nest2ITELinear"
        # prog_type = 'NestITELinear'
        # prog_type = 'ITELinear'
        # prog_type = 'Linear'
        trainsteps = 1000
    elif benchmark in ["CarRetrieval"]:
        prog_type = "NestITEConstant"
        trainsteps = 150
    elif benchmark in ["Quad", "QuadFull", "QuadTest", "QuadFullTest"]:
        prog_type = "MLP"
        trainsteps = 150
    elif benchmark in [
        "LunarLander",
        "InvertedDoublePendulum-v2",
        "Humanoid-v2",
        "Ant-v2",
        "Walker2d-v2",
        "Reacher-v2",
    ]:
        prog_type = "NestITELinear"
        trainsteps = 200
    elif benchmark in ["LunarLanderContinuous-v2"]:
        prog_type = "LunarLanderPD"
        trainsteps = 100
    elif benchmark in ["PendulumPID"]:
        prog_type = "PendulumPID"
        trainsteps = 100
    elif benchmark in ["Car2d2"]:
        prog_type = "ITELinear"
        trainsteps = 300
    elif benchmark in ["CarRacing", "Car2d"]:
        prog_type = "Linear"
        trainsteps = 100  # each phase trained using 100 iterations.
        phaselearning = True
        phase_trainiter_inc = 20  # i-th phase trained 20 iterations more than i-1.
    elif benchmark in ["CarMaze"]:
        prog_type = "ITELinear"
        trainsteps = 100  # each phase trained using 100 iterations.
        phaselearning = True
        phase_trainiter_inc = 100  # i-th phase trained 100 iterations more than i-1.
    elif benchmark in ["CarFall", "CarPush"]:  # Test on very simple environments.
        prog_type = "NestITEConstant"
        trainsteps = 100  # each phase trained using 100 iterations.
        phaselearning = True
        phase_trainiter_inc = 100  # i-th phase trained 100 iterations more than i-1.
    elif benchmark in ["TORA", "TORAEq"]:
        prog_type = "Linear"
        trainsteps = 200  # each phase trained using 100 iterations.
    elif benchmark in ["ReachNN1"]:
        prog_type = "Linear"
        trainsteps = 50
    elif benchmark in [
        "ReachNN2",
        "ReachNN3",
        "ReachNN4",
        "ReachNN5",
        "ReachNN6",
        "OS",
    ]:
        prog_type = "Linear"
        trainsteps = 100
    elif benchmark in ["UnicycleCar"]:
        prog_type = "ITELinear"
        trainsteps = 100
    elif benchmark in ["AccCAV", "AccCMP"]:
        prog_type = "ITELinear"
        trainsteps = 350
    elif benchmark in ["PP"]:
        prog_type = "Linear"
        trainsteps = 150
    elif benchmark in ["QMPC"]:
        prog_type = "Nest2ITEConstant"
        trainsteps = 260
    elif benchmark in ["meta_room"]:
        prog_type = "Linear"
        trainsteps = 300
    else:
        assert False

    if eval:
        visualize_policy(benchmark, num_episodes=5, mode="evaluation", discrete=False)
    elif interpret:
        interpret_policy(benchmark, phaselearning=phaselearning)
    elif save_model:
        save_policy(benchmark, phaselearning=phaselearning)
    else:
        train(
            benchmark,
            prog_type,
            trainsteps,
            seed,
            phaselearning=phaselearning,
            phase_trainiter_inc=phase_trainiter_inc,
        )
        save_policy(benchmark, phaselearning=phaselearning)

    print(f"Total Cost Time: {timer() - start}s")
    pass
