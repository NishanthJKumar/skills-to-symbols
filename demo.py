from envs.continuous_playroom import ContinuousPlayroomEnv, create_playroom_options, \
    partition_playroom_options, create_playroom_pres_and_effs
from algorithms import create_operators_from_options
from structs import DecisionTreeClassifier
from utils import create_domain_file, create_problem_file, run_planner
import imageio
import numpy as np

def render_continuous_playroom_initial_state():
    env = ContinuousPlayroomEnv()
    obs = env.reset()
    print("Initial obs:", obs)
    img = env.render()
    imageio.imwrite("continuous_playroom.png", img)
    print("Wrote out to continuous_playroom.png.")

def demo_continuous_playroom_random_actions():
    env = ContinuousPlayroomEnv()
    obs = env.reset()
    img = env.render()
    images = [img]
    for _ in range(25):
        action = env.get_random_action()
        obs, _, _, _ = env.step(action)
        images.append(env.render())
    imageio.mimwrite("continuous_playroom_random_actions.mp4", images)
    print("Wrote out to continuous_playroom_random_actions.mp4.")

def demo_continuous_playroom_hardcoded_actions():
    env = ContinuousPlayroomEnv(seed=0)
    obs = env.reset()
    img = env.render()
    images = [img]
    actions = []
    # Move the hand towards the light
    actions += [("move", "hand", (0.05, -0.04))] * 5
    # Move the eye towards the light
    actions += [("move", "eye", (0.05, -0.018))] * 7
    # Turn the light on
    actions += [("interact", "light_switch")]
    # Move hand towards the green button
    actions += [("move", "hand", (0.05, -0.04))] * 3
    # Move eye towards the green button
    actions += [("move", "eye", (0.05, -0.05))] * 3
    # Turn on the music
    actions += [("interact", "green")]
    # Move the hand towards the light
    actions += [("move", "hand", (-0.05, 0.04))] * 3
    # Move the eye towards the light
    actions += [("move", "eye", (-0.05, 0.05))] * 3
    # Turn the light off
    actions += [("interact", "light_switch")]
    # Move the hand towards the ball
    actions += [("move", "hand", (-0.05, -0.03))] * 14
    # Move the eye towards the ball
    actions += [("move", "eye", (-0.05, -0.03))] * 14
    # Move the marker towards the bell
    actions += [("move", "marker", (-0.005, 0.05))] * 8
    # Throw the ball
    actions += [("interact", "ball")]
    # Run demo
    for action in actions:
        obs, _, done, _ = env.step(action)
        images.append(env.render())
    imageio.mimwrite("continuous_playroom_harcoded_actions.mp4", images)
    print("Wrote out to continuous_playroom_harcoded_actions.mp4.")

def demo_continuous_playroom_random_options(num_steps=50, render=False):
    rng = np.random.RandomState(0)
    playroom_options = sorted(create_playroom_options())
    env = ContinuousPlayroomEnv()
    obs = env.reset()
    if render:
        img = env.render()
        images = [img]
    current_option = None
    previous_obs = obs
    for _ in range(num_steps):
        if current_option is None:
            # Select a new option
            option_idxs = list(range(len(playroom_options)))
            rng.shuffle(option_idxs)
            for idx in option_idxs:
                option = playroom_options[idx]
                # Check if applicable
                applicable = option.is_applicable(obs)
                if applicable:
                    current_option = option
                    break
            if current_option is None:
                raise Exception("No options applied!")
        print(f"Executing option {current_option.name}")
        action = current_option.get_action(obs)
        obs, _, _, _ = env.step(action)
        if np.any((np.array(obs) != previous_obs) & (~current_option.mask)):
            assert current_option.mask[idx], \
                f"Mask incorrect for option {current_option.name}!"
        if current_option.is_terminal(obs):
            current_option = None
        previous_obs = obs
        if render:
            images.append(env.render())
    if render:
        imageio.mimwrite("continuous_playroom_random_options.mp4", images)
        print("Wrote out to continuous_playroom_random_options.mp4.")

def demo_option_plan(env, obs, goal, plan, render=False, outfile=None,
                     max_steps_per_option=100):
    if render:
        img = env.render()
        images = [img]
    for option in plan:
        if not option.is_applicable(obs):
            print("Tried to execute a not applicable option! Skipping.")
            continue
        for _ in range(max_steps_per_option):
            action = option.get_action(obs)
            obs, _, _, _ = env.step(action)
            if render:
                images.append(env.render())
            if option.is_terminal(obs):
                break
        else:
            print("Warning: option did not terminate within budget.")
    if render:
        imageio.mimwrite(outfile, images)
        print(f"Wrote out to {outfile}.")

def demo_continuous_playroom_skills_to_symbols(do_tests=False):
    options = create_playroom_options()
    env = ContinuousPlayroomEnv()
    feature_names = env.OBS_VARS
    options = partition_playroom_options(options)
    option_to_preconditions, option_to_effects = \
        create_playroom_pres_and_effs(options)
    if do_tests:
        test_preconditions_and_effects(options, env,
            option_to_preconditions, option_to_effects)
    operators, propositions = create_operators_from_options(
        options, option_to_preconditions, option_to_effects)
    domain_file = "playroom_operators.pddl"
    create_domain_file(operators, propositions, "playroom", domain_file)
    obs = env.reset()
    # Make the monkey cry
    goal = DecisionTreeClassifier(
        feature_names=feature_names,
        tuples=(False, (env.OBS_VAR_NAME_TO_IDX["monkey_cry"], ">=", 1e-6), True),
    )
    sym_initial_state = {p for p in propositions if p.classifier.predict(obs)}
    sym_goal = {p for p in propositions if goal.issubset(p.classifier)}
    problem_file = "playroom_problem.pddl"
    create_problem_file(sym_initial_state, sym_goal, "playroom", problem_file)
    plan = run_planner(domain_file, problem_file, options)
    print("Found plan:", [o.name for o in plan])
    # Run plan in env
    demo_option_plan(env, obs, goal, plan, render=True, outfile="playroom_result.mp4")



def test_preconditions_and_effects(options, env,
    option_to_preconditions, option_to_effects,
    num_steps=10000):
    """Take random options and always check that
    the preconditions and effects are anticipated
    """
    rng = np.random.RandomState(0)
    options = sorted(options)
    obs = env.reset()
    current_option = None
    for _ in range(num_steps):
        # Check all option preconditions
        for option in options:
            pre_tree = option_to_preconditions[option]
            if not (pre_tree.predict(obs) == \
                option.is_applicable(obs)):
                pre_tree.predict(obs, verbose=True)
                import ipdb; ipdb.set_trace()
        # Select a new option
        if current_option is None:
            option_idxs = list(range(len(options)))
            rng.shuffle(option_idxs)
            for idx in option_idxs:
                option = options[idx]
                # Check if applicable
                applicable = option.is_applicable(obs)
                if applicable:
                    current_option = option
                    break
            if current_option is None:
                raise Exception("No options applied!")
        print(f"Executing option {current_option.name}")
        action = current_option.get_action(obs)
        obs, _, _, _ = env.step(action)
        if current_option.is_terminal(obs):
            # Check effects
            eff_tree = option_to_effects[current_option]
            if not eff_tree.predict(obs):
                eff_tree.predict(obs, verbose=True)
                import ipdb; ipdb.set_trace()
            current_option = None

def test_domain_and_problem(domain_file, problem_file, options, env, obs, goal):
    plan = run_planner(domain_file, problem_file, options)
    print("Found plan:", [o.name for o in plan])
    # Run plan in env
    demo_option_plan(env, obs, goal, plan, render=True, outfile="playroom_result.mp4")


if __name__ == "__main__":
    # render_continuous_playroom_initial_state()
    # demo_continuous_playroom_random_actions()
    # demo_continuous_playroom_hardcoded_actions()
    # demo_continuous_playroom_random_options(num_steps=50000)
    # demo_continuous_playroom_random_options(num_steps=100, render=True)
    # demo_continuous_playroom_skills_to_symbols()
    domain_file = "playroom_operators.pddl"
    problem_file = "playroom_problem.pddl"
    env = ContinuousPlayroomEnv()
    feature_names = env.OBS_VARS
    options = create_playroom_options()
    obs = env.reset()
    # Make the monkey cry
    goal = DecisionTreeClassifier(
        feature_names=feature_names,
        tuples=(False, (env.OBS_VAR_NAME_TO_IDX["monkey_cry"], ">=", 1e-6), True),
    )
    test_domain_and_problem(domain_file, problem_file, options, env, obs, goal)    

