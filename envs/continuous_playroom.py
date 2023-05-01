"""Continuous playroom domain (Singh et al., 2005; Konidaris & Barto, 2009a)
"""
from .utils import get_asset_path, draw_token, fig2data
from structs import Option, DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


class ContinuousPlayroomEnv:
    """Continuous playroom domain (Singh et al., 2005; Konidaris & Barto, 2009a)
    """
    # Order of the state variables
    STATE_VARS = ["hand", "marker", "eye", "red", "green", "light_switch",
                  "bell", "ball", "light", "music", "monkey_cry"]
    STATE_VAR_NAME_TO_IDX = {name : idx for idx, name in enumerate(STATE_VARS)}

    # Order of the observation variables
    OBS_VARS = [f"{effector}-{obj}.x" for effector in ["hand", "marker", "eye"]\
                for obj in ["red", "green", "light_switch", "bell", "ball"]] + \
               [f"{effector}-{obj}.y" for effector in ["hand", "marker", "eye"]\
                for obj in ["red", "green", "light_switch", "bell", "ball"]] + \
               ["music", "light", "monkey_cry"]
    OBS_VAR_NAME_TO_IDX = {name : idx for idx, name in enumerate(OBS_VARS)}

    # Icons for rendering
    HAND_ICON = plt.imread(get_asset_path("hand.png"))
    MARKER_ICON = plt.imread(get_asset_path("marker.png"))
    EYE_ICON = plt.imread(get_asset_path("eye.png"))
    RED_BUTTON_ICON = plt.imread(get_asset_path("red_button.png"))
    GREEN_BUTTON_ICON = plt.imread(get_asset_path("green_button.png"))
    LIGHT_SWITCH_ICON = plt.imread(get_asset_path("light_switch.png"))
    BELL_ICON = plt.imread(get_asset_path("bell.png"))
    MONKEY_ICON = plt.imread(get_asset_path("monkey.png"))
    BALL_ICON = plt.imread(get_asset_path("ball.png"))

    # Monkey position never changes
    MONKEY_POS = (0.1, 0.8)

    def __init__(self, seed=0):
        self._state = None # set in reset
        self._rng = np.random.RandomState(seed)

    def reset(self):
        """Randomly reset the state
        """
        # Randomly sample the positions of the objects and effectors
        hand, marker, eye, red, green, light_switch, bell, ball = \
           map(tuple, self._rng.uniform(size=(8, 2)))

        # The light is initially off
        light = 0.

        # The music volume is also initially off
        music = 0.

        # The monkey has not initially cried out
        monkey_cry = 0.

        # Set the state
        self._state = (hand, marker, eye, red, green, light_switch, \
                       bell, ball, light, music, monkey_cry)

        # Get the observation
        return self._get_observation()

    def _get_observation(self):
        """Extract features from the state

        From Konidaris: "We use 33 continuous features computed from the state
        variables, describing the x and y distance between each of the agent's
        effectors and each object instead of their absolute positions, plus
        the music, light, and monkey scream."
        """
        obs = tuple(0 for _ in self.OBS_VARS)

        # Relative positions
        for effector in ["hand", "marker", "eye"]:
            effector_pos = self._get_state_variable(effector)
            for obj in ["red", "green", "light_switch", "bell", "ball"]:
                object_pos = self._get_state_variable(obj)
                rel_x, rel_y = np.subtract(object_pos, effector_pos)
                obs = self._set_obs_variable(obs, f"{effector}-{obj}.x", rel_x)
                obs = self._set_obs_variable(obs, f"{effector}-{obj}.y", rel_y)

        # Music, light, monkey cry
        for attribute in ["music", "light", "monkey_cry"]:
            obs = self._set_obs_variable(obs, attribute, self._get_state_variable(attribute))

        assert len(obs) == 33

        return obs

    def step(self, action):
        """Update the state given a low-level action
        """
        if action[0] == "move":
            assert len(action) == 3
            done = self._step_move(action)
        elif action[0] == "interact":
            assert len(action) == 2
            done = self._step_interact(action)
        else:
            raise Exception(f"Unrecognized action {action}")
        # Include trivial reward, done, debug info for rough
        # compatibility with OpenAI gym API
        return self._get_observation(), 0., done, {}

    def _get_state_variable(self, var_name):
        """
        """
        idx = self.STATE_VAR_NAME_TO_IDX[var_name]
        return self._state[idx]

    def _set_state_variable(self, var_name, new_value):
        """
        """
        idx = self.STATE_VAR_NAME_TO_IDX[var_name]
        self._state = [s for s in self._state]
        self._state[idx] = new_value
        self._state = tuple(self._state)

    @classmethod
    def get_obs_variable(cls, obs, var_name):
        """
        """
        idx = cls.OBS_VAR_NAME_TO_IDX[var_name]
        return obs[idx]

    def _set_obs_variable(self, obs, var_name, new_value):
        """
        """
        idx = self.OBS_VAR_NAME_TO_IDX[var_name]
        obs = [s for s in obs]
        obs[idx] = new_value
        obs = tuple(obs)
        return obs

    def _step_move(self, action):
        """
        """
        assert action[0] == "move"
        assert action[1] in ["hand", "marker", "eye"]
        assert -0.05 <= action[2][0] <= 0.05
        assert -0.05 <= action[2][1] <= 0.05
        x, y = self._get_state_variable(action[1])
        dx, dy = action[2]
        x, y = np.clip((x+dx, y+dy), 0., 1.)
        self._set_state_variable(action[1], (x, y))
        # Moving the eye updates the light value when the light is on
        if action[1] == "eye" and self._get_state_variable("light") > 0:
            light_state = 1. - ((x - 0.5)**2 + (y - 0.5)**2)
            self._set_state_variable("light", light_state)
        return False

    def _step_interact(self, action):
        """
        """
        assert action[0] == "interact"
        assert action[1] in ["red", "green", "light_switch", "bell", "ball"]

        # Interacting with any object requires the hand and eye to be on it
        # Check if hand on object
        if not self._effector_on_object("hand", action[1]):
            return False
        # Check if eye on object
        if not self._effector_on_object("eye", action[1]):
            return False

        # Light switch
        if action[1] == "light_switch":
            # Toggle the light
            light_state = self._get_state_variable("light")
            if light_state == 0.:
                eye_x, eye_y = self._get_state_variable("eye")
                light_state = 1. - ((eye_x - 0.5)**2 + (eye_y - 0.5)**2)
            else:
                light_state = 0.
            self._set_state_variable("light", light_state)
            return False

        # Buttons
        if action[1] in ["red", "green"]:
            # Check if light on
            if self._get_state_variable("light") == 0:
                return False
            # Ok, we can interact!
            if action[1] == "red":
                # Turns the music off
                self._set_state_variable("music", 0)
                return False
            assert action[1] == "green"
            # Turns on music
            music_val = self._rng.uniform(0.3, 1.0)
            self._set_state_variable("music", music_val)
            return False

        # Ball (and indirectly, bell)
        if action[1] == "ball":
            # Check if marker on bell
            if not self._effector_on_object("marker", "bell"):
                return False
            # Throw the ball
            marker_pos = self._get_state_variable("marker")
            self._set_state_variable("ball", marker_pos)
            # Check if the monkey will cry
            light_is_on = self._get_state_variable("light") != 0
            music_is_on = self._get_state_variable("music") != 0
            monkey_cry = (not light_is_on) and music_is_on
            self._set_state_variable("monkey_cry", monkey_cry)
            return monkey_cry

        # Interacting with bell does nothing (?)
        assert action[1] == "bell" 
        return False

    def _effector_on_object(self, effector_name, object_name):
        """
        """
        eff_x, eff_y = self._get_state_variable(effector_name)
        obj_x, obj_y = self._get_state_variable(object_name)
        if abs(eff_x - obj_x) > 0.05 or abs(eff_y - obj_y) > 0.05:
            return False
        return True

    def render(self):
        """Return an image based on the state
        """
        # Unpack the state
        hand, marker, eye, red, green, light_switch, \
            bell, ball, light, music, monkey_cry = self._state

        # Set up the figure
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_facecolor((237/255, 237/255, 237/255))

        # Put unrendered state aspects in the title
        title = f"Light: {light:.2f} Music: {music:.2f} Monkey Cry: {monkey_cry:.0f}"
        plt.title(title, fontsize=18)

        # Draw the objects
        for icon, pos in [
            (self.RED_BUTTON_ICON, red),
            (self.GREEN_BUTTON_ICON, green),
            (self.LIGHT_SWITCH_ICON, light_switch),
            (self.BELL_ICON, bell),
            (self.BALL_ICON, ball),
            (self.MONKEY_ICON, self.MONKEY_POS),
            (self.HAND_ICON, hand),
            (self.MARKER_ICON, marker),
            (self.EYE_ICON, eye),
        ]:
            draw_token(icon, pos[0], pos[1], ax, zoom=0.25)

        ax.set_xlim((-0.1, 1.1))
        ax.set_ylim((-0.1, 1.1))
        plt.tight_layout()

        # Create the image
        im = fig2data(fig)
        plt.close(fig)

        return im

    def get_random_action(self):
        """Just for demonstrating the environment

        Note that random action != random option!
        """
        # Interact or move?
        if self._rng.uniform() < 0.5:
            # Move
            effector_idx = self._rng.choice(3)
            effector = ["hand", "marker", "eye"][effector_idx]
            # dx, dy
            dx, dy = self._rng.uniform(-0.05, 0.05, size=2)
            return ("move", effector, (dx, dy))
        # Interact
        obj_idx = self._rng.choice(5)
        obj = ["red", "green", "light_switch", "bell", "ball"][obj_idx]
        return ("interact", obj)


def create_playroom_options():
    """Options are for moving effectors to objects, and interacting
    """
    playroom_options = set()

    # Relative movement options
    for effector in ["hand", "marker", "eye"]:
        for obj in ["red", "green", "light_switch", "bell", "ball"]:
            option = create_move_option(effector, obj)
            playroom_options.add(option)

    # Interact options
    for obj in ["red", "green", "light_switch", "bell", "ball"]:
        option = create_interact_option(obj)
        playroom_options.add(option)

    return playroom_options

def create_move_option(effector, obj):
    # Create termination conditions
    def is_terminal(obs):
        rel_x = ContinuousPlayroomEnv.get_obs_variable(obs, f"{effector}-{obj}.x")
        rel_y = ContinuousPlayroomEnv.get_obs_variable(obs, f"{effector}-{obj}.y")
        return abs(rel_x) < 0.05 and abs(rel_y) < 0.05
    # Create preconditions
    def is_applicable(obs):
        return True
    # Create policy
    def get_action(obs):
        rel_x = ContinuousPlayroomEnv.get_obs_variable(obs, f"{effector}-{obj}.x")
        rel_y = ContinuousPlayroomEnv.get_obs_variable(obs, f"{effector}-{obj}.y")
        dx, dy = np.clip((rel_x, rel_y), -0.05, 0.05)
        return ("move", effector, (dx, dy))

    # Create mask: this option will effect all state variables with effector
    # involved. If the effector is the eye, it will also effect light.
    mask = np.zeros(len(ContinuousPlayroomEnv.OBS_VARS), dtype=bool)
    for o in ["red", "green", "light_switch", "bell", "ball"]:
        for xy in ["x", "y"]:
            name = f"{effector}-{o}.{xy}"
            idx = ContinuousPlayroomEnv.OBS_VAR_NAME_TO_IDX[name]
            mask[idx] = 1
    if effector == "eye":
        idx = ContinuousPlayroomEnv.OBS_VAR_NAME_TO_IDX["light"]
        mask[idx] = 1

    # Create option
    return Option(f"move-{effector}-{obj}", is_applicable, get_action,
                  is_terminal, mask)

def _effector_on_object(effector, obj, obs):
    """Helper for is_applicable
    """
    rel_x = ContinuousPlayroomEnv.get_obs_variable(obs, f"{effector}-{obj}.x")
    rel_y = ContinuousPlayroomEnv.get_obs_variable(obs, f"{effector}-{obj}.y")
    return abs(rel_x) < 0.05 and abs(rel_y) < 0.05

def create_interact_option(obj):
    # Create termination conditions
    def is_terminal(obs):
        # Always terminate right away
        return True

    # Create preconditions
    def is_applicable(obs):
        # Interacting with any object requires the hand and eye to be on it
        # Check if hand on object
        if not _effector_on_object("hand", obj, obs):
            return False
        # Check if eye on object
        if not _effector_on_object("eye", obj, obs):
            return False

        # Light switch
        if obj == "light_switch":
            return True

        # Buttons
        if obj in ["red", "green"]:
            # Check if light on
            light = ContinuousPlayroomEnv.get_obs_variable(obs, "light")
            if light == 0.:
                return False
            # Check if music is toggled
            music = ContinuousPlayroomEnv.get_obs_variable(obs, "music")
            if music > 0 and obj == "green":
                return False
            if music == 0 and obj == "red":
                return False
            # Ok, we can interact!
            return True

        # Ball (and indirectly, bell)
        if obj == "ball":
            # Check if marker on bell
            if not _effector_on_object("marker", "bell", obs):
                return False
            return True

        # Interacting with bell does nothing (?)
        assert obj == "bell" 
        return False

    # Create policy
    def get_action(obs):
        return ("interact", obj)

    # Create mask
    mask = np.zeros(len(ContinuousPlayroomEnv.OBS_VARS), dtype=bool)

    # Interacting with the light switch
    if obj == "light_switch":
        idx = ContinuousPlayroomEnv.OBS_VAR_NAME_TO_IDX["light"]
        mask[idx] = 1

    # Interacting with the buttons changes the music
    elif obj in ["red", "green"]:
        idx = ContinuousPlayroomEnv.OBS_VAR_NAME_TO_IDX["music"]
        mask[idx] = 1

    # Interacting with the ball changes the monkey cry
    elif obj == "ball":
        idx = ContinuousPlayroomEnv.OBS_VAR_NAME_TO_IDX["monkey_cry"]
        mask[idx] = 1

    # Create option
    return Option(f"interact-{obj}", is_applicable, get_action, is_terminal, mask)

def partition_playroom_options(options):
    """Split options so that they become abstract subgoal options.

    Do this manually. In the real pipeline, this would be automated
    and done approximately according to data.

    The only move options that need splitting are the ones that move
    the eye. They should be split according to whether the light
    is on or off.

    For the interact options...
        -The light switch options need to be split according to
         whether the light is on or off
        -The ball option needs to be split according to whether
         the light is off and the music is on, or not
        -The red, green, and bell options do not need to be split

    Returns
    -------
    abstract_subgoal_options : { Option }
    """
    abstract_subgoal_options = set()

    for option in options:
        if option.name.startswith("move-eye-"):
            # Create two new options
            def is_applicable_light_on(obs):
                return ContinuousPlayroomEnv.get_obs_variable(obs, "light") != 0.

            def is_applicable_light_off(obs):
                return ContinuousPlayroomEnv.get_obs_variable(obs, "light") == 0.

            # The mask should include the light if the light is on
            light_idx = ContinuousPlayroomEnv.OBS_VAR_NAME_TO_IDX["light"]
            light_on_mask = option.mask.copy()
            light_on_mask[light_idx] = 1
            # But not if the light is off
            light_off_mask = option.mask.copy()
            light_off_mask[light_idx] = 0

            # Create new options
            light_on_option = Option(f"{option.name}-light_on", is_applicable_light_on,
                                     option.get_action, option.is_terminal, light_on_mask)
            light_off_option = Option(f"{option.name}-light_off", is_applicable_light_off,
                                      option.get_action, option.is_terminal, light_off_mask)

            abstract_subgoal_options.add(light_on_option)
            abstract_subgoal_options.add(light_off_option)
        
        elif option.name == "interact-light_switch":
            # Create two new options
            def is_applicable_light_on(obs):
                # Interacting with any object requires the hand and eye to be on it
                # Check if hand on object
                if not _effector_on_object("hand", "light_switch", obs):
                    return False
                # Check if eye on object
                if not _effector_on_object("eye", "light_switch", obs):
                    return False
                return ContinuousPlayroomEnv.get_obs_variable(obs, "light") != 0.

            def is_applicable_light_off(obs):
                # Interacting with any object requires the hand and eye to be on it
                # Check if hand on object
                if not _effector_on_object("hand", "light_switch", obs):
                    return False
                # Check if eye on object
                if not _effector_on_object("eye", "light_switch", obs):
                    return False
                return ContinuousPlayroomEnv.get_obs_variable(obs, "light") == 0.

            # The termination conditions can still just be "always True"
            # and the masks don't change
            # Create new options
            light_on_option = Option(f"{option.name}-light_on", is_applicable_light_on,
                                     option.get_action, option.is_terminal, option.mask)
            light_off_option = Option(f"{option.name}-light_off", is_applicable_light_off,
                                      option.get_action, option.is_terminal, option.mask)

            abstract_subgoal_options.add(light_on_option)
            abstract_subgoal_options.add(light_off_option)

        elif option.name == "interact-ball":
            # Create two new options
            def is_applicable_monkey_will_cry(obs):
                # Interacting with any object requires the hand and eye to be on it
                # Check if hand on object
                if not _effector_on_object("hand", "ball", obs):
                    return False
                # Check if eye on object
                if not _effector_on_object("eye", "ball", obs):
                    return False
                # Check if marker on bell
                if not _effector_on_object("marker", "bell", obs):
                    return False
                # Monkey will cry if light is off and music is on
                light_is_on = ContinuousPlayroomEnv.get_obs_variable(obs, "light") != 0
                music_is_on = ContinuousPlayroomEnv.get_obs_variable(obs, "music") != 0
                return (not light_is_on) and music_is_on

            def is_applicable_monkey_wont_cry(obs):
                # Interacting with any object requires the hand and eye to be on it
                # Check if hand on object
                if not _effector_on_object("hand", "ball", obs):
                    return False
                # Check if eye on object
                if not _effector_on_object("eye", "ball", obs):
                    return False
                # Check if marker on bell
                if not _effector_on_object("marker", "bell", obs):
                    return False
                # Monkey won't cry if light is on and music is off
                light_is_on = ContinuousPlayroomEnv.get_obs_variable(obs, "light") != 0
                music_is_on = ContinuousPlayroomEnv.get_obs_variable(obs, "music") != 0
                return light_is_on or not music_is_on

            # The termination conditions can still just be "always True"
            # and the masks don't change
            # Create new options
            monkey_will_cry_option = Option(f"{option.name}-monkey_will_cry",
                is_applicable_monkey_will_cry, option.get_action, option.is_terminal,
                option.mask)
            monkey_wont_cry_option = Option(f"{option.name}-monkey_wont_cry",
                is_applicable_monkey_wont_cry, option.get_action, option.is_terminal,
                option.mask)

            abstract_subgoal_options.add(monkey_will_cry_option)
            abstract_subgoal_options.add(monkey_wont_cry_option)

        else:
            abstract_subgoal_options.add(option)

    return abstract_subgoal_options


def create_playroom_pres_and_effs(abstract_subgoal_options):
    """Create hardcoded preconditions and effects for playroom options.

    In the real pipeline, these would be learned from data.

    Parameters
    ----------
    abstract_subgoal_options : { Option }
        Options that have already been partitioned.

    Returns
    -------
    option_to_preconditions : { Option : DecisionTreeClassifier }
        The preconditions are a binary DT classifier, where
        True/1 means that the preconditions hold.
    option_to_effects : { Option : DecisionTreeClassifier }
        The effects are a binary DT classifier, where
        True/1 means that the preconditions hold.
    """
    option_to_preconditions = {}
    option_to_effects = {}

    # Convenient shorthand
    I = lambda name : ContinuousPlayroomEnv.OBS_VAR_NAME_TO_IDX[name]
    feature_names = ContinuousPlayroomEnv.OBS_VARS

    for option in abstract_subgoal_options:
        # Move options
        if option.name.startswith("move-hand-") or \
           option.name.startswith("move-marker-"):
            _, effector, obj = option.name.split("-")
            # Preconditions: true everywhere
            preconditions = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=True,
            )
            # Effects: relative pos to object is within 0.05.
            effects = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I(f"{effector}-{obj}.x"), "<", 0.05),
                          (False, (I(f"{effector}-{obj}.x"), ">=", -0.05),
                            (False, (I(f"{effector}-{obj}.y"), "<", 0.05),
                              (False, (I(f"{effector}-{obj}.y"), ">=", -0.05),
                                True,
                        )))))
            option_to_preconditions[option] = preconditions
            option_to_effects[option] = effects

        elif option.name.startswith("move-eye-"):
            _, effector, obj, light_status = option.name.split("-")
            if light_status == "light_on":
                # Preconditions: true if the light is on
                preconditions = DecisionTreeClassifier(
                    feature_names=feature_names,
                    tuples=(False, (I("light"), ">=", 1e-6), True),
                )
                # Effects: relative pos to object is within 0.05
                # and light is still on
                effects = DecisionTreeClassifier(
                    feature_names=feature_names,
                    tuples=(False, (I(f"{effector}-{obj}.x"), "<", 0.05),
                              (False, (I(f"{effector}-{obj}.x"), ">=", -0.05),
                                (False, (I(f"{effector}-{obj}.y"), "<", 0.05),
                                  (False, (I(f"{effector}-{obj}.y"), ">=", -0.05),
                                    (False, (I("light"), ">=", 1e-6),
                                      True,
                            ))))))
            else:
                assert light_status == "light_off"
                # Preconditions: true if the light is off
                preconditions = DecisionTreeClassifier(
                    feature_names=feature_names,
                    tuples=(False, (I("light"), "<", 1e-6), True),
                )
                # Effects: relative pos to object is within 0.05
                effects = DecisionTreeClassifier(
                    feature_names=feature_names,
                    tuples=(False, (I(f"{effector}-{obj}.x"), "<", 0.05),
                              (False, (I(f"{effector}-{obj}.x"), ">=", -0.05),
                                (False, (I(f"{effector}-{obj}.y"), "<", 0.05),
                                  (False, (I(f"{effector}-{obj}.y"), ">=", -0.05),
                                    True,
                            )))))
            option_to_preconditions[option] = preconditions
            option_to_effects[option] = effects

        # Interact options
        elif option.name == "interact-light_switch-light_on":
            # Preconditions: hand and eye are at light,
            # and light is on.
            preconditions = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("hand-light_switch.x"), "<", 0.05),
                          (False, (I("hand-light_switch.x"), ">=", -0.05),
                            (False, (I("hand-light_switch.y"), "<", 0.05),
                              (False, (I("hand-light_switch.y"), ">=", -0.05),
                                (False, (I("eye-light_switch.x"), "<", 0.05),
                                  (False, (I("eye-light_switch.x"), ">=", -0.05),
                                    (False, (I("eye-light_switch.y"), "<", 0.05),
                                      (False, (I("eye-light_switch.y"), ">=", -0.05),
                                        (False, (I("light"), ">=", 1e-6),
                                          True,
                        ))))))))))
            # Effects: light is off
            effects = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("light"), "<", 1e-6),
                         True,
                       ))
            option_to_preconditions[option] = preconditions
            option_to_effects[option] = effects

        elif option.name == "interact-light_switch-light_off":
            # Preconditions: hand and eye are at light,
            # and light is off.
            preconditions = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("hand-light_switch.x"), "<", 0.05),
                          (False, (I("hand-light_switch.x"), ">=", -0.05),
                            (False, (I("hand-light_switch.y"), "<", 0.05),
                              (False, (I("hand-light_switch.y"), ">=", -0.05),
                                (False, (I("eye-light_switch.x"), "<", 0.05),
                                  (False, (I("eye-light_switch.x"), ">=", -0.05),
                                    (False, (I("eye-light_switch.y"), "<", 0.05),
                                      (False, (I("eye-light_switch.y"), ">=", -0.05),
                                        (False, (I("light"), "<", 1e-6),
                                          True,
                        ))))))))))
            # Effects: light is on
            effects = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("light"), ">=", 1e-6),
                         True,
                       ))
            option_to_preconditions[option] = preconditions
            option_to_effects[option] = effects

        elif option.name == "interact-red":
            # Preconditions: hand and eye are at red, light is on, music is on
            preconditions = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("hand-red.x"), "<", 0.05),
                          (False, (I("hand-red.x"), ">=", -0.05),
                            (False, (I("hand-red.y"), "<", 0.05),
                              (False, (I("hand-red.y"), ">=", -0.05),
                                (False, (I("eye-red.x"), "<", 0.05),
                                  (False, (I("eye-red.x"), ">=", -0.05),
                                    (False, (I("eye-red.y"), "<", 0.05),
                                      (False, (I("eye-red.y"), ">=", -0.05),
                                        (False, (I("light"), ">=", 1e-6),
                                          (False, (I("music"), ">=", 0.3),
                                            True,
                        )))))))))))

            # Effects: music is off
            effects = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("music"), "<", 0.3), True))
            option_to_preconditions[option] = preconditions
            option_to_effects[option] = effects

        elif option.name == "interact-green":
            # Preconditions: hand and eye are at green, and light is on, music is off
            preconditions = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("hand-green.x"), "<", 0.05),
                          (False, (I("hand-green.x"), ">=", -0.05),
                            (False, (I("hand-green.y"), "<", 0.05),
                              (False, (I("hand-green.y"), ">=", -0.05),
                                (False, (I("eye-green.x"), "<", 0.05),
                                  (False, (I("eye-green.x"), ">=", -0.05),
                                    (False, (I("eye-green.y"), "<", 0.05),
                                      (False, (I("eye-green.y"), ">=", -0.05),
                                        (False, (I("light"), ">=", 1e-6),
                                          (False, (I("music"), "<", 0.3),
                                            True,
                        )))))))))))

            # Effects: music is on
            effects = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("music"), ">=", 0.3), True))
            option_to_preconditions[option] = preconditions
            option_to_effects[option] = effects

        elif option.name == "interact-ball-monkey_will_cry":
            # Preconditions: hand and eye are at ball,
            # marker is at bell, music is on, light is off
            preconditions = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("hand-ball.x"), "<", 0.05),
                          (False, (I("hand-ball.x"), ">=", -0.05),
                            (False, (I("hand-ball.y"), "<", 0.05),
                              (False, (I("hand-ball.y"), ">=", -0.05),
                                (False, (I("eye-ball.x"), "<", 0.05),
                                  (False, (I("eye-ball.x"), ">=", -0.05),
                                    (False, (I("eye-ball.y"), "<", 0.05),
                                      (False, (I("eye-ball.y"), ">=", -0.05),
                                        (False, (I("marker-bell.x"), "<", 0.05),
                                          (False, (I("marker-bell.x"), ">=", -0.05),
                                            (False, (I("marker-bell.y"), "<", 0.05),
                                              (False, (I("marker-bell.y"), ">=", -0.05),
                                                (False, (I("light"), "<", 1e-6),
                                                  (False, (I("music"), ">=", 0.3),
                                                    True,
                        )))))))))))))))

            # Effects: monkey crying
            effects = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("monkey_cry"), ">=", 1e-6), True))

            option_to_preconditions[option] = preconditions
            option_to_effects[option] = effects

        elif option.name == "interact-ball-monkey_wont_cry":
            # Preconditions: hand and eye are at ball,
            # marker is at bell, music is off, OR light is ON
            preconditions = DecisionTreeClassifier(
                feature_names=feature_names,
                tuples=(False, (I("hand-ball.x"), "<", 0.05),
                          (False, (I("hand-ball.x"), ">=", -0.05),
                            (False, (I("hand-ball.y"), "<", 0.05),
                              (False, (I("hand-ball.y"), ">=", -0.05),
                                (False, (I("eye-ball.x"), "<", 0.05),
                                  (False, (I("eye-ball.x"), ">=", -0.05),
                                    (False, (I("eye-ball.y"), "<", 0.05),
                                      (False, (I("eye-ball.y"), ">=", -0.05),
                                        (False, (I("marker-bell.x"), "<", 0.05),
                                          (False, (I("marker-bell.x"), ">=", -0.05),
                                            (False, (I("marker-bell.y"), "<", 0.05),
                                              (False, (I("marker-bell.y"), ">=", -0.05),
                                                ((False, (I("music"), "<", 0.3), True),
                                                    (I("light"), ">=", 1e-6),
                                                    True,
                        ))))))))))))))

            # Effects: none
            effects = DecisionTreeClassifier(True)

            option_to_preconditions[option] = preconditions
            option_to_effects[option] = effects

        elif option.name == "interact-bell":
            # Preconditions: none
            preconditions = DecisionTreeClassifier(False)

            # Effects: none
            effects = DecisionTreeClassifier(True)
            option_to_preconditions[option] = preconditions
            option_to_effects[option] = effects

        else:
            raise Exception(f"Unrecognized option {option.name}.")

    assert set(option_to_preconditions) == abstract_subgoal_options
    assert set(option_to_effects) == abstract_subgoal_options
    
    return option_to_preconditions, option_to_effects




