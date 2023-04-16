import gymnasium
from nasim.envs.environment import NASimEnv
from nasim.envs.observation import Observation
from nasim.scenarios import Scenario, make_benchmark_scenario


class NASimGymEnv(NASimEnv):
    """A wrapper around the NASimEnv compatible with gymnasium.make()

    See nasim.NASimEnv for details.
    """

    def __init__(self,
                 scenario,
                 fully_obs=False,
                 flat_actions=True,
                 flat_obs=True,
                 render_mode=None,
                 seed=None):
        """
        Parameters
        ----------
        scenario : str or or nasim.scenarios.Scenario
            either the name of benchmark environment (str) or a nasim Scenario
            instance
        fully_obs : bool, optional
            the observability mode of environment, if True then uses fully
            observable mode, otherwise partially observable (default=False)
        flat_actions : bool, optional
            if true then uses a flat action space, otherwise will use
            parameterised action space (default=True).
        flat_obs : bool, optional
            if true then uses a 1D observation space. If False
            will use a 2D observation space (default=True)
        render_mode : str, optional
            The render mode to use for the environment.
        """
        if not isinstance(scenario, Scenario):
            scenario = make_benchmark_scenario(scenario, seed)
        super().__init__(scenario,
                         fully_obs=fully_obs,
                         flat_actions=flat_actions,
                         flat_obs=flat_obs,
                         render_mode=render_mode)

        if self.flat_obs:
            obs_shape = self.current_state.shape_useful_flat()
        else:
            obs_shape = self.last_obs.shape()
        obs_low, obs_high = Observation.get_space_bounds(self.scenario)
        self.observation_space = gymnasium.spaces.Box(
            low=obs_low, high=obs_high, shape=obs_shape
        )

    def reset(self, *, seed=None, options=None):
        """Reset the state of the environment and returns the initial state.

        Implements gymnasium.Env.reset().

        Parameters
        ----------
        seed : int, optional
            the optional seed for the environments RNG
        options : dict, optional
            optional environment options (does nothing in NASim at the moment)

        Returns
        -------
        numpy.Array
            the initial observation of the environment
        dict
            auxiliary information regarding reset
        """
        gymnasium.Env.reset(self, seed=seed, options=options)
        self.steps = 0
        self.current_state = self.network.reset(self.current_state)
        self.last_obs = self.current_state.get_initial_observation(
            self.fully_obs
        )

        if self.flat_obs:
            obs = self.current_state.useful_numpy_flat()
        else:
            obs = self.last_obs.numpy()

        return obs, {}

    def step(self, action):
        """Step the environment by one timestep.

        NOTE: 15/04/2023 Rubén: Fix to remove irrelevant and noisy information
        from the observation space.

        Parameters
        ----------
        action : Action or int or list or NumpyArray
            Action to perform. If not Action object, then if using
            flat actions this should be an int and if using non-flat actions
            this should be an indexable array.

        Returns
        -------
        numpy.Array
            observation from performing action
        float
            reward from performing action
        bool
            whether the episode reached a terminal state or not (i.e. all
            target machines have been successfully compromised)
        bool
            whether the episode has reached the step limit (if one exists)
        dict
            auxiliary information regarding step
            (see :func:`nasim.env.action.ActionResult.info`)
        """
        next_state, obs, reward, done, info = self.generative_step(
            self.current_state,
            action
        )
        self.current_state = next_state
        self.last_obs = obs

        if self.flat_obs:
            obs = next_state.useful_numpy_flat()
        else:
            obs = obs.numpy()

        self.steps += 1

        step_limit_reached = (
            self.scenario.step_limit is not None
            and self.steps >= self.scenario.step_limit
        )

        return obs, reward, done, step_limit_reached, info
