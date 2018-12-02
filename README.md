# `mazelab`: A customizable framework to create maze and gridworld environments.

This repository contains a customizable framework to create maze and gridworld environments with gym-like API. It has modular designs and it allows large flexibility for the users to easily define their own environments such as changing rendering colors, adding more objects, define available actions etc. The motivation of this
repository is, as maze or gridworld are used very often in the reinforcement learning community, however, 
it is still lack of a standardized framework. 

The repo will be actively maintained, any comments, feedbacks or improvements are highly welcomed. 

# Installation

## Install dependencies
Run the following command to install [all required dependencies](requirements.txt):

```bash
pip install -r requirements.txt
```

Note that it is highly recommanded to use an Miniconda environment:

```bash
conda create -n mazelab python=3.7
```

## Install mazelab
Run the following commands to install mazelab from source:

```bash
git clone https://github.com/zuoxingdong/mazelab.git
cd mazelab
pip install -e .
```

Installing from source allows to flexibly modify and adapt the code as you pleased, this is very convenient for research purpose which often needs fast prototyping.

# Getting started

Detailed tutorials is coming soon. For now, it is recommended to have a look in [examples/](examples) or the source code.

# Examples

We have provided a Jupyter Notebook for each example to illustrate how to make various of maze environments, and generate animation
of the agent's trajectory following the optimal actions solved by our build-in Dijkstra optimal planner. 

## [Simple empty maze](examples/simple_empty_maze)
![Simple empty maze](data/simple_empty_maze.gif)
## [Random shape maze](examples/random_shape_maze)
![Random shape maze](data/random_shape_maze.gif)
## [Random maze](examples/random_maze)
![Random maze](data/random_maze.gif)
## [U-maze](examples/u_maze)
![U-maze](data/u_maze.gif)
## [Double T-maze](examples/t_maze)
![Double T-maze](data/t_maze.gif)
## [Morris water maze](examples/morris_water_maze)
![Morris water maze](data/morris_water_maze.gif)

# How to create your own maze/gridworld environment

- **Define Generator**: You can define your own maze generator, simply by creating a class inherited from base class `BaseGenerator` 
and the class should look like at least: 

```python
    class Generator(BaseGenerator):
        def make_objects(self):
            ...

        def __call__(self):
            ...
```

- **Create maze object**: 

```python
    generator = Generator()
    maze = Maze(generator)
```

- **Define Motion**: define your own available actions

```python
    motion = Motion()
    motion.add('north', [-1, 0])
    motion.add('south', [1, 0])
    motion.add('west', [0, -1])
    motion.add('east', [0, 1])
```

- **Define Gym-like Environment**:

```python
    class Env(BaseNavigationEnv):
        def step(self, action):
            ...

        def reset(self):
            self.state = self.make_state()
            self.goal = self.make_goal()

            return self.get_observation()

        def make_state(self):
            ...

        def make_goal(self):
            ...

        def is_valid(self, position):
            ...

        def is_goal(self, position):
```

- **Create environment**:

```python
    env = Env(maze, motion)
```

- **Solve environment**:

```python
    actions = dijkstra_solver(np.array(env.maze.to_impassable()), env.motion, env.state.positions[0], env.goal.positions[0])
```

- **Record video of executing optimal actions**:

```python
    env = Monitor(env, directory='./', force=True)
    env.reset()
    for action in actions:
        env.step(action)
    env.close()
```

# Roadmap
- More extensive documentations
- More different kinds of mazes
- More color patterns

# Reference
Please use this bibtex if you want to cite this repository in your publications:
    
    @misc{mazelab,
          author = {Zuo, Xingdong},
          title = {mazelab: A customizable framework to create maze and gridworld environments.},
          year = {2018},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/zuoxingdong/mazelab}},
        }

