from gym_maze.envs import MazeEnv
from gym_maze.envs.Astar_solver import AstarSolver
from gym_maze.envs.generators import WaterMazeGenerator

maze = WaterMazeGenerator(7, 1, obstacle_ratio=0.25)
# env = MazeEnv(maze, action_type='VonNeumann', render_trace=True)


def solvemaze(maze, action_type='VonNeumann', render_trace=False, gif_file='video.gif'):
    env = MazeEnv(maze, action_type=action_type, live_display=True, render_trace=render_trace)
    s = env.reset()
    print(s.shape)

    # Solve maze by A* search from current state to goal
    solver = AstarSolver(env, env.goal_states[0])
    if not solver.solvable():
        print('unsolvable')
        while True:
            env.render()
        raise Error('The maze is not solvable given the current state and the goal state')
    ret = 0.
    for action in solver.get_actions():
        s_, r, _, _ = env.step(action)
        diff = s - s_
        s = s_
        ret += r
        fig = env.render()
    print('return:', ret)

    while True:
        env.reset()
        env.render()

    return env._get_video(interval=200, gif_path=gif_file).to_html5_video()


anim = solvemaze(maze, action_type='VonNeumann', render_trace=True, gif_file='data/tf.gif')