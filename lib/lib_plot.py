
"""
plotting saved instances:
    python lib/lib_plot.py
        -f --file : if the saved DIRECTORY is data/random_20230822124105, only input 'random_20230822124105'

sample run:
    python lib/lib_plot.py -f 'random_20230822124021'
"""

import os
import time

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Plotting Graph (Realtime Update)

usage:
    initialization: 
        plot.init_plot('qtrain')
    update visuals: 
        plot.update_table(REWARD_TABLE, {'final_reward': total_reward}, epoch)
        :: REWARD_TABLE: legacy input
        :: {'final_reward': total_reward}: dictionary containing entries
        :: epoch: the x coordinate of the plot
"""
global subdir, summary_dir, summary_writer

ax = plt.axes()
df = pd.DataFrame()
is_main = False


def init_plot(prefix: str):

    global subdir, summary_dir, df
    now = time.localtime()
    subdir = time.strftime(f'{prefix}_%Y%m%d%H%M%S', now)
    subdir = os.path.join('data', subdir)

    summary_dir = os.path.join(subdir, 'logs.csv')
    os.makedirs(subdir)
    """
    global summary_writer
    summary_writer = tf.summary.create_file_writer(summary_dir)
    """
    df = pd.DataFrame()
    """
    t = threading.Thread(target=launch_plot, args=([]))
    t.start()
    time.sleep(3)
    """
    plt.tight_layout()
    plt.ion()


STEP_COLUMN = 'step'


def animate(i):
    plt.cla()

    global df

    if is_main:
        df = pd.read_csv(summary_dir)
    df2 = df.rolling(max(1, df.size//100), min_periods=1).mean()
    for key in df.columns:
        if key == STEP_COLUMN:
            continue
        ax.plot(df[STEP_COLUMN], df[key], alpha=0.3)
        ax.plot(df2[STEP_COLUMN], df2[key], label=key)

    plt.legend(loc='upper left')


last_update = time.time()
TIME_THRESHOLD = 0.5


def update_table(table_name: str, vals: dict[str, float], step: int):
    """
    with summary_writer.as_default():
        for key, val in vals.items():
            tf.summary.scalar(name=f'{table_name}/{key}', data=val, step=step)
    summary_writer.flush()
    """
    global df, last_update
    vals.update({STEP_COLUMN: step})
    df2 = pd.DataFrame([vals])
    df = pd.concat([df, df2], ignore_index=True)
    df2.to_csv(summary_dir, mode='a', header=not os.path.exists(summary_dir), index=False)

    now = time.time()
    if now-last_update > TIME_THRESHOLD:
        animate(step)
        plt.draw()
        plt.pause(0.01)
        last_update = now


def write(file:str, contents: str):
    """
    Outputting text to file.

    usage:
        model.summary(print_fn=lambda contents: plot.write('model_structure.txt', contents))
            OR
        plot.write('model_structure.txt', contents)

    The file will be written under the {subdir} directory.
    """
    with open(os.path.join(subdir, file), 'a+') as f:
        f.write(contents + '\n')


"""
Graph Plotting (Main Thread)
"""
if __name__ == '__main__':
    # not working
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='File to graph')
    args = parser.parse_args()

    subdir = args.file
    subdir = os.path.join('data', subdir)
    summary_dir = os.path.join(subdir, 'logs.csv')
    print(summary_dir)

    is_main = True

    ani = FuncAnimation(plt.gcf(), animate, interval=100, save_count=1)

    plt.tight_layout()
    plt.show()