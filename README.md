# Final Project Presentation


## My installation
- python=3.7.16
- pytorch=1.13.1
- pybullet=3.2.5
- stable-baselines3=1.3.0
- mujoco-py=2.1.2.14


## Usage

All parameters are stored in the [config.yaml](https://github.com/ShuyZzz/Project_code/blob/master/config.yaml), including environment, episodes, learning rate etc. All parameters can be modified directly in this file.

**1. Train the model and recode its success rate to tensorboard:**

```
python train.py --run_name Give_a_name --use_her --sparse_reward --soft_tau 0.0001
```
The default procedure uses dense rewards without Hindsight Experience Replay. The update rate of the target network is 0.001. You can train the model in different situation:
```
python train.py --run_name Give_a_name --use_her --sparse_reward --soft_tau 0.0001
```
`--use_her` means using HER, `--sparse_reward` means using sparse reward environment. `--soft_tau 0.0001` means the the update rate of the target network is 0.0001. Some pre-written training commands are recorded in [run.bat](https://github.com/ShuyZzz/Project_code/blob/master/run.bat). You can directly use them.
The training resualts are stored in [fetchreach_logs](https://github.com/ShuyZzz/Project_code/tree/master/fetchreach_logs), to see the recorded success rate, command line to start tensorboard:
```
tensorboard --logdir ./fetchreach_logs/
```

**2. Train the model and save it:**
```
python train_show.py --run_name Give_a_name
```

**3. See the pre-trained models:**
```
python Demo.py 
```
Four pre-trained models (reach/slide/push/pick_and_place)are saved in [pre_trained](). 

**4. Observe and record the velocity and position of the end-effector:**
```
python velocity.py --run_name Give_a_name
```
Observed position and velocity are stored in [fetchreach_velocity_logs](https://github.com/ShuyZzz/Project_code/tree/master/fetchreach_velocity), to see results, command line to start tensorboard:
```
tensorboard --logdir ./fetchreach_velocity_logs/
```

**5. Others**

[excels_data](https://github.com/ShuyZzz/Project_code/tree/master/excel_data) stored stores all success rate, position, and velocity data. You can plot them by running [plot.py](https://github.com/ShuyZzz/Project_code/blob/master/plot.py) and [plot_velocity.py](https://github.com/ShuyZzz/Project_code/blob/master/plot_velocity.py).

[SB3](https://github.com/ShuyZzz/Project_code/tree/master/SB3) has the training result of Stable Baseline 3.
