# Experimental Modular Meta-RL Algorithm
## Combining PEARL (Meta-RL) with BounceGRAD (End-to-end training of combinatorial modules)

Course Project for Stanford CS 332: Advanced Survey of Reinforcement Learning.

This project implements an experimental approach to modular meta-RL, aimed at improving the ability of context-based meta-policies to represent diverse, non-parametric behavior modes. The core idea is to implement internal layers in actor/critic networks as reconfigurable combinations of trainable modules (implemented as graph neural network layers). Unlike PEARL, which represents task encodings as continuous latent variables, this algorithm represents tasks as specific configurations of trainable modules, allowing discrete changes in computation from task to task, rather than continuous interpolation. Using the BounceGRAD algorithm, modules are shared among all tasks, and probabilistic task encoding (task adaptation) is performed through simulated annealing of the graph structure connecting the modules.

The full report pdf is [here](CS_332_Course_Project_Report.pdf), and the slides for the corresponding presentation can be found [here](https://docs.google.com/presentation/d/1Qn4wvB3S_tghZz6s3GFurqKo8_7zFuxLczsOHy0DSjc/edit?usp=sharing).

This repo is built on top of the original PEARL implementation, which contains further details about environment setup.

To train, run the command: `python launch_experiment.py config/graph-ant-dir.json` (or another config file beginning with `graph-`).
