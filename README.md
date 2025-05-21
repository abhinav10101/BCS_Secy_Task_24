#BCS_Secy_Task

When I first started working on this project, my approach was to build things up step by step.

I began with a really simple environment—a square grid that just had an agent and a target. No maze, no obstacles. My goal was to get comfortable with the basics: how to design a reward system, how to tune hyperparameters, and how to render the environment using pygame. Once I got that working, I moved on to a rectangular grid, still without any maze structure. These two simpler environments helped me build confidence and get a feel for how everything fits together.

The next challenge was adding a maze. My idea was to prevent the agent from moving into walls by making a list of all the blocked cells and modifying the step function. If the agent tried to move into a wall, it just wouldn’t move. That worked pretty well.

Then I introduced a Death Eater into the environment. At first, performance was terrible—the agent reached the target only around 15–20% of the time. So I started digging into the code to figure out why. The biggest issues were:

-> The agent would try to move into walls and waste turns instead of picking another direction.

-> The state only included the agent’s location, with no information about the Death Eater or the target.

To fix this, I updated the state to include the positions of the agent, the Death Eater, and the target. I also changed the behavior so that if an action would move the agent into a wall, it would instead choose the action that moves it furthest away from the Death Eater.

After that, I turned my attention to the reward system and hyperparameters. Early on, I had read about sparse and dense rewards. I had been using a sparse rewards system, only giving rewards for reaching the target or penalties for taking too long or getting caught. I switched to a more dense reward system where the agent is:

-> Rewarded for reaching the target and for moving closer to it (based on Manhattan distance).

-> Penalized for moving toward the Death Eater or getting caught.

I also made sure that the rewards for moving toward the target had more weight than those for avoiding the Death Eater,since reaching the target is the priority.

For tuning hyperparameters, I created a list of different epsilon values, learning rates, and discount factors. I ran training with all the combinations and looked at which ones led to the best performance.

Final Parameters 
-> Epsilon: 0.1

-> Learning rate: 0.2

-> Discount factor: 0.96

During training, I started with a high epsilon (0.9) and used an epsilon decay function. That way, the agent explored a lot in the beginning, then gradually reduced it and at the end used the Q table 90% of the time.

The file GobletOfFire_train.py provides the Q table and stores it in the file Q_table.npy.

Now to obtain the plot for success rate for every 100 episodes over the course of 10,000 episodes you can run the file GobletOfFire_Success_Rate.py

To obtain the plot for rewards per episode over the course of 100 episodes you can run the file GobletOfFir_Rewards.py

The file GobletOfFire_test.py tests the code for one episode and provides a pygame rendering of it.

To find the number of generations before Harry escapes the death eater consecutively 10 times you can run the GobletOfFire_Gen.py .When I ran it it took 5275 generations.
