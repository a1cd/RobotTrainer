# render box using arcade
import os
import random
from typing import Optional, Union, Tuple

import arcade
import numpy
import numpy as np
from arcade.pymunk_physics_engine import PymunkPhysicsEngine
from numpy import ndarray

import RL


class PhysicsObject:
    def __init__(self, mass):
        self.mass = mass
        self.sprite = None
        self.body = None
        self.shape = None

    def add_to_engine(self, engine):
        engine.add_sprite(self.sprite, friction=0.5, mass=self.mass)

    def update(self):
        pass

    def draw(self):
        self.sprite.draw()


class Robot(PhysicsObject):
    def __init__(self, mass, player = False):
        super().__init__(mass)

        self.sprite = arcade.sprite.SpriteSolidColor(50, 50, random.choice([arcade.color.RED, arcade.color.GREEN, arcade.color.BLUE,
                                                                            arcade.color.YELLOW, arcade.color.ORANGE, arcade.color.PURPLE]))
        self.sprite.center_x = 100
        self.sprite.center_y = 100
        self.sprite.angle = 0

        self.forward_force = 0
        self.strafe_force = 0
        self.rotation_force = 0

        self.player = player

    def add_to_engine(self, engine):
        super().add_to_engine(engine)
        # set the robot's initial velocity
        physics_object = engine.get_physics_object(self.sprite)
        physics_object.body.velocity = (0, 0)

    def update(self, delta_time=0, physics_engine: PymunkPhysicsEngine = None):
        # apply forces

        physics_object = physics_engine.get_physics_object(self.sprite)

        force = (self.forward_force, self.strafe_force)
        print(force)
        physics_object.body.apply_force_at_local_point(force, (0, 0))
        print(physics_object.body.force)

        # apply torque
        physics_object.body.torque = self.rotation_force

        # update sprite
        self.sprite.center_x = physics_object.body.position.x
        self.sprite.center_y = physics_object.body.position.y
        self.sprite.angle = numpy.rad2deg(physics_object.body.angle)

        print(physics_object.body.velocity)

        physics_engine.step(delta_time)



    def draw(self):
        self.sprite.draw()
        # draw a line to show the direction of the robot
        arcade.draw_line(
            self.sprite.center_x,
            self.sprite.center_y,
            self.sprite.center_x + 50 * numpy.cos(numpy.deg2rad(self.sprite.angle)),
            self.sprite.center_y + 50 * numpy.sin(numpy.deg2rad(self.sprite.angle)),
            arcade.color.BLACK, 2
        )

    def on_key_press(self, key, modifiers):
        if self.player:
            forcePower = 100
        else:
            forcePower = 0
        if key == arcade.key.W:
            self.forward_force = forcePower
        elif key == arcade.key.S:
            self.forward_force = -forcePower
        elif key == arcade.key.A:
            self.strafe_force = -forcePower
        elif key == arcade.key.D:
            self.strafe_force = forcePower
        elif key == arcade.key.Q:
            self.rotation_force = forcePower
        elif key == arcade.key.E:
            self.rotation_force = -forcePower

    def on_key_release(self, key, modifiers):
        if key == arcade.key.W:
            self.forward_force = 0
        elif key == arcade.key.S:
            self.forward_force = 0
        elif key == arcade.key.A:
            self.strafe_force = 0
        elif key == arcade.key.D:
            self.strafe_force = 0
        elif key == arcade.key.Q:
            self.rotation_force = 0
        elif key == arcade.key.E:
            self.rotation_force = 0


class Box(PhysicsObject):
    def __init__(self, mass):
        super().__init__(mass)

        self.sprite = arcade.sprite.SpriteSolidColor(50, 50, arcade.color.BLUE)
        self.sprite.center_x = 300
        self.sprite.center_y = 300
        self.sprite.angle = 0


class Game(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        # set the background color
        self.ai: RL.DQNAgent = None
        arcade.set_background_color(arcade.color.WHITE)

        self.robot: Optional[Robot] = None
        self.robots: [Robot] = []
        self.box: Optional[Box] = None

        self.physics_engine: Optional[PymunkPhysicsEngine] = None

        # robot inputs
        self.forward_force = 0
        self.strafe_force = 0
        self.rotation_force = 0

    def setup(self):
        # create the physics engine
        self.physics_engine = PymunkPhysicsEngine(damping=0.5, gravity=(0, 0))
        # create the ai
        self.ai = RL.DQNAgent(3, 3)
        if os.path.exists("./models/robot_model.h5"):
            self.ai.load("./models/robot_model.h5")

        # create the robot
        self.robot = Robot(1, True)
        self.robots.append(self.robot)
        for i in range(5):
            self.robots.append(Robot(1))
        for robot in self.robots:
            robot.add_to_engine(self.physics_engine)

        # randomize the robots' positions
        for robot in self.robots:
            physics_object = self.physics_engine.get_physics_object(robot.sprite)
            physics_object.body.position = (random.randint(0, 600), random.randint(0, 600))
            physics_object.body.angle = random.randint(0, 360)
        # create the box
        self.box = Box(1)
        self.box.add_to_engine(self.physics_engine)

    def reset(self):
        """Reset the environment to the start state"""
        # reset the physics engine
        self.physics_engine.reset()
        # reset the robots
        for robot in self.robots:
            # randomize the robots' positions
            physics_object = self.physics_engine.get_physics_object(robot.sprite)
            physics_object.body.position = (random.randint(0, 600), random.randint(0, 600))
            physics_object.body.angle = random.randint(0, 360)
        # reset the box
        physics_object = self.physics_engine.get_physics_object(self.box.sprite)
        physics_object.body.position = (random.randint(0, 600), random.randint(0, 600))
        physics_object.body.angle = random.randint(0, 360)


    def on_draw(self):
        arcade.start_render()
        for robot in self.robots:
            robot.draw()
        self.box.draw()

    def on_update(self, delta_time):
        delta_time = min(delta_time, 1 / 5)

        # Define the goal position as a fixed location in the game world
        goal_position = (100, 100)

        aiRobot: Robot = self.robots[0]
        # take an action
        state = self.get_state()
        action = self.ai.act(state)
        aiRobot.forward_force = action[0] * 100
        aiRobot.strafe_force = action[1] * 100
        aiRobot.rotation_force = action[2] * 360 * 2

        # update the physics engine
        for robot in self.robots:
            robot.update(delta_time, self.physics_engine)
        self.box.update()
        next_state, reward, done = self.get_reward(goal_position)
        self.ai.remember(state, action, reward, next_state, done)
        self.ai.replay(10)

        # save the model to a file
        if done:
            # create the models folder if it doesn't exist
            if not os.path.exists("./models"):
                os.makedirs("./models")
            self.ai.save("./models/robot_model.h5")
        else:
            if not os.path.exists("./models"):
                os.makedirs("./models")
            self.ai.save("./models/robot_model_last.h5")

    def get_reward(self, goal_position):
        # Compute the Euclidean distance between the current position of the agent and the goal position
        agent_position = self.robots[0].sprite.center_x, self.robots[0].sprite.center_y
        distance_to_goal = np.linalg.norm(np.array(agent_position) - np.array(goal_position))

        # Define the reward function
        if distance_to_goal < 10:
            # If the agent reaches the goal position, give it a positive reward
            reward = 100
            done = True
            # Reset the environment
            self.setup()
        else:
            # If the agent moves closer to the goal position, give it a small positive reward
            # If the agent moves farther away from the goal position, give it a negative reward
            reward = -distance_to_goal
            done = False
        return self.get_state(), reward, done

    def get_state(self):
        state = []
        ai: Robot = self.robots[0]
        # add AI position
        state.extend([ai.sprite.center_x / self.width, ai.sprite.center_y / self.height, (ai.sprite.angle % 360.0) / 360])
        # add box position
        # state.extend([self.box.sprite.center_x / self.width, self.box.sprite.center_y / self.height])
        # add other robots position
        # for robot in self.robots[1:]:
        #     state.extend([robot.sprite.center_x / self.width, robot.sprite.center_y / self.height])
        return np.array([state])

    def on_key_press(self, key, modifiers):
        for robot in self.robots:
            robot.on_key_press(key, modifiers)
        # self.box.on_key_press(key, modifiers)

    def on_key_release(self, key, modifiers):
        for robot in self.robots:
            robot.on_key_release(key, modifiers)
        # self.box.on_key_release(key, modifiers)


# main function
def main():
    window = Game(800, 600, "Robot")
    window.setup()
    arcade.run()

if __name__ == "__main__":
    main()
