# ==============================================================================
# Vector Racer AI - Level 3.5 (Optimized for Speed + Interactive Reel)
# ==============================================================================
#
# KEY FIXES:
# 1. Truly Headless Training: The main training loop is now completely
#    separated from Pygame. It will run as fast as your CPU allows,
#    finishing in seconds.
# 2. Interactive UI after Training: The Pygame window and interactive buttons
#    will ONLY appear *after* all the training is complete.
# 3. Live Terminal Counter: You still get a live-updating episode counter in
#    the terminal so you can see the progress.
#
# ==============================================================================

import pygame
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from collections import deque
import math
import sys

# --- Imports for the AI Brain (PyTorch) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Display Constants ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# --- Colors ---
BLACK, WHITE, GREEN, RED, BLUE = (0,0,0), (255,255,255), (0,255,0), (255,0,0), (100,100,255)
GRAY, BUTTON_COLOR, YELLOW = (40,40,40), (100,100,100), (255,255,0)

# --- Car Physics Constants ---
MAX_SPEED, ACCELERATION, DECELERATION, TURN_SPEED = 8, 0.2, 0.1, 0.1

class Car:
    """ The Car class handles the game physics and state """
    def __init__(self, track_checkpoints):
        self.track = track_checkpoints
        self.reset()

    def reset(self):
        self.pos = np.array(self.track[0], dtype=float)
        self.speed = 0
        self.angle = 0
        self.current_checkpoint_index = 1
        return self.get_state()

    def get_state(self):
        next_checkpoint = self.track[self.current_checkpoint_index % len(self.track)]
        vec_to_checkpoint = np.array(next_checkpoint) - self.pos
        dist_to_checkpoint = np.linalg.norm(vec_to_checkpoint)
        vec_to_checkpoint /= dist_to_checkpoint if dist_to_checkpoint > 0 else 1
        return np.array([self.speed / MAX_SPEED, vec_to_checkpoint[0], vec_to_checkpoint[1], math.sin(self.angle), math.cos(self.angle)])

    def step(self, action):
        if action == 0: self.speed = min(MAX_SPEED, self.speed + ACCELERATION)
        if action == 2: self.angle -= TURN_SPEED
        if action == 3: self.angle += TURN_SPEED
        self.speed = max(0, self.speed - DECELERATION)
        self.pos[0] += self.speed * math.cos(self.angle)
        self.pos[1] += self.speed * math.sin(self.angle)
        
        reward, done = -0.1, False
        
        next_checkpoint = self.track[self.current_checkpoint_index % len(self.track)]
        if np.linalg.norm(np.array(next_checkpoint) - self.pos) < 50:
            reward = 20
            self.current_checkpoint_index += 1
            if self.current_checkpoint_index >= len(self.track):
                reward = 100
                done = True
        
        min_dist_to_track = min(np.linalg.norm(np.array(self.track[i]) - self.pos) for i in range(len(self.track)))
        if min_dist_to_track > 250:
            reward = -100
            done = True
            
        return self.get_state(), reward, done

class DQNAgent:
    """ The AI Agent using a Deep Q-Network """
    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.memory = deque(maxlen=100000)
        self.gamma, self.epsilon, self.epsilon_min, self.epsilon_decay = 0.99, 1.0, 0.01, 0.999
        self.learning_rate, self.batch_size = 0.001, 64
        self.policy_net, self.target_net = self.build_model(), self.build_model()
        self.update_target_net()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def build_model(self):
        return nn.Sequential(nn.Linear(self.state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, self.action_dim))

    def update_target_net(self): self.target_net.load_state_dict(self.policy_net.state_dict())
    def remember(self, state, action, reward, next_state, done): self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, is_training=True):
        if is_training and np.random.rand() <= self.epsilon: return random.randrange(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.policy_net(state_tensor).argmax().item()

    def learn(self):
        if len(self.memory) < self.batch_size: return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(np.array(states)); actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1); next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        loss = F.mse_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

def train_ai(agent, car, num_episodes):
    """ The new, purely headless training function """
    scores, highlights = [], []
    highlight_milestones = {100, 500, 1000, 2000}
    best_score = -float('inf')
    best_run_actions = []

    for episode in range(num_episodes):
        state, done, episode_score, step_count = car.reset(), False, 0, 0
        current_actions = []
        while not done and step_count < 1500:
            action = agent.choose_action(state)
            current_actions.append(action)
            next_state, reward, done = car.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state, episode_score, step_count = next_state, episode_score + reward, step_count + 1
        
        scores.append(episode_score)
        agent.update_target_net()
        # Live progress counter for the terminal
        sys.stdout.write(f"\rTraining... Episode {episode + 1}/{num_episodes}")
        sys.stdout.flush()

        if episode_score > best_score:
            best_score = episode_score
            best_run_actions = current_actions
        
        if (episode + 1) in highlight_milestones:
            highlights.append({'episode': episode + 1, 'score': round(episode_score, 2), 'actions': current_actions})
    
    highlights.append({'episode': 'Best Run', 'score': round(best_score, 2), 'actions': best_run_actions})
    return scores, highlights

def run_interactive_playback(car, track, highlights):
    """ The new, dedicated Pygame UI function """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Vector Racer AI - Highlight Reel")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    current_highlight_index = len(highlights) - 1 # Start on the best run
    buttons = { "prev": pygame.Rect(10, SCREEN_HEIGHT - 60, 100, 50), "replay": pygame.Rect(120, SCREEN_HEIGHT - 60, 120, 50), "next": pygame.Rect(250, SCREEN_HEIGHT - 60, 100, 50) }

    def draw_ui(highlight):
        draw_environment(screen, car, track)
        for name, rect in buttons.items():
            pygame.draw.rect(screen, BUTTON_COLOR, rect)
            text_surf = font.render(name.capitalize(), True, WHITE)
            screen.blit(text_surf, text_surf.get_rect(center=rect.center))
        info_text = f"Showing Highlight: Episode {highlight['episode']} (Score: {highlight['score']:.2f})"
        draw_text(screen, info_text, (10, 10), font=font, color=YELLOW)
        draw_text(screen, "Click 'Replay' to watch this run.", (10, 50), font=font, color=WHITE)
        pygame.display.flip()

    def replay_episode(actions):
        car.reset()
        for action in actions:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return False
            car.step(action)
            draw_environment(screen, car, track)
            pygame.display.flip()
            clock.tick(FPS)
        return True

    running = True
    while running:
        current_highlight = highlights[current_highlight_index]
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if buttons['prev'].collidepoint(event.pos): current_highlight_index = (current_highlight_index - 1 + len(highlights)) % len(highlights)
                elif buttons['next'].collidepoint(event.pos): current_highlight_index = (current_highlight_index + 1) % len(highlights)
                elif buttons['replay'].collidepoint(event.pos):
                    if not replay_episode(current_highlight['actions']): running = False
        
        car.reset() # Reset car to start for static display
        draw_ui(current_highlight)
        clock.tick(30)
    
    pygame.quit()

def draw_text(screen, text, pos, font, color):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, pos)

def draw_environment(screen, car, track):
    screen.fill(GRAY)
    pygame.draw.lines(screen, BLACK, True, track, 30)
    pygame.draw.lines(screen, WHITE, True, track, 24)
    for i, point in enumerate(track): pygame.draw.circle(screen, BLUE if i > 0 else GREEN, point, 15)
    
    car_surface = pygame.Surface((40, 20), pygame.SRCALPHA); car_surface.fill(RED)
    rotated_car = pygame.transform.rotate(car_surface, -math.degrees(car.angle))
    screen.blit(rotated_car, rotated_car.get_rect(center=car.pos).topleft)

if __name__ == '__main__':
    track = [(100, 400), (200, 200), (400, 150), (600, 200), (800, 300), (1000, 400), (1100, 600), (900, 700), (600, 650), (300, 600), (100, 400)]
    car = Car(track)
    agent = DQNAgent(state_dim=5, action_dim=4)

    # --- PHASE 1: HEADLESS TRAINING ---
    start_time = time.time()
    scores, highlights = train_ai(agent, car, num_episodes=2000)
    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds!")

    # --- PHASE 2: PLOT RESULTS ---
    plt.figure(figsize=(12, 6)); plt.plot(scores, label='Score per Episode')
    moving_avg = [np.mean(scores[max(0, i-100):i+1]) for i in range(len(scores))]
    plt.plot(moving_avg, color='red', linewidth=2, label='Moving Average (100 episodes)')
    plt.title('AI Racer Score Progression'); plt.xlabel('Episode'); plt.ylabel('Total Reward (Score)')
    plt.legend(); plt.grid(True); plt.savefig('racer_progress.png')
    print("Saved training progress graph to 'racer_progress.png'")

    # --- PHASE 3: INTERACTIVE PLAYBACK ---
    print("Launching interactive highlight reel...")
    run_interactive_playback(car, track, highlights)