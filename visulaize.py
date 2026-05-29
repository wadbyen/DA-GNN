import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List

class MetricsVisualizer:
    """Generate separate PNG files for each training metric, including LR analysis."""

    def __init__(self, output_dir: str = "training_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_reward(self, rewards: List[float], window: int = 20):
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, alpha=0.6, label='Episode Reward')
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), ma, 'r-', label=f'Moving Avg ({window})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'reward.png'), dpi=150)
        plt.close()

    def plot_avg_reward(self, rewards: List[float], window: int = 50):
        avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        plt.figure(figsize=(10, 6))
        plt.plot(avg, 'g-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel(f'Average Reward (last {window} eps)')
        plt.title('Average Reward')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'avg_reward.png'), dpi=150)
        plt.close()

    def plot_delay(self, delays: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(delays, color='orange', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Delay (ms)')
        plt.title('Average End-to-End Delay')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'delay.png'), dpi=150)
        plt.close()
    
    def plot_packet_loss(self, losses: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(losses, color='red', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Packet Loss')
        plt.title('Average Packet Loss')
    
        # Only use log scale if all values are positive
        if all(l > 0 for l in losses):
            plt.yscale('log')
        else:
            # Optionally add a small epsilon to zeros for log scale
            # But linear scale is fine for mostly zero loss
            pass
    
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'packet_loss.png'), dpi=150)
        plt.close()
    
    def plot_learning_loss(self, learning_losses: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(learning_losses, color='brown', linewidth=1)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('DQN Learning Loss')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'learning_loss.png'), dpi=150)
        plt.close()

    def plot_exploration_rate(self, epsilons: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, color='purple', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'exploration_rate.png'), dpi=150)
        plt.close()

    def plot_completion_rate(self, completion_rates: List[float]):
        plt.figure(figsize=(10, 6))
        plt.plot(completion_rates, color='green', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Completion Rate')
        plt.title('Flow Completion Rate (Success)')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'completion_rate.png'), dpi=150)
        plt.close()

    def plot_lr_vs_reward(self, learning_rates: List[float], rewards: List[float]):
        """Plot learning rate over episodes and its correlation with episode reward."""
        if not learning_rates or not rewards:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Subplot 1: Learning rate over episodes
        ax1.plot(learning_rates, color='blue', linewidth=1.5)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Learning Rate')
        ax1.set_title('Learning Rate Schedule')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Scatter of LR vs Episode Reward
        episodes = np.arange(len(rewards))
        sc = ax2.scatter(learning_rates, rewards, c=episodes, cmap='viridis', alpha=0.6, edgecolors='k', linewidth=0.5)
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Episode Reward')
        ax2.set_title('Learning Rate vs. Episode Reward')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label('Episode Index')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'lr_vs_reward.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def plot_lr_comparison(self, lr_data: Dict[str, List[float]], window: int = 10):
        """
        Compare average reward across multiple learning rates.
        """
        if not lr_data:
            return

        plt.figure(figsize=(12, 8))
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        for i, (label, rewards) in enumerate(lr_data.items()):
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                episodes = range(window-1, len(rewards))
            else:
                smoothed = rewards
                episodes = range(len(rewards))
            color = colors[i % len(colors)]
            plt.plot(episodes, smoothed, linewidth=2, label=f'a = {label}', color=color)

        plt.xlabel('Episode')
        plt.ylabel('Average Reward (smoothed)')
        plt.title('Learning Rate Comparison: Average Reward vs. Episode')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'lr_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_eval_reward(self, eval_rewards: List[float], eval_episodes: List[int]):
        """Plot average evaluation reward at each evaluation point."""
        if not eval_rewards:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(eval_episodes, eval_rewards, 'o-', color='darkorange', linewidth=1.5, markersize=4)
        plt.xlabel('Training Episode')
        plt.ylabel('Average Evaluation Reward (over 3 episodes)')
        plt.title('Evaluation Reward vs. Training Episode')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'eval_reward.png'), dpi=150)
        plt.close()

    def plot_avg_eval_reward(self, eval_rewards: List[float], window: int = 3):
        """Plot smoothed average of evaluation rewards to show trend."""
        if len(eval_rewards) < window:
            return
        smoothed = np.convolve(eval_rewards, np.ones(window)/window, mode='valid')
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed, 'g-', linewidth=2)
        plt.xlabel('Evaluation Point (each point = one evaluation)')
        plt.ylabel(f'Smoothed Evaluation Reward (window={window})')
        plt.title('Average Evaluation Reward Trend')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'avg_eval_reward.png'), dpi=150)
        plt.close()
    
    def plot_all(self, metrics: Dict[str, List[float]], eval_rewards: List[float] = None, eval_episodes: List[int] = None):
        """Generate all available plots from training metrics dict."""
        if 'episode_rewards' in metrics:
            self.plot_reward(metrics['episode_rewards'])
            self.plot_avg_reward(metrics['episode_rewards'])
        if 'avg_delays' in metrics:
            self.plot_delay(metrics['avg_delays'])
        if 'avg_losses' in metrics:
            self.plot_packet_loss(metrics['avg_losses'])
        if 'learning_losses' in metrics:
            self.plot_learning_loss(metrics['learning_losses'])
        if 'exploration_rates' in metrics:
            self.plot_exploration_rate(metrics['exploration_rates'])
        if 'completion_rates' in metrics:
            self.plot_completion_rate(metrics['completion_rates'])
        elif 'qos_violation_rates' in metrics:
            success = [1.0 - v for v in metrics['qos_violation_rates']]
            self.plot_completion_rate(success)
        if 'learning_rates' in metrics and 'episode_rewards' in metrics:
            self.plot_lr_vs_reward(metrics['learning_rates'], metrics['episode_rewards'])
    
        # New: evaluation reward plots
        if eval_rewards and eval_episodes:
            self.plot_eval_reward(eval_rewards, eval_episodes)
            self.plot_avg_eval_reward(eval_rewards)
        