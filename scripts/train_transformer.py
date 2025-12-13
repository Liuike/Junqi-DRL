import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pyspiel

# Ensure project root on path so local packages resolve.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from junqi_drl.agents.junqi_transformer import JunqiMoveTransformer
from junqi_drl.agents.transformer_agent import TransformerAgent
from junqi_drl.agents.random_agent import RandomAgent
from junqi_drl.core.metrics import MetricsLogger
from junqi_drl.game import junqi_8x3  # noqa: F401
from junqi_drl.game import junqi_standard  # noqa: F401

def apply_transformer_action(state, action_int, num_cells):
    """
    Map flattened transformer action to two sequential env actions.
    Returns True if the move was successfully applied, False if the selection was rejected.
    """
    from_idx = action_int // num_cells
    to_idx = action_int % num_cells

    if not getattr(state, "selecting_piece", True):
        raise RuntimeError("Environment expected destination but transformer picked a move.")

    state.apply_action(from_idx)
    if getattr(state, "selecting_piece", True):
        # Transformer pointed at an illegal or stale piece. Abort this move so caller can retry.
        return False

    state.apply_action(to_idx)
    return True


def train_transformer(
    game_mode="junqi_8x3",
    num_iterations=5000,
    num_steps=512,
    minibatch_size=32,
    update_epochs=4,
    opponent_update_freq=50, 
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1,
    lr_start=1e-4,
    lr_end=5e-6,
    gamma=0.99,
    gae_lambda=0.95,
    clip_coef=0.2,
    ent_coef_start=0.02,
    ent_coef_end=0.001,
    vf_coef=0.5,
    max_grad_norm=0.5,
    save_dir="models_transformer",
    device="cpu",
    eval_every=100,
    eval_episodes=50,
    use_wandb=False,
    wandb_project="junqi-transformer",
    wandb_run_name=None
):
    """
    Train transformer agent with PPO.
    
    Args:
        game_mode: "junqi_8x3" or "junqi_standard"
        num_iterations: Total training iterations
        num_steps: Steps per iteration
        minibatch_size: Minibatch size for PPO updates
        update_epochs: PPO update epochs per iteration
        d_model: Transformer model dimension
        opponent_update_freq: Iterations between syncing opponent weights
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
        lr_start: Starting learning rate
        lr_end: Ending learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_coef: PPO clip coefficient
        ent_coef_start: Starting entropy coefficient
        ent_coef_end: Ending entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm for clipping
        save_dir: Directory to save models
        device: "cpu" or "cuda"
        eval_every: Evaluate every N iterations
        eval_episodes: Number of episodes for evaluation
        use_wandb: Enable WandB logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name
    """
    os.makedirs(save_dir, exist_ok=True)
    
    board_variant = "small" if game_mode == "junqi_8x3" else "standard"
    
    # Initialize metrics logger
    wandb_config = {
        "project": wandb_project,
        "name": wandb_run_name,
        "config": {
            "game_mode": game_mode,
            "board_variant": board_variant,
            "num_iterations": num_iterations,
            "num_steps": num_steps,
            "minibatch_size": minibatch_size,
            "update_epochs": update_epochs,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
            "lr_start": lr_start,
            "lr_end": lr_end,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_coef": clip_coef,
            "ent_coef_start": ent_coef_start,
            "ent_coef_end": ent_coef_end,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "device": str(device),
        }
    }
    
    metrics_logger = MetricsLogger(
        use_wandb=use_wandb,
        wandb_config=wandb_config
    )
    
    game = pyspiel.load_game(game_mode)

    print(f"--- Starting PPO Training: {game_mode} ---")
    print(f"Device: {device}")

    agent = TransformerAgent(
        player_id=0,
        board_variant=board_variant,
        device=device,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        game=game,
        lr_start=lr_start,
        lr_end=lr_end,
        ent_coef_start=ent_coef_start,
        ent_coef_end=ent_coef_end,
        training_mode=True
    )
    opponent_agent = TransformerAgent(
        player_id=1,
        board_variant=board_variant,
        device=device,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=0.0, # No dropout for inference opponent
        game=game,
        training_mode=False
    )
    opponent_agent.model.load_state_dict(agent.model.state_dict())
    print("Initialized Opponent with Main Agent weights.")

    eval_opponent = RandomAgent(player_id=1)
    best_win_rate = 0.0

    for iteration in range(1, num_iterations + 1):
        frac = (iteration - 1.0) / num_iterations
        current_lr, current_ent_coef = agent.update_parameters(frac)

        obs_buffer, mask_buffer = [], []
        action_buffer, logprob_buffer = [], []
        reward_buffer, value_buffer, done_buffer = [], [], []

        state = game.new_initial_state()
        steps_collected = 0

        while steps_collected < num_steps:
            if state.is_terminal():
                state = game.new_initial_state()
                opponent_agent.pending_to_idx = None
                continue

            player_id = state.current_player()
            # current_player() may set terminal if no legal moves
            if state.is_terminal():
                if reward_buffer:
                    final_reward = state.returns()[0]
                    reward_buffer[-1] = torch.tensor(final_reward, device=device)
                    done_buffer[-1] = torch.tensor(1.0, dtype=torch.float32, device=device)
                state = game.new_initial_state()
                opponent_agent.pending_to_idx = None
                continue
            
            if player_id == 0:
                if not getattr(state, "selecting_piece", True):
                    raise RuntimeError("Transformer step requested while env awaits destination.")

                with torch.no_grad():
                    obs = agent.process_obs(state, player_id)
                    mask = agent.process_mask(state, player_id)
                    logits, value = agent.model(obs, mask=mask)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)

                transformer_action = action.item()
                success = apply_transformer_action(
                    state,
                    transformer_action,
                    agent.num_cells,
                )
                if not success:
                    # Skip logging this step so the buffer stays consistent.
                    continue

                rewards = state.rewards()
                step_reward = rewards[player_id]

                obs_buffer.append(obs)
                mask_buffer.append(mask)
                action_buffer.append(action)
                logprob_buffer.append(logprob)
                value_buffer.append(value)
                reward_buffer.append(torch.tensor(step_reward, device=device))
                done_buffer.append(torch.tensor(state.is_terminal(), dtype=torch.float32, device=device))

                steps_collected += 1
            else:
                action = opponent_agent.choose_action(state)
                state.apply_action(action)

                if state.is_terminal():
                    if reward_buffer:
                        final_reward = state.returns()[0]
                        reward_buffer[-1] = torch.tensor(final_reward, device=device)
                        done_buffer[-1] = torch.tensor(1.0, dtype=torch.float32, device=device)
                    state = game.new_initial_state()
                    opponent_agent.pending_to_idx = None

        with torch.no_grad():
            if state.is_terminal():
                next_value = 0.0
            else:
                pid = state.current_player()
                obs = agent.process_obs(state, pid)
                mask = agent.process_mask(state, pid)
                _, next_value_tensor = agent.model(obs, mask=mask)
                next_value = next_value_tensor.item()

        rewards = torch.stack(reward_buffer).view(-1)
        values = torch.stack(value_buffer).view(-1)
        dones = torch.stack(done_buffer).view(-1)

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

        returns = advantages + values

        b_obs = torch.cat(obs_buffer)
        b_masks = torch.cat(mask_buffer)
        b_actions = torch.stack(action_buffer).view(-1)
        b_logprobs = torch.stack(logprob_buffer).view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)

        b_inds = np.arange(num_steps)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                logits, newvalue = agent.model(b_obs[mb_inds], mask=b_masks[mb_inds])
                dist = Categorical(logits=logits)
                newlogprob = dist.log_prob(b_actions[mb_inds])
                entropy = dist.entropy()
                newvalue = newvalue.view(-1)

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - current_ent_coef * entropy_loss + vf_coef * v_loss

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.model.parameters(), max_grad_norm)
                agent.optimizer.step()

        if iteration % 20 == 0:
            print(
                f"Iter {iteration} | Loss {loss.item():.4f} | "
                f"Adv {advantages.mean().item():.3f} | LR {current_lr:.2e}"
            )
            
            # Log training metrics
            metrics_logger.log({
                "iteration": iteration,
                "loss": loss.item(),
                "pg_loss": pg_loss.item(),
                "v_loss": v_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "advantages_mean": advantages.mean().item(),
                "learning_rate": current_lr,
                "entropy_coef": current_ent_coef
            }, step=iteration)

        if iteration % eval_every == 0:
            agent.model.eval()
            eval_wins = 0
            for _ in range(eval_episodes):
                s = game.new_initial_state()
                while not s.is_terminal():
                    p = s.current_player()
                    # current_player() may set terminal if no legal moves
                    if s.is_terminal():
                        break
                    
                    if p == 0:
                        if not getattr(s, "selecting_piece", True):
                            raise RuntimeError("Evaluation desync: env expects destination.")
                        with torch.no_grad():
                            o = agent.process_obs(s, p)
                            m = agent.process_mask(s, p)
                            logits, _ = agent.model(o, mask=m)
                            move = logits.argmax(dim=1).item()
                        success = apply_transformer_action(s, move, agent.num_cells)
                        if not success:
                            # If greedy move became stale, skip this game
                            break
                    else:
                        legal_actions = s.legal_actions(1)
                        s.apply_action(eval_opponent.choose_action(s, legal_actions, eval_mode=True))
                if s.is_terminal() and s.returns()[0] > 0:
                    eval_wins += 1

            agent.model.train()
            win_rate = eval_wins / eval_episodes
            print(f"\n>>> Eval @ Iter {iteration}: Win rate vs Random = {win_rate:.2%}")
            
            # Log evaluation metrics
            metrics_logger.log({
                "eval_vs_random_winrate": win_rate
            }, step=iteration)
            
            if win_rate >= best_win_rate:
                best_win_rate = win_rate
                save_path = os.path.join(save_dir, f"transformer_best_wr{win_rate:.2f}_iter{iteration}.pth")
                torch.save(agent.model.state_dict(), save_path)
                print(f"✓ Saved new best model to {save_path}")
                
                metrics_logger.log({
                    "best_eval_winrate": best_win_rate
                }, step=iteration)

            if iteration % 500 == 0:
                periodic_path = os.path.join(save_dir, f"transformer_checkpoint_iter{iteration}.pth")
                torch.save(agent.model.state_dict(), periodic_path)
                print(f"  (Checkpoint saved: {periodic_path})")

        if iteration % opponent_update_freq == 0:
            opponent_agent.model.load_state_dict(agent.model.state_dict())
            opponent_agent.reset()

    torch.save(agent.model.state_dict(), os.path.join(save_dir, "transformer_final.pth"))
    print("Training Complete.")
    print(f"Best win rate vs random: {best_win_rate:.2%}")
    
    metrics_logger.finish()
