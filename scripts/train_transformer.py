import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pyspiel

# Ensure project root on path so local packages resolve.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from junqi_drl.agents.junqi_transformer import JunqiMoveTransformer
from junqi_drl.agents.random_agent import RandomAgent
# Import game modules so pyspiel knows about registrations.
from junqi_drl.game import junqi_8x3  # noqa: F401
from junqi_drl.game import junqi_standard  # noqa: F401

# --- Configuration ---
GAMEMODE = "junqi_8x3"  # or "junqi_standard"
BOARD_VARIANT = "small" if GAMEMODE == "junqi_8x3" else "standard"
SAVE_DIR = "models_transformer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PPO Hyperparameters ---
NUM_ITERATIONS = 5000
NUM_STEPS = 512          # GPU: 2048
MINIBATCH_SIZE = 32      # GPU: 64
UPDATE_EPOCHS = 4
LR_START = 1e-4
LR_END = 5e-6
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF_START = 0.02
ENT_COEF_END = 0.001
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5


def apply_transformer_action(state, action_int, num_cells, pass_action_env_id):
    """
    Map flattened transformer action to two sequential env actions.
    Returns True if the move was successfully applied, False if the selection was rejected.
    """
    if action_int == num_cells * num_cells:
        state.apply_action(pass_action_env_id)
        return True

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


class JunqiPPOAgent(nn.Module):
    """Wrapper holding the transformer model plus preprocessing utilities."""

    def __init__(self, board_variant, device, game):
        super().__init__()
        self.device = device
        self.game = game
        self.model = JunqiMoveTransformer(
            board_variant=board_variant,
            d_model=64,      # GPU: 256
            nhead=4,         # GPU: 8
            num_layers=2,    # GPU: 4
            dropout=0.1,
        ).to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_START, eps=1e-5)

        self.height = self.model.height
        self.width = self.model.width
        self.channels = self.model.channels
        self.num_cells = self.model.seq_len
        self.total_actions = self.model.total_actions

        self.lr_scheduler = lambda frac: LR_START + frac * (LR_END - LR_START)
        self.ent_scheduler = lambda frac: ENT_COEF_START + frac * (ENT_COEF_END - ENT_COEF_START)

    def process_obs(self, state, player_id):
        obs_flat = state.observation_tensor(player_id)
        obs_3d = np.array(obs_flat, dtype=np.float32).reshape(self.height, self.width, self.channels)
        return torch.tensor(obs_3d, device=self.device).unsqueeze(0)

    def process_mask(self, state, player_id):
        """Construct a (1, S*S+1) legality mask by querying env rules."""
        if not getattr(state, "selecting_piece", True):
            raise RuntimeError("Mask requested while environment expects destination choice.")

        full_mask = np.zeros(self.total_actions, dtype=np.float32)
        env_legal_actions = state.legal_actions(player_id)
        pass_env_id = self.game.num_distinct_actions() - 1

        if pass_env_id in env_legal_actions:
            full_mask[-1] = 1.0
            if len(env_legal_actions) == 1:
                return torch.tensor(full_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        decode_action = state.decode_action
        for from_idx in env_legal_actions:
            if from_idx == pass_env_id or from_idx >= self.num_cells:
                continue
            row, col = decode_action[from_idx]
            try:
                destinations = state._get_legal_destinations([row, col], player_id)
            except AttributeError:
                destinations = []
            for r_to, c_to in destinations:
                to_idx = r_to * self.width + c_to
                flat_idx = from_idx * self.num_cells + to_idx
                full_mask[flat_idx] = 1.0

        return torch.tensor(full_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

    def update_parameters(self, frac):
        new_lr = self.lr_scheduler(frac)
        new_ent = self.ent_scheduler(frac)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr, new_ent


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    game = pyspiel.load_game(GAMEMODE)

    print(f"--- Starting PPO Training: {GAMEMODE} ---")
    print(f"Device: {DEVICE}")

    agent = JunqiPPOAgent(BOARD_VARIANT, DEVICE, game)
    eval_opponent = RandomAgent(player_id=1)
    best_win_rate = 0.0
    pass_action_env_id = game.num_distinct_actions() - 1

    for iteration in range(1, NUM_ITERATIONS + 1):
        frac = (iteration - 1.0) / NUM_ITERATIONS
        current_lr, current_ent_coef = agent.update_parameters(frac)

        obs_buffer, mask_buffer = [], []
        action_buffer, logprob_buffer = [], []
        reward_buffer, value_buffer, done_buffer = [], [], []

        state = game.new_initial_state()
        steps_collected = 0

        while steps_collected < NUM_STEPS:
            if state.is_terminal():
                state = game.new_initial_state()
                continue

            if not getattr(state, "selecting_piece", True):
                raise RuntimeError("Transformer step requested while env awaits destination.")

            player_id = state.current_player()

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
                pass_action_env_id,
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
            reward_buffer.append(torch.tensor(step_reward, device=DEVICE))
            done_buffer.append(torch.tensor(state.is_terminal(), dtype=torch.float32, device=DEVICE))

            steps_collected += 1

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
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]
            delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam

        returns = advantages + values

        b_obs = torch.cat(obs_buffer)
        b_masks = torch.cat(mask_buffer)
        b_actions = torch.stack(action_buffer).view(-1)
        b_logprobs = torch.stack(logprob_buffer).view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)

        b_inds = np.arange(NUM_STEPS)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
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
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - current_ent_coef * entropy_loss + VF_COEF * v_loss

                agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.model.parameters(), MAX_GRAD_NORM)
                agent.optimizer.step()

        if iteration % 20 == 0:
            print(
                f"Iter {iteration} | Loss {loss.item():.4f} | "
                f"Adv {advantages.mean().item():.3f} | LR {current_lr:.2e}"
            )

        if iteration % 100 == 0:
            agent.model.eval()
            eval_wins = 0
            num_eval_games = 50
            for _ in range(num_eval_games):
                s = game.new_initial_state()
                while not s.is_terminal():
                    p = s.current_player()
                    if p == 0:
                        if not getattr(s, "selecting_piece", True):
                            raise RuntimeError("Evaluation desync: env expects destination.")
                        with torch.no_grad():
                            o = agent.process_obs(s, p)
                            m = agent.process_mask(s, p)
                            logits, _ = agent.model(o, mask=m)
                            move = logits.argmax(dim=1).item()
                        success = apply_transformer_action(s, move, agent.num_cells, pass_action_env_id)
                        if not success:
                            # If greedy move became stale, treat as pass to keep eval moving.
                            s.apply_action(pass_action_env_id)
                    else:
                        s.apply_action(eval_opponent.step(s))
                if s.returns()[0] > 0:
                    eval_wins += 1

            agent.model.train()
            win_rate = eval_wins / num_eval_games
            print(f"\n>>> Eval @ Iter {iteration}: Win rate vs Random = {win_rate:.2%}")
            if win_rate >= best_win_rate:
                best_win_rate = win_rate
                os.makedirs(SAVE_DIR, exist_ok=True)
                save_path = os.path.join(SAVE_DIR, f"transformer_best_wr{win_rate:.2f}.pth")
                torch.save(agent.model.state_dict(), save_path)
                print(f"Saved new best model to {save_path}")

    torch.save(agent.model.state_dict(), os.path.join(SAVE_DIR, "transformer_final.pth"))
    print("Training Complete.")


if __name__ == "__main__":
    train()