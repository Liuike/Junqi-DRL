import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pyspiel


# Ensure the project root is on the path so local modules are discoverable.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from junqi_drl.agents.junqi_transformer import JunqiMoveTransformer
from junqi_drl.agents.random_agent import RandomAgent
# Import game modules so OpenSpiel registers them.
from junqi_drl.game import junqi_8x3  # noqa: F401
from junqi_drl.game import junqi_standard  # noqa: F401


#   Global Configuration
GAMEMODE = "junqi_8x3"  # or "junqi_standard"
BOARD_VARIANT = "small" if GAMEMODE == "junqi_8x3" else "standard"
SAVE_DIR = "models_transformer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Enable Weights & Biases logging by default when available.
USE_WANDB = True
WANDB_PROJECT = "junqi-transformer"
WANDB_RUN_NAME = f"{GAMEMODE}-ppo-transformer"




# --- PPO Hyperparameters ---
NUM_ITERATIONS = 5000
NUM_STEPS = 512          # GPU: 2048 | Reduced for CPU speed
MINIBATCH_SIZE = 32      # GPU: 64   | Reduced for CPU memory stability
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




# ----------------------------- Helper Utilities -----------------------------
def apply_transformer_action(state, action_int, num_cells, pass_action_env_id):
   """
   Attempts to apply a transformer action (From->To flattened index) to the OpenSpiel state.
   Returns True if successful, False if the move was illegal (e.g., wrong piece selected).
   """
   # Handle explicit pass action
   if action_int == num_cells * num_cells:
       state.apply_action(pass_action_env_id)
       return True


   from_idx = action_int // num_cells
   to_idx = action_int % num_cells


   # Synchronization check: Ensure environment is ready for a 'Select' move
   if not getattr(state, "selecting_piece", True):
       raise RuntimeError("Environment expected destination but transformer picked a move.")


   # Step 1: Select Piece
   state.apply_action(from_idx)
  
   # Check if selection failed (e.g., selected empty square or enemy piece)
   if getattr(state, "selecting_piece", True):
       return False


   # Step 2: Select Destination
   state.apply_action(to_idx)
   return True




class JunqiPPOAgent(nn.Module):
   """Thin wrapper that hosts the transformer policy/value network and helpers."""


   def __init__(self, board_variant, device, game):
       super().__init__()
       self.device = device
       self.game = game
      
       # CPU-Optimized Model Architecture (Slim version)
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
       """
       Constructs action mask. shape: [1, S*S+1]
       """
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
   # Optimization for CPU: Single-threaded operation often faster for small batches
   torch.set_num_threads(1)
   os.environ["OMP_NUM_THREADS"] = "1"


   os.makedirs(SAVE_DIR, exist_ok=True)
   game = pyspiel.load_game(GAMEMODE)


   print(f"--- Starting PPO Training: {GAMEMODE} ---")
   print(f"Device: {DEVICE}")


   # Optional experiment tracking hook (no-op when USE_WANDB=False).
   if USE_WANDB:
       import wandb
       wandb.init(
           project="Junqi-DRL",
           entity=None,
           name=f"{GAMEMODE}-ppo-transformer",
           group=f"{GAMEMODE}-runs",
           tags=["PPO", "Transformer", BOARD_VARIANT],
           config={
               "game_mode": GAMEMODE,
               "board_variant": BOARD_VARIANT,
               "num_iterations": NUM_ITERATIONS,
               "num_steps_per_iter": NUM_STEPS,
               "minibatch_size": MINIBATCH_SIZE,
               "update_epochs": UPDATE_EPOCHS,
               "gamma": GAMMA,
               "gae_lambda": GAE_LAMBDA,
               "clip_coef": CLIP_COEF,
               "vf_coef": VF_COEF,
               "max_grad_norm": MAX_GRAD_NORM,
               "lr_start": LR_START,
               "lr_end": LR_END,
               "entropy_coef_start": ENT_COEF_START,
               "entropy_coef_end": ENT_COEF_END,
               "model": "JunqiMoveTransformer",
               "d_model": 64,
               "nhead": 4,
               "num_layers": 2,
               "dropout": 0.1,
               "eval_games": 50,
               "device": str(DEVICE),
           },
       )
   else:
       wandb = None


   # The Training Agent (Transformer)
   agent = JunqiPPOAgent(BOARD_VARIANT, DEVICE, game)
  
   # The Evaluation Opponent (Random) - Used ONLY in periodic eval, NOT in training loop
   eval_opponent = RandomAgent(player_id=1)
  
   best_win_rate = 0.0
   pass_action_env_id = game.num_distinct_actions() - 1


   global_episode = 0  # Tracks absolute episode count for logging.


   for iteration in range(1, NUM_ITERATIONS + 1):
       frac = (iteration - 1.0) / NUM_ITERATIONS
       current_lr, current_ent_coef = agent.update_parameters(frac)


       # rollout buffer
       obs_buffer, mask_buffer = [], []
       action_buffer, logprob_buffer = [], []
       reward_buffer, value_buffer, done_buffer = [], [], []


       state = game.new_initial_state()
       steps_collected = 0


       # Stats for this iteration
       ep_rewards_this_iter = []
       ep_lengths_this_iter = []
       cur_ep_reward = 0.0
       cur_ep_len = 0


       # --------- Collect NUM_STEPS transitions for PPO rollout ---------
       while steps_collected < NUM_STEPS:
           # Handle terminal state from previous step
           if state.is_terminal():
               if cur_ep_len > 0:
                   global_episode += 1
                   ep_rewards_this_iter.append(cur_ep_reward)
                   ep_lengths_this_iter.append(cur_ep_len)
                   cur_ep_reward = 0.0
                   cur_ep_len = 0


               state = game.new_initial_state()
               continue


           # Check environment sync
           if not getattr(state, "selecting_piece", True):
               raise RuntimeError("Transformer step requested while env awaits destination.")


           player_id = state.current_player()


           # 1. Agent selects action
           with torch.no_grad():
               obs = agent.process_obs(state, player_id)
               mask = agent.process_mask(state, player_id)
               logits, value = agent.model(obs, mask=mask)
               dist = Categorical(logits=logits)
               action = dist.sample()
               logprob = dist.log_prob(action)


           # 2. Try to apply action to environment
           transformer_action = action.item()
           success = apply_transformer_action(
               state,
               transformer_action,
               agent.num_cells,
               pass_action_env_id,
           )


           # 3. [CRITICAL FIX] Handle Illegal Moves (Survivor Bias Fix)
           # If the agent makes an illegal move, we MUST record it as a failure
           # instead of skipping it.
           if not success:
               step_reward = -1.0  # Penalty for illegal move
              
               # Store this "Lesson" in the buffer
               obs_buffer.append(obs)
               mask_buffer.append(mask)
               action_buffer.append(action)
               logprob_buffer.append(logprob)
               value_buffer.append(value)
               reward_buffer.append(torch.tensor(step_reward, device=DEVICE))
              
               # Mark as terminal/done because we are resetting the environment
               done_buffer.append(torch.tensor(1.0, dtype=torch.float32, device=DEVICE))
              
               # Update episode stats
               cur_ep_reward += step_reward
               ep_rewards_this_iter.append(cur_ep_reward)
               ep_lengths_this_iter.append(cur_ep_len + 1)
              
               # Reset counters and state
               cur_ep_reward = 0.0
               cur_ep_len = 0
               state = game.new_initial_state()
               global_episode += 1
              
               steps_collected += 1
               continue


           # 4. Handle Valid Moves
           # Rewards only appear at terminal states for this game definition, usually
           step_reward = 0.0
           if state.is_terminal():
               final_returns = state.returns()
               step_reward = final_returns[player_id]


           # Update stats
           cur_ep_reward += step_reward
           cur_ep_len += 1


           # Store valid experience
           obs_buffer.append(obs)
           mask_buffer.append(mask)
           action_buffer.append(action)
           logprob_buffer.append(logprob)
           value_buffer.append(value)
           reward_buffer.append(torch.tensor(step_reward, device=DEVICE))
           done_buffer.append(torch.tensor(state.is_terminal(), dtype=torch.float32, device=DEVICE))


           steps_collected += 1


       # Handle the dangling episode at the end of collection
       if state.is_terminal() and cur_ep_len > 0:
           global_episode += 1
           ep_rewards_this_iter.append(cur_ep_reward)
           ep_lengths_this_iter.append(cur_ep_len)


       # --------- compute GAE advantage ---------
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


   # PPO MINIBATCH UPDATE LOOP
       b_inds = np.arange(NUM_STEPS)
       total_pg_loss = 0.0
       total_v_loss = 0.0
       total_entropy = 0.0
       total_loss = 0.0
       num_updates = 0


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


               # iteration avg loss
               total_pg_loss += pg_loss.item()
               total_v_loss += v_loss.item()
               total_entropy += entropy_loss.item()
               total_loss += loss.item()
               num_updates += 1


       avg_pg_loss = total_pg_loss / max(num_updates, 1)
       avg_v_loss = total_v_loss / max(num_updates, 1)
       avg_entropy = total_entropy / max(num_updates, 1)
       avg_loss = total_loss / max(num_updates, 1)
       adv_mean = advantages.mean().item()
       returns_mean = returns.mean().item()


      
       if len(ep_rewards_this_iter) > 0:
           mean_ep_reward = float(np.mean(ep_rewards_this_iter))
           mean_ep_len = float(np.mean(ep_lengths_this_iter))
       else:
           mean_ep_reward = 0.0
           mean_ep_len = 0.0


       # --------- print iteration PPO information---------
       # Prints every 20 iterations to reduce clutter
       if iteration % 20 == 0:
           print(
               f"Iter {iteration:4d} | Loss {avg_loss:+.4f} | "
               f"MeanEpR {mean_ep_reward:+.3f} | "
               f"Eps {len(ep_rewards_this_iter)} | "
               f"LR {current_lr:.2e}"
           )


       # wandb record train logs
       if USE_WANDB:
           wandb.log(
               {
                   "train/iter": iteration,
                   "train/loss_total": avg_loss,
                   "train/loss_policy": avg_pg_loss,
                   "train/loss_value": avg_v_loss,
                   "train/entropy": avg_entropy,
                   "train/adv_mean": adv_mean,
                   "train/mean_ep_reward": mean_ep_reward,
                   "train/mean_ep_length": mean_ep_len,
                   "train/episodes_per_iter": len(ep_rewards_this_iter),
                   "train/lr": current_lr,
               },
               step=iteration,
           )


       # --------- Periodic Evaluation against baseline agent ---------
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
                           # In eval, if agent makes illegal move, force a pass (or loss)
                           s.apply_action(pass_action_env_id)
                   else:
                       # Opponent plays here
                       s.apply_action(eval_opponent.step(s))
               if s.returns()[0] > 0:
                   eval_wins += 1


           agent.model.train()
           win_rate = eval_wins / num_eval_games
           print(f"\n>>> Eval @ Iter {iteration}: Win rate vs Random = {win_rate:.2%}")


           # wandb record eval
           if USE_WANDB:
               wandb.log(
                   {
                       "eval/iter": iteration,
                       "eval/win_rate": win_rate,
                       "eval/best_win_rate": best_win_rate,
                   },
                   step=iteration,
               )


           # update best model
           if win_rate >= best_win_rate:
               best_win_rate = win_rate
               os.makedirs(SAVE_DIR, exist_ok=True)
               save_path = os.path.join(
                   SAVE_DIR,
                   f"transformer_best_wr{win_rate:.2f}_iter{iteration}.pth",
               )
               torch.save(agent.model.state_dict(), save_path)
               print(f"Saved new best model to {save_path}")


       # checkpoint
       if iteration % 500 == 0:
           ckpt_path = os.path.join(SAVE_DIR, f"transformer_iter{iteration}.pth")
           torch.save(agent.model.state_dict(), ckpt_path)
           print(f"[Checkpoint] Saved model at iter {iteration} to {ckpt_path}")


   # Final checkpoint after training loop completes.
   final_path = os.path.join(SAVE_DIR, "transformer_final.pth")
   torch.save(agent.model.state_dict(), final_path)
   print(f"Training Complete. Final model saved to {final_path}")


   if USE_WANDB:
       wandb.finish()




if __name__ == "__main__":
   train()
