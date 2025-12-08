import torch
import numpy as np
from junqi_drl.agents.junqi_transformer import JunqiMoveTransformer


class TransformerAgent:
   """
   Agent wrapping the JunqiMoveTransformer for evaluation/inference.
   Includes stateful logic to adapt 1-step Transformer output to 2-step Environment input.
   """
   def __init__(
       self,
       player_id: int,
       board_variant: str = "standard",
       model_path: str = None,
       device: str = "cpu",
       deterministic: bool = True,
       *,
       d_model: int = 64,
       nhead: int = 4,
       num_layers: int = 2,
       dropout: float = 0.0,
   ):
       self.player_id = player_id
       self.device = device
       self.deterministic = deterministic
      
       # Model Init
       self.model = JunqiMoveTransformer(
           board_variant=board_variant,
           d_model=d_model,
           nhead=nhead,
           num_layers=num_layers,
           dropout=dropout
       ).to(self.device)
       self.model.eval()


       if model_path:
           self.load(model_path)
      
       self.height = self.model.height
       self.width = self.model.width
       self.channels = self.model.channels
       self.seq_len = self.model.seq_len
      
       # [NEW] Internal memory for the 2-step move process
       self.pending_to_idx = None


   def load(self, path: str):
       try:
           checkpoint = torch.load(path, map_location=self.device)
           if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
               self.model.load_state_dict(checkpoint['model_state_dict'])
           else:
               self.model.load_state_dict(checkpoint)
           print(f"Loaded model from {path}")
       except Exception as e:
           print(f"Error loading model: {e}")


   def _process_obs(self, state) -> torch.Tensor:
       obs_flat = state.observation_tensor(self.player_id)
       obs_3d = np.array(obs_flat, dtype=np.float32).reshape(self.height, self.width, self.channels)
       return torch.tensor(obs_3d, device=self.device).unsqueeze(0)


   def _process_mask(self, state) -> torch.Tensor:
       """
       Replicates the mask construction logic using env rules.
       """
       num_cells = self.seq_len
       full_mask = np.zeros(self.model.total_actions, dtype=np.float32)
      
       pass_action_env_id = len(state.decode_action)
       env_legal_actions = state.legal_actions(self.player_id)
      
       # Pass check
       if pass_action_env_id in env_legal_actions:
           full_mask[-1] = 1.0
           if len(env_legal_actions) == 1:
               return torch.tensor(full_mask, dtype=torch.float32, device=self.device).unsqueeze(0)


       decode_action = state.decode_action
      
       # Move check
       for from_idx in range(num_cells):
           # Only consider pieces that are legal to select NOW
           if from_idx in env_legal_actions:
               row, col = decode_action[from_idx]
               try:
                   # Accessing protected method to build full mask
                   destinations = state._get_legal_destinations([row, col], self.player_id)
               except AttributeError:
                   destinations = []


               for r_to, c_to in destinations:
                   to_idx = r_to * self.width + c_to
                   flat_action_idx = from_idx * num_cells + to_idx
                   full_mask[flat_action_idx] = 1.0


       return torch.tensor(full_mask, dtype=torch.float32, device=self.device).unsqueeze(0)


   def choose_action(self, state) -> int:
       """
       Stateful action selection with Safety Net.
       """
       # Case 1: executing the second half of a move (Dest Selection)
       if self.pending_to_idx is not None:
           # Safety Check: If we have a pending move, but the env is asking to "Select Piece" again,
           # it means our previous "Select Piece" failed. Discard the pending move.
           if state.selecting_piece:
               self.pending_to_idx = None
               # Don't return here; fall through to Case 3 to pick a new move immediately
           else:
               action = self.pending_to_idx
               self.pending_to_idx = None
               return action

       # Case 2: Pass action handling (Env might only allow Pass)
       pass_id = len(state.decode_action)
       legal_actions = state.legal_actions(self.player_id)
       
       if legal_actions == [pass_id]:
           return pass_id

       # Case 3: Start new move (Step 1: Source Selection)
       obs_tensor = self._process_obs(state)
       mask_tensor = self._process_mask(state)

       with torch.no_grad():
           action_logits, _ = self.model(obs_tensor, mask=mask_tensor)
           action_logits = action_logits.squeeze(0)

       if self.deterministic:
           action_int = torch.argmax(action_logits).item()
       else:
           probs = torch.softmax(action_logits, dim=0)
           action_int = torch.multinomial(probs, 1).item()

       # Decode
       pass_idx_transformer = self.model.total_actions - 1
       
       if action_int == pass_idx_transformer:
           return pass_id
       else:
           num_cells = self.seq_len
           from_idx = action_int // num_cells
           to_idx = action_int % num_cells
           
           # --- [NEW] SAFETY NET ---
           # Check if the model's chosen 'from_idx' is actually legal.
           # If the model is hallucinating (picking an empty square or enemy piece),
           # we intercept it here to prevent the state machine desync.
           if from_idx not in legal_actions:
               # Option A: Fallback to a random legal move (Safe & Simple)
               # We do NOT set pending_to_idx, so the agent will re-evaluate next step.
               import numpy as np
               fallback_action = np.random.choice(legal_actions)
               return fallback_action.item()

           # Only set pending if the first step is valid
           self.pending_to_idx = to_idx
           return from_idx