"""Replay buffer implementations for off-policy RL algorithms."""

import random
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Any


class ReplayBuffer:
    """
    Standard experience replay buffer for DQN-style algorithms.
    
    Stores transitions and samples uniformly at random.
    """
    
    def __init__(self, maxlen: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            maxlen: Maximum buffer capacity
        """
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen
    
    def append(self, transition: Tuple):
        """
        Add a transition to the buffer.
        
        Args:
            transition: (obs, action, reward, next_obs, done) tuple
        """
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of transitions uniformly at random.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


class TemporalStratifiedReplayBuffer:
    """
    Temporal stratified replay buffer that divides the buffer into time segments
    and samples proportionally from each segment to ensure early experiences are represented.
    
    This helps prevent catastrophic forgetting of early learned behaviors.
    """
    
    def __init__(self, maxlen: int = 100000, num_segments: int = 4):
        """
        Initialize temporal stratified replay buffer.
        
        Args:
            maxlen: Maximum total buffer size
            num_segments: Number of temporal segments to divide the buffer into
        """
        self.maxlen = maxlen
        self.num_segments = num_segments
        self.segment_size = maxlen // num_segments
        
        # Create separate deques for each temporal segment
        self.segments = [deque(maxlen=self.segment_size) for _ in range(num_segments)]
        self.current_segment = 0
        self.total_added = 0
    
    def append(self, transition: Tuple):
        """
        Add transition to the current segment.
        Advances to next segment when current is full.
        
        Args:
            transition: (obs, action, reward, next_obs, done) tuple
        """
        self.segments[self.current_segment].append(transition)
        self.total_added += 1
        
        # Move to next segment in round-robin fashion when segment is full
        if len(self.segments[self.current_segment]) >= self.segment_size:
            self.current_segment = (self.current_segment + 1) % self.num_segments
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample uniformly from all temporal segments.
        Each segment contributes proportionally to its size.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled transitions
        """
        batch = []
        
        # Count non-empty segments
        non_empty_segments = [seg for seg in self.segments if len(seg) > 0]
        
        if not non_empty_segments:
            return batch
        
        # Calculate samples per segment (uniform across non-empty segments)
        samples_per_segment = batch_size // len(non_empty_segments)
        remainder = batch_size % len(non_empty_segments)
        
        for i, segment in enumerate(non_empty_segments):
            # Add extra sample to first 'remainder' segments to reach exact batch_size
            n_samples = samples_per_segment + (1 if i < remainder else 0)
            n_samples = min(n_samples, len(segment))
            
            batch.extend(random.sample(list(segment), n_samples))
        
        # If we still don't have enough, sample with replacement from all segments
        if len(batch) < batch_size:
            all_transitions = []
            for segment in non_empty_segments:
                all_transitions.extend(list(segment))
            
            if all_transitions:
                remaining = batch_size - len(batch)
                batch.extend(random.choices(all_transitions, k=remaining))
        
        return batch
    
    def __len__(self) -> int:
        """Return total number of transitions across all segments."""
        return sum(len(seg) for seg in self.segments)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about buffer composition.
        
        Returns:
            Dictionary with buffer statistics
        """
        stats = {
            'total': len(self),
            'total_added': self.total_added,
            'current_segment': self.current_segment,
            'segments': []
        }
        
        for i, segment in enumerate(self.segments):
            stats['segments'].append({
                'id': i,
                'size': len(segment),
                'is_current': i == self.current_segment
            })
        
        return stats
    
    def clear(self):
        """Clear all segments."""
        for segment in self.segments:
            segment.clear()
        self.current_segment = 0
        self.total_added = 0
