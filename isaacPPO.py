import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOPolicy(nn.Module):
    def __init__(self, room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory, isaacNumber):
        super(PPOPolicy, self).__init__()
        self.room_shape = room_shape
        self.map_shape = map_shape
        self.n_critical = n_critical
        self.n_items = n_items
        self.n_entity_memory = n_entity_memory
        self.visualData = None
        self.isaacNumber = isaacNumber

        # Single convolution for room_grid
        self.room_conv = nn.Conv2d(room_shape[0], 32, kernel_size=1)  # 1x1 conv, preserves 16x28
        room_conv_out_size = 32 * room_shape[1] * room_shape[2]

        self.map_conv_base = nn.Conv2d(map_shape[0], 16, kernel_size=1)  # 1x1 conv, preserves 13x13
        map_conv_out_size = 16 * map_shape[1] * map_shape[2]  # 16 * 13 * 13 = 2704

        self.critical_fc = nn.Sequential(
            nn.Linear(n_critical, 64),
            nn.Linear(64, 32)
        )
        self.items_fc = nn.Sequential(
            nn.Linear(n_items, 64),
            nn.Linear(64, 32)
        )
        self.memory_fc = nn.Sequential(
            nn.Linear(n_entity_memory, 128),
            nn.Linear(128, 64)
        )

        total_features = room_conv_out_size + map_conv_out_size + 32 + 32 + 64
        self.fc1 = nn.Linear(total_features, 1024)
        self.actor = nn.Sequential(nn.Linear(1024, 128), nn.Linear(128, action_size))
        self.critic = nn.Sequential(nn.Linear(1024, 128), nn.Linear(128, 1))

        self._initialize_weights()

    def forward(self, room_grid, map_grid, additional_values):
        batch_size = room_grid.size(0)

        # Process room_grid as a single input
        x_room = self.room_conv(room_grid)  # Shape: [batch_size, 64, 16, 28], preserves negative values
        x_room = x_room.view(batch_size, -1)  # Flatten: [batch_size, 64 * 16 * 28]

        x_map = self.map_conv_base(map_grid)  # Shape: [batch_size, 16, 13, 13]
        x_map = x_map.view(batch_size, -1)  # Flatten: [batch_size, 16 * 13 * 13]

        crit_end = self.n_critical
        item_end = crit_end + self.n_items
        x_critical = self.critical_fc(additional_values[:, :crit_end])
        x_items = self.items_fc(additional_values[:, crit_end:item_end])
        x_memory = self.memory_fc(additional_values[:, item_end:])

        x = torch.cat([x_room, x_map, x_critical, x_items, x_memory], dim=1)
        x = self.fc1(x)
        value = self.critic(x)
        logits = self.actor(x)

        # Updated visualData
        self.visualData = (x_room.view(batch_size, 32, 16, 28),x_map.view(batch_size, 16, 13, 13),x.view(batch_size, 32, 32))

        return logits, value

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

class PPOAgent:
    def __init__(self, room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory, isaacNumber, lr=0.0003, gamma=0.99, clip_param=0.2, value_loss_coef=0.25, entropy_coef=0.01, max_grad_norm=0.5, n_steps=4096):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPOPolicy(room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory, isaacNumber).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': [p for n, p in self.policy.named_parameters() if 'critic' not in n], 'lr': lr},
            {'params': [p for n, p in self.policy.named_parameters() if 'critic' in n], 'lr': lr*.5}
        ])
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.step_counter = 0
        self.episode_counter = 0 #just for the overlay
        self.progress = 0 #just for the overlay
        self.entropy = 0 #just for the overlay
        self.probs = 0 #just for the overlay

    def act(self, state):
        room_grid, map_grid, additional_values = state
        with torch.no_grad():
            logits, value = self.policy(room_grid, map_grid, additional_values)
            self.probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(self.probs, 1).item()
            log_prob = self.probs.log().gather(1, torch.tensor([[action]], device=self.device))
        return action, log_prob, value

    def learn(self, rollouts, n_epochs=12, batch_size=128):
        states, actions, rewards, next_states, dones, old_log_probs, values = rollouts

        # Use old policy outputs from act()
        values = torch.cat(values).squeeze(-1)  # Pre-computed value estimates
        values = (values - values.mean()) / (values.std() + 1e-8)
        next_states_tensor = self._process_state_batch(next_states)
        next_room_grids, next_map_grids, next_adds = [t.to(self.device) for t in next_states_tensor]
        with torch.no_grad():
            _, next_value = self.policy(next_room_grids, next_map_grids, next_adds)
        next_value = next_value.squeeze(-1)

        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value[t]
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        # Gradient accumulation settings
        accumulation_steps = 1
        effective_batch_size = batch_size  # 128
        mini_batch_size = effective_batch_size // accumulation_steps  # 128 with accumulation_steps=1

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        total_batches = 0

        # Calculate total number of batch iterations for progress
        batches_per_epoch = (len(states) + effective_batch_size - 1) // effective_batch_size  # Ceiling division
        total_iterations = n_epochs * batches_per_epoch
        current_iteration = 0

        for epoch in range(n_epochs):
            for i in range(0, len(states), effective_batch_size):  # Process in full batches
                self.optimizer.zero_grad()  # Clear gradients at the start of each batch
                for j in range(accumulation_steps):  # Accumulate over mini-batches
                    batch_start = i + j * mini_batch_size
                    if batch_start >= len(states):
                        break
                    batch_indices = slice(batch_start, min(batch_start + mini_batch_size, len(states)))

                    # Process batch
                    batch_states = self._process_state_batch(states[batch_indices])
                    batch_room_grids, batch_map_grids, batch_adds = [t.to(self.device) for t in batch_states]
                    batch_actions = torch.tensor(actions[batch_indices], device=self.device)
                    batch_old_log_probs = torch.stack(old_log_probs[batch_indices]).squeeze()
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]

                    # Forward pass
                    logits, value = self.policy(batch_room_grids, batch_map_grids, batch_adds)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    log_probs = dist.log_prob(batch_actions)
                    self.entropy = dist.entropy().mean()

                    # Compute loss
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    expected_loss = -torch.min(surr1, surr2).mean().item()
                    #print(f"Expected: {expected_loss:.4f}, Logged: {policy_loss.item():.4f}")
                    value_loss = F.mse_loss(value.squeeze(-1), batch_returns, reduction='mean')
                    entropy_loss = -self.entropy_coef * self.entropy
                    #print(f"Ratio: {ratio.mean():.4f}, Surr1: {surr1.mean():.4f}, Surr2: {surr2.mean():.4f}, Adv: {batch_advantages.mean():.4f}")
                    #print(f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy Loss: {entropy_loss.item():.4f}")
                    loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * self.entropy

                    # Scale loss for accumulation
                    loss = loss / accumulation_steps
                    loss.backward()  # Accumulate gradients

                    # Track losses
                    batch_size_actual = batch_actions.size(0)
                    total_policy_loss += policy_loss.item() * batch_size_actual
                    total_value_loss += value_loss.item() * batch_size_actual
                    total_entropy += self.entropy.item() * batch_size_actual
                    total_loss += loss.item() * batch_size_actual * accumulation_steps
                    total_batches += batch_size_actual

                # After accumulating gradients, update weights
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()  # Apply the accumulated gradients

                # Update progress
                current_iteration += 1
                self.progress = (current_iteration / total_iterations) * 100

                print("=== Gradients Summary ===")
                for name, param in self.policy.named_parameters():
                    if param.grad is not None:
                        print(f"{name}: mean={param.grad.mean():.6f}, std={param.grad.std():.6f}")
                    else:
                        print(f"{name}: No gradient computed")

        # Rest of the summary code
        total_policy_loss /= total_batches
        total_value_loss /= total_batches
        total_entropy /= total_batches
        total_loss /= total_batches

        self.entropy = total_entropy
        self.step_counter += len(states)
        self.progress = 0  # Set to 100 at completion

        print(f"--- Learn Summary ---")
        print(f"Step: {self.step_counter}")
        print(f"Total Loss: {total_loss:.4f}")
        print(f"Policy Loss: {total_policy_loss:.4f}")
        print(f"Value Loss: {total_value_loss:.4f}")
        print(f"Entropy: {total_entropy:.4f}")
        print(f"Raw rewards - Mean: {sum(rewards)/len(rewards):.4f}, Min: {min(rewards):.4f}, Max: {max(rewards):.4f}")

    def _process_state_batch(self, states):
        room_grids = torch.stack([state[0].squeeze(0) for state in states], dim=0)
        map_grids = torch.stack([state[1].squeeze(0) for state in states], dim=0)
        adds = torch.stack([state[2].squeeze(0) for state in states], dim=0)
        return room_grids, map_grids, adds

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_counter': self.step_counter,
            'episode_counter': self.episode_counter
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_counter = checkpoint['step_counter']
        self.episode_counter = checkpoint['episode_counter']
        print(f"Model loaded from {path}")
