import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
#torch.set_printoptions(profile="full")

class PPOPolicy(nn.Module):
    def __init__(self, room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory, isaacNumber, lstm_hidden_size = 512):
        super(PPOPolicy, self).__init__()
        self.room_shape = room_shape  # [14, 16, 28]
        self.map_shape = map_shape    # [6, 13, 13]
        self.n_critical = n_critical
        #self.n_items = n_items
        #self.n_entity_memory = n_entity_memory
        self.isaacNumber = isaacNumber
        self.visualData = []

        room_out_channels = 64
        # Room grid processing
        self.room_conv1 = nn.Conv2d(room_shape[0], 16, kernel_size=3, padding=1)
        self.room_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.room_conv3 = nn.Conv2d(32, room_out_channels, kernel_size=3, padding=1)

        self.room_pool = nn.MaxPool2d(2)
        room_out_size = room_out_channels * 8 * 14  # 16x28 → 8x14 after pooling

        # Map grid processing
        self.map_conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.map_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        map_out_size = 32 * 13 * 13  # No pooling, stays 13x13

        critical_out_size = 256
        # Critical values processing
        self.critical_fc = nn.Sequential(
            nn.Linear(n_critical, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, critical_out_size)
        )


        # Combined features
        total_features = room_out_size + map_out_size + critical_out_size

        self.pre_lstm_fc = nn.Linear(total_features, 1024)  # Reduce to a manageable size

        # LSTM layer
        self.lstm = nn.LSTM(1024, lstm_hidden_size, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                start, end = n // 4, n // 2  # Forget gate bias range
                param.data.fill_(0)
                param.data[start:end].fill_(1)  # Set forget gate bias to 1

        # Actor and critic heads
        self.fc1 = nn.Linear(lstm_hidden_size, 512)
        self.actor = nn.Linear(512, action_size)
        self.critic = nn.Linear(512, 1)

    def forward(self, room_grid, map_grid, additional_values, hidden_state=None):
        batch_size = room_grid.size(0)

        # Room grid processing
        x_room = F.leaky_relu(self.room_conv1(room_grid), 0.1)
        x_room = F.leaky_relu(self.room_conv2(x_room), 0.1)
        x_room = F.leaky_relu(self.room_conv3(x_room), 0.1)
        x_roomV = self.room_pool(x_room)
        x_room = x_roomV.view(batch_size, -1)

        # Map grid processing
        x_map = F.leaky_relu(self.map_conv1(map_grid), 0.1)
        x_mapV = F.leaky_relu(self.map_conv2(x_map), 0.1)
        x_map = x_mapV.view(batch_size, -1)

        # Critical values processing
        x_critical = self.critical_fc(additional_values[:, :self.n_critical])

        x = torch.cat([x_room, x_map, x_critical], dim=1)
        x = F.leaky_relu(self.pre_lstm_fc(x), 0.1)  # Reduce size before LSTM
        x = x.unsqueeze(1)

        # LSTM processing
        if hidden_state is None:
            # Initialize hidden state if not provided
            h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device)
            hidden_state = (h0, c0)
        lstm_out, hidden_state = self.lstm(x, hidden_state)  # [batch_size, 1, lstm_hidden_size]

        # Remove time dimension
        x = lstm_out.squeeze(1)  # [batch_size, lstm_hidden_size]

        # Final layers
        x = F.leaky_relu(self.fc1(x), 0.1)
        logits = self.actor(x)
        value = self.critic(x)

        self.setVisualData([hidden_state[0], hidden_state[1], x_roomV, x_mapV, x])

        return logits, value, hidden_state

    def setVisualData(self, rawData):
        for index, data in enumerate(rawData):
            if len(data.shape) == 4:
                batch_size, channels, viewX, viewY = data.shape
                data_padded = data
            else:
                viewX = 28
                if len(data.shape) == 2:
                    channels = 1
                    batch_size, n = data.shape
                elif len(data.shape) == 3:
                    channels, batch_size, n = data.shape
                if n == 512:
                    viewX = 32
                    viewY = 16
                    data_padded = data
                elif n == 1024:
                    viewX = 32
                    viewY = 32
                    data_padded = data
                else:
                    viewY = ceil(n / viewX)
                    target_n = viewY * viewX
                    if n < target_n:
                        padding = target_n - n
                        data_padded = torch.nn.functional.pad(data, (0, padding))
                    else:
                        data_padded = data


            if len(self.visualData) < index + 1:
                self.visualData.append(data_padded.view(batch_size, channels, viewY, viewX))
            else:
                self.visualData[index] = data_padded.view(batch_size, channels, viewY, viewX)

class PPOAgent:
    def __init__(self, room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory, isaacNumber, clip_param=0.2, value_loss_coef=0.5, gamma=0.9, max_grad_norm=0.5, n_steps=2048):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPOPolicy(room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory, isaacNumber).to(self.device)
        # Share the policy's parameters across processes
        self.policy.share_memory()  # Makes the model's parameters shared in memory

        lr = 0.0002 * isaacNumber
        entropy_coef = 0.003 * isaacNumber

        self.optimizer = torch.optim.Adam([
            {'params': [p for n, p in self.policy.named_parameters() if 'critic' not in n], 'lr': lr},
            {'params': [p for n, p in self.policy.named_parameters() if 'critic' in n], 'lr': lr}
        ])
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.step_counter = 0
        self.episode_counter = 0
        self.progress = 0
        self.entropy = 0
        self.probs = 0
        self.action_size = action_size
        self.isaacNumber = isaacNumber

    def act(self, state, hidden_state=None):
        room_grid, map_grid, additional_values = state
        with torch.no_grad():
            logits, value, hidden_state = self.policy(room_grid, map_grid, additional_values, hidden_state)
            self.probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(self.probs, 1).item()
            log_prob = self.probs.log().gather(1, torch.tensor([[action]], device=self.device))
        return action, log_prob, value, hidden_state

    def _process_state_batch(self, states, hidden):
        room_grids = torch.stack([state[0].squeeze(0) for state in states], dim=0).to(self.device)
        map_grids = torch.stack([state[1].squeeze(0) for state in states], dim=0).to(self.device)
        adds = torch.stack([state[2].squeeze(0) for state in states], dim=0).to(self.device)

        h_list = [h.squeeze(1) for h, c in hidden]  # [1, 1, hidden_size] -> [1, hidden_size]
        c_list = [c.squeeze(1) for h, c in hidden]  # Same
        h_stacked = torch.stack(h_list, dim=1).to(self.device)  # [1, T, hidden_size]
        c_stacked = torch.stack(c_list, dim=1).to(self.device)  # Same
        hidden_stacked = (h_stacked, c_stacked)

        return room_grids, map_grids, adds, hidden_stacked

    def learn(self, rollouts, n_epochs=8, batch_size=128):
        states, actions, rewards, dones, old_log_probs, values, hidden_states = rollouts
        next_hidden_states = hidden_states[1:]
        next_hidden_states.append(hidden_states[0])
        next_states = states[1:]
        states = states[:-1]
        values = torch.cat(values).squeeze(-1)
        # Chunk next_value computation
        next_value_chunks = []
        for i in range(0, len(next_states), batch_size):
            chunk_states = next_states[i:i + batch_size]
            chunk_hidden = next_hidden_states[i:i + batch_size]
            room_grids, map_grids, adds, hidden = self._process_state_batch(chunk_states, chunk_hidden)
            with torch.no_grad():
                _, chunk_value, _ = self.policy(room_grids, map_grids, adds, hidden)
            next_value_chunks.append(chunk_value.squeeze(-1))
            del room_grids, map_grids, adds, hidden
        next_value = torch.cat(next_value_chunks)
        rewards = torch.tensor(rewards, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        advantages = rewards - values + next_value * (1 - dones) * self.gamma
        print(f"{self.isaacNumber}. Advantages not normalized - Mean: {advantages.mean():.4f}, Std: {advantages.std():.4f}")
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss, total_value_loss, total_entropy, total_loss, total_batches = 0.0, 0.0, 0.0, 0.0, 0
        grad_sums = {}

        # Calculate total iterations
        total_data = len(states)
        total_batches = (total_data + batch_size - 1) // batch_size  # Ceiling division
        total_iterations = n_epochs * total_batches
        current_iteration = 0

        for epoch in range(n_epochs):
            for i in range(0, total_data, batch_size):
                batch_end = min(i + batch_size, total_data)
                batch_indices = slice(i, batch_end)

                self.optimizer.zero_grad()
                batch_states = states[batch_indices]
                batch_hidden_states = hidden_states[batch_indices]
                batch_room_grids, batch_map_grids, batch_add, batch_hidden = self._process_state_batch(batch_states, batch_hidden_states)
                batch_actions = torch.tensor(actions[batch_indices], device=self.device)
                batch_old_log_probs = torch.stack(old_log_probs[batch_indices]).squeeze()
                batch_advantages = advantages[batch_indices]
                batch_rewards = rewards[batch_indices]

                logits, value, _ = self.policy(batch_room_grids, batch_map_grids, batch_add, batch_hidden)

                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(batch_actions)
                self.entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value.squeeze(-1), batch_rewards, reduction='mean')
                entropy_loss = -self.entropy_coef * self.entropy
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * self.entropy

                loss.backward()
                for name, param in self.policy.named_parameters():  # Changed from self.network to self.policy
                    if param.grad is not None:
                        if name not in grad_sums:
                            grad_sums[name] = {'mean': 0.0, 'std': 0.0, 'count': 0}
                        grad_sums[name]['mean'] += param.grad.mean().item()
                        grad_sums[name]['std'] += param.grad.std(unbiased=False).item() if param.grad.numel() > 1 else 0.0
                        grad_sums[name]['count'] += 1
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # Changed from self.network to self.policy
                self.optimizer.step()

                batch_size_actual = batch_actions.size(0)
                total_policy_loss += policy_loss.item() * batch_size_actual
                total_value_loss += value_loss.item() * batch_size_actual
                total_entropy += self.entropy.item() * batch_size_actual
                total_loss += loss.item() * batch_size_actual
                total_batches += batch_size_actual

                current_iteration += 1
                self.progress = (current_iteration / total_iterations) * 100

        total_policy_loss /= total_batches
        total_value_loss /= total_batches
        total_entropy /= total_batches
        total_loss /= total_batches

        self.entropy = total_entropy
        self.step_counter += len(states)
        self.progress = 0

        print(f"=== Isaac {self.isaacNumber} Gradient Summary (Averaged) ===")
        for name, stats in grad_sums.items():
            mean = stats['mean'] / stats['count']
            std = stats['std'] / stats['count'] if stats['count'] > 0 else 0.0
            print(f"{self.isaacNumber}. {name}: mean={mean:.6f}, std={std:.6f}")

        print(f"--- Isaac {self.isaacNumber} Learn Summary ---")
        print(f"{self.isaacNumber}. Step: {self.step_counter}")
        print(f"{self.isaacNumber}. Policy Loss: {total_policy_loss:.4f} | Should be negative.")
        print(f"{self.isaacNumber}. Value Loss: {total_value_loss:.4f} | Should be small and positive, critic output.")
        print(f"{self.isaacNumber}. Total Loss: {total_loss:.4f} | Should be small and negative. Policy loss+Value loss * stuff.")
        print(f"{self.isaacNumber}. Entropy: {total_entropy:.4f} | Action size: {self.action_size}")
        print(f"{self.isaacNumber}. Raw rewards - Mean: {sum(rewards)/len(rewards):.4f}, Min: {min(rewards):.4f}, Max: {max(rewards):.4f}")

        print("\n")
        torch.cuda.empty_cache()

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_counter': self.step_counter,
            'episode_counter': self.episode_counter
        }, path)
        print(f"{self.isaacNumber}. Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_counter = checkpoint['step_counter']
        self.episode_counter = checkpoint['episode_counter']
        print(f"{self.isaacNumber}. Model loaded from {path}")
