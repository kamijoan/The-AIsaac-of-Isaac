import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from math import ceil

class PPOPolicy(nn.Module):
    def __init__(self, room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory, isaacNumber):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(PPOPolicy, self).__init__()
        self.room_shape = room_shape  # [5, 16, 28]
        self.map_shape = map_shape    # [6, 13, 13]
        self.n_critical = n_critical
        self.n_items = n_items
        self.n_entity_memory = n_entity_memory
        self.isaacNumber = isaacNumber
        self.visualData = []

        room_out_channels = 64
        # Room grid processing
        self.room_conv1 = nn.Conv2d(room_shape[0], 16, kernel_size=3, padding=1)
        self.room_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.room_conv3 = nn.Conv2d(32, room_out_channels, kernel_size=3, padding=1)
        room_out_size = room_out_channels * room_shape[1] * room_shape[2]
        self.room_projection = nn.Sequential(
            nn.Linear(room_out_size, 256),
            nn.ReLU(),
        )

        # Map grid processing
        self.map_conv1 = nn.Conv2d(map_shape[0], 16, kernel_size=3, padding=1)
        self.map_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        map_out_size = 32 * map_shape[1] * map_shape[2]
        self.map_projection = nn.Sequential(
            nn.Linear(map_out_size, 256),
            nn.ReLU(),
        )

        # Critical values processing
        self.critical_fc = nn.Sequential(
            nn.Linear(n_critical + n_items, 64),  # Adjusted for input size
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        # Entity memory processing
        self.entity_embed = nn.Linear(13, 64)
        self.entity_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ),
            num_layers=2
        )
        self.entity_pool = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
        )

        # Fusion transformer
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                activation='relu',
                batch_first=True
            ),
            num_layers=1
        )
        self.fusion_pool = nn.Linear(256 * 4, 1024)

        # LSTM layer
        self.lstm = nn.LSTM(1024, 512, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data.fill_(0)
                param.data[start:end].fill_(1)

        # Actor and critic heads
        self.fc1 = nn.Linear(512, 512)
        self.actor = nn.Linear(512, action_size)
        self.critic = nn.Linear(512, 1)

    def forward(self, room_grid, map_grid, additional_values, hidden_state=None):
        # Convert inputs to tensors if they are NumPy arrays
        if isinstance(room_grid, ndarray):
            room_grid = torch.from_numpy(room_grid).float().unsqueeze(0).to(self.device)
        if isinstance(map_grid, ndarray):
            map_grid = torch.from_numpy(map_grid).float().unsqueeze(0).to(self.device)
        if isinstance(additional_values, ndarray):
            additional_values = torch.from_numpy(additional_values).float().unsqueeze(0).to(self.device)

        batch_size = room_grid.size(0)

        # Room grid processing
        x_room = F.relu(self.room_conv1(room_grid))
        x_room = F.relu(self.room_conv2(x_room))
        x_roomV = F.relu(self.room_conv3(x_room))
        x_room = x_roomV.view(batch_size, -1)
        x_room = self.room_projection(x_room)  # [batch_size, 256]

        # Map grid processing
        x_map = F.relu(self.map_conv1(map_grid))
        x_mapV = F.relu(self.map_conv2(x_map))
        x_map = x_mapV.view(batch_size, -1)
        x_map = self.map_projection(x_map)  # [batch_size, 256]

        # Critical values processing
        critical_input = additional_values[:, :self.n_critical + self.n_items]
        x_critical = self.critical_fc(critical_input)  # [batch_size, 256]

        # Entity memory processing
        entities = additional_values[:, self.n_critical + self.n_items:self.n_critical + self.n_items + self.n_entity_memory]
        entities = entities.view(batch_size, 200, 13)  # [batch_size, 200, 13]
        x_entities = F.relu(self.entity_embed(entities))  # [batch_size, 200, 64]
        mask = torch.all(entities == 0, dim=-1)  # [batch_size, 200]
        x_entities = self.entity_transformer(x_entities, src_key_padding_mask=mask)  # [batch_size, 200, 64]
        x_entities = self.entity_pool(torch.mean(x_entities, dim=1))  # [batch_size, 256]

        # Fuse features with transformer
        combined = torch.stack([x_room, x_map, x_critical, x_entities], dim=1)  # [batch_size, 4, 256]
        fused = self.fusion_transformer(combined)  # [batch_size, 4, 256]
        fused = fused.view(batch_size, -1)  # [batch_size, 4 * 256]
        x = self.fusion_pool(fused)  # [batch_size, 1024]
        x = F.relu(x)
        x = x.unsqueeze(1)  # [batch_size, 1, 1024]

        # LSTM processing
        if hidden_state is None:
            h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(self.device)
            c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(self.device)
            hidden_state = (h0, c0)
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        x = lstm_out.squeeze(1)  # [batch_size, 512]

        # Final layers
        x = F.relu(self.fc1(x))
        logits = self.actor(x)
        value = self.critic(x)

        # Store visualization data
        #self.setVisualData([hidden_state[0], hidden_state[1], x_roomV, x_mapV, x])
        self.setVisualData([x])
        return logits, value, hidden_state

    def setVisualData(self, rawData):
        for index, data in enumerate(rawData):
            if len(data.shape) == 4:
                batch_size, channels, viewY, viewX = data.shape
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
    def __init__(self, room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory, isaacNumber, shared_model, learn_lock, rollout_queue):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPOPolicy(room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory, isaacNumber).to(self.device)

        lr = 0.0002
        self.optimizer = torch.optim.Adam([
            {'params': [p for n, p in self.policy.named_parameters() if 'critic' not in n], 'lr': lr},
            {'params': [p for n, p in self.policy.named_parameters() if 'critic' in n], 'lr': lr}
        ])
        self.gamma = 0.99
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.003
        self.max_grad_norm = 0.5
        self.n_steps = 2048
        self.data_chunk = 256

        self.step_counter = 0
        self.episode_counter = 0
        self.progress = 0
        self.entropy = None
        self.probs = None

        self.shared_model = shared_model
        self.learn_lock = learn_lock
        self.action_size = action_size
        self.isaacNumber = isaacNumber
        self.rollout_queue = rollout_queue

        # Initialize rollout storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs_list = []
        self.values_list = []
        self.hidden_states = []
        self.current_hidden_state = None

    def act(self, room_grid, map_grid, additional_values):
        # Pass raw NumPy arrays to forward()
        with torch.no_grad():
            logits, value, hidden_state = self.policy(room_grid, map_grid, additional_values, self.current_hidden_state)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            log_prob = probs.log().gather(1, torch.tensor([[action]], device=self.device))
            self.entropy = -torch.sum(probs * torch.log(probs + 1e-6))
            min_prob = probs.min()
            max_prob = probs.max()
            graphProbs = ((probs - min_prob) / (max_prob - min_prob + 1e-6)) * 100
            self.probs = graphProbs.squeeze().tolist()
        # Store the new hidden state (move to CPU to avoid pickling issues)
        self.current_hidden_state = (hidden_state[0], hidden_state[1])
        # Store rollout data as raw NumPy arrays or CPU tensors
        self.states.append((room_grid, map_grid, additional_values))  # Store raw NumPy arrays
        self.actions.append(action)
        self.log_probs_list.append(log_prob.item())
        self.values_list.append(value.item())
        self.hidden_states.append((hidden_state[0].cpu(), hidden_state[1].cpu()))

        return action

    def sendRollouts(self, done):
        if len(self.states) > self.n_steps:
            if sum(self.rewards) != 0:
                self.rollout_queue.put((self.states, self.actions, self.rewards, self.dones, self.log_probs_list, self.values_list, self.hidden_states))
                print(f"Isaac {self.isaacNumber}: Sent rollout, Episode completed")
            self.states = [self.states[-1]]
            self.actions = []
            self.rewards = []
            self.dones = []
            self.log_probs_list = []
            self.values_list = []
            self.hidden_states = []
            self.current_hidden_state = None if done else self.current_hidden_state

    def run(self):
        while True:
            try:
                rollouts = self.rollout_queue.get(timeout=1.0)
                if len(rollouts) > 1:
                    print("Learn: Grabbed a rollout...")
                    self.learn(rollouts)
            except Exception as e:
                #print(e)
                pass  # Queue empty, continue polling

    def _process_state_batch(self, states, hidden):
        # Convert NumPy arrays to tensors and stack them
        room_grids = torch.stack([torch.from_numpy(state[0]).float() for state in states], dim=0).to(self.device)
        map_grids = torch.stack([torch.from_numpy(state[1]).float() for state in states], dim=0).to(self.device)
        adds = torch.stack([torch.from_numpy(state[2]).float() for state in states], dim=0).to(self.device)

        h_list = [h.squeeze(1) for h, c in hidden]  # [1, 1, hidden_size] -> [1, hidden_size]
        c_list = [c.squeeze(1) for h, c in hidden]  # Same
        h_stacked = torch.stack(h_list, dim=1).to(self.device)  # [1, T, hidden_size]
        c_stacked = torch.stack(c_list, dim=1).to(self.device)  # Same
        hidden_stacked = (h_stacked, c_stacked)
        return room_grids, map_grids, adds, hidden_stacked

    def learn(self, rollouts, n_epochs=8,):
        states, actions, rewards, dones, old_log_probs, values, hidden_states = rollouts

        # Convert lists to tensors where needed (rewards and dones are likely Python lists or NumPy arrays)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        values = torch.tensor(values, device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        hidden_states = [(h.to(self.device), c.to(self.device)) for h, c in hidden_states]

        # Compute next_value for all next_states
        next_hidden_states = hidden_states[1:] + [hidden_states[0]]
        next_states = states[1:]
        states = states[:-1]

        next_value_chunks = []
        for i in range(0, len(next_states), self.data_chunk):
            chunk_states = next_states[i:i + self.data_chunk]
            chunk_hidden = next_hidden_states[i:i + self.data_chunk]
            room_grids, map_grids, adds, hidden = self._process_state_batch(chunk_states, chunk_hidden)
            with torch.no_grad():
                _, chunk_value, _ = self.policy(room_grids, map_grids, adds, hidden)
            next_value_chunks.append(chunk_value.squeeze(-1))
        next_value = torch.cat(next_value_chunks)

        # Compute GAE
        lambda_ = 0.95
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lambda_ * (1 - dones[t]) * gae
            advantages[t] = gae
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Rest of the learning loop
        total_policy_loss, total_value_loss, total_entropy, total_loss, total_batches = 0.0, 0.0, 0.0, 0.0, 0
        grad_sums = {}
        total_data = len(states)
        total_batches = (total_data + self.data_chunk - 1) // self.data_chunk
        total_iterations = n_epochs * total_batches
        current_iteration = 0

        for epoch in range(n_epochs):
            for i in range(0, total_data, self.data_chunk):
                batch_end = min(i + self.data_chunk, total_data)
                batch_indices = slice(i, batch_end)

                self.optimizer.zero_grad()
                batch_states = states[batch_indices]
                batch_hidden_states = hidden_states[batch_indices]
                batch_room_grids, batch_map_grids, batch_add, batch_hidden = self._process_state_batch(batch_states, batch_hidden_states)
                batch_actions = torch.tensor(actions[batch_indices], device=self.device)
                batch_old_log_probs = old_log_probs[batch_indices].squeeze().to(self.device)
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
                for name, param in self.policy.named_parameters():
                    if param.grad is not None:
                        if name not in grad_sums:
                            grad_sums[name] = {'mean': 0.0, 'std': 0.0, 'count': 0}
                        grad_sums[name]['mean'] += param.grad.mean().item()
                        grad_sums[name]['std'] += param.grad.std(unbiased=False).item() if param.grad.numel() > 1 else 0.0
                        grad_sums[name]['count'] += 1
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
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
        self.episode_counter += 1
        self.progress = 0

        with self.learn_lock:
            cpu_state_dict = {key: value.cpu() for key, value in self.policy.state_dict().items()}
            self.shared_model.clear()
            self.shared_model.update(cpu_state_dict)

        print("Episode:", self.episode_counter)
        if self.episode_counter % 5 == 0:
            self.save("F:/isaacPPOModel.pth")

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
        print(f"{self.isaacNumber}. Raw rewards - Mean: {rewards.mean():.4f}, Min: {rewards.min():.4f}, Max: {rewards.max():.4f}")
        print("\n")

    def save(self, path):
        with self.learn_lock:
            cpu_state_dict = {key: value.cpu() for key, value in self.policy.state_dict().items()}
            torch.save({
                'policy_state_dict': cpu_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'step_counter': self.step_counter,
                'episode_counter': self.episode_counter
            }, path)
            print(f"{self.isaacNumber}. Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        with self.learn_lock:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.step_counter = checkpoint['step_counter']
            self.episode_counter = checkpoint['episode_counter']
            print(f"{self.isaacNumber}. Model loaded from {path}")
