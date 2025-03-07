from pymem import Pymem
from time import sleep
from keyboard import is_pressed
from threading import Thread
import torch,cv2,time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label,gaussian_filter
from torch_geometric.nn import GCNConv
import math,random,os
torch.set_printoptions(profile="full")


class PPOPolicy(nn.Module):
    def __init__(self, room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory):
        super(PPOPolicy, self).__init__()
        self.room_shape = room_shape
        self.map_shape = map_shape
        self.n_critical = n_critical
        self.n_items = n_items
        self.n_entity_memory = n_entity_memory
        self.visualData = None

        # Single convolution for room_grid without ReLU to preserve negative values
        self.room_conv = nn.Conv2d(room_shape[0], 64, kernel_size=1)  # 1x1 conv, preserves 16x28

        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.map_conv_base = nn.Conv2d(map_shape[0], 16, kernel_size=1)  # 1x1 conv, preserves 13x13

        map_conv_out_size = 16 * map_shape[1] * map_shape[2]  # 16 * 13 * 13 = 2704

        self.critical_fc = nn.Sequential(
            nn.Linear(n_critical, 512),
            nn.Linear(512, 256)
        )
        self.items_fc = nn.Sequential(
            nn.Linear(n_items, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.memory_fc = nn.Sequential(
            nn.Linear(n_entity_memory, 512),
            nn.Linear(512, 256)
        )
        self.memory_attention = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        room_conv_out_size = 64 * room_shape[1] * room_shape[2]  # 64 * 16 * 28 = 28672
        total_features = room_conv_out_size + map_conv_out_size + 256 + 64 + 256
        self.fc1 = nn.Linear(total_features, 512)
        self.actor = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, action_size))
        self.critic = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 1))

        self._initialize_weights()

    def forward(self, room_grid, map_grid, additional_values, prev_map_grid=None):
        batch_size = room_grid.size(0)

        # Process room_grid as a single input
        x_room = self.room_conv(room_grid)  # Shape: [batch_size, 64, 16, 28], preserves negative values

        attn = self.attention(x_room)  # Shape: [batch_size, 1, 16, 28]
        # Normalize attention to enhance effect
        attn = attn / (attn.max() + 1e-8)  # Normalize to [0, 1] with max=1
        x_room_attended = x_room * attn  # Apply attention, preserving negatives

        x_room = x_room_attended.view(batch_size, -1)  # Flatten: [batch_size, 64 * 16 * 28]

        x_map = self.map_conv_base(map_grid)  # Shape: [batch_size, 16, 13, 13]
        if prev_map_grid is not None:
            map_diff = self.map_conv_base(map_grid - prev_map_grid)
            x_map = x_map + map_diff
        x_map = x_map.view(batch_size, -1)  # Flatten: [batch_size, 16 * 13 * 13]

        crit_end = self.n_critical
        item_end = crit_end + self.n_items
        x_critical = self.critical_fc(additional_values[:, :crit_end])
        x_items = self.items_fc(additional_values[:, crit_end:item_end])
        x_memory_raw = self.memory_fc(additional_values[:, item_end:])
        x_memory = x_memory_raw * self.memory_attention(x_memory_raw)

        x = torch.cat([x_room, x_map, x_critical, x_items, x_memory], dim=1)
        x = self.fc1(x)
        value = self.critic(x)
        logits = self.actor(x)

        # Updated visualData
        self.visualData = (
            room_grid,  # Raw input for reference, shape [batch_size, 13, 16, 28]
            x_room.view(batch_size, 64, 16, 28),  # Pre-attention room features, including negatives
            attn,  # Attention map, shape [batch_size, 1, 16, 28]
            x_room_attended.view(batch_size, 64, 16, 28),  # Post-attention room features, including negatives
            x_map.view(batch_size, 16, 13, 13)  # Map features, 13x13
        )

        return logits, value

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

class PPOAgent:
    def __init__(self, room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory, lr=0.0003, gamma=0.99, clip_param=0.5, value_loss_coef=0.5, entropy_coef=0.001, max_grad_norm=0.5, n_steps=4096):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPOPolicy(room_shape, map_shape, action_size, n_critical, n_items, n_entity_memory).to(self.device)
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
        self.episode_counter = 0
        self.progress = 0
        self.entropy = 0
        self.probs = 0

    def act(self, state):
        room_grid, map_grid, additional_values = state
        self.last_room_grid = room_grid.clone().to('cpu')
        self.last_map_grid = map_grid.clone().to('cpu')
        room_grid = room_grid.to(self.device)
        map_grid = map_grid.to(self.device)
        additional_values = additional_values.to(self.device)

        prev_map_grid = self.last_map_grid.to(self.device) if hasattr(self, 'last_map_grid') else None
        with torch.no_grad():
            logits, value = self.policy(room_grid, map_grid, additional_values, prev_map_grid)
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

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        total_batches = 0

        # Gradient accumulation settings
        accumulation_steps = 4  # Number of mini-batches to accumulate gradients over
        effective_batch_size = batch_size  # 128, your original batch size
        mini_batch_size = effective_batch_size // accumulation_steps  # 32

        for epoch in range(n_epochs):
            total_batches_per_epoch = len(states) // effective_batch_size + (1 if len(states) % effective_batch_size != 0 else 0)
            self.optimizer.zero_grad()  # Clear gradients at the start of each epoch
            for batch_idx, i in enumerate(range(0, len(states), mini_batch_size)):
                self.progress = ((epoch * total_batches_per_epoch + batch_idx // accumulation_steps) / (n_epochs * total_batches_per_epoch)) * 100
                batch_indices = slice(i, min(i + mini_batch_size, len(states)))
                batch_states = self._process_state_batch(states[batch_indices])
                batch_room_grids, batch_map_grids, batch_adds = [t.to(self.device) for t in batch_states]
                batch_actions = torch.tensor(actions[batch_indices], device=self.device)
                batch_old_log_probs = torch.stack(old_log_probs[batch_indices]).squeeze()  # From act()
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Compute current policy outputs
                logits, value = self.policy(batch_room_grids, batch_map_grids, batch_adds)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(batch_actions)  # New log probs for current policy
                self.entropy = dist.entropy().mean()

                # Use old log probs from act() for ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value.squeeze(-1), batch_returns, reduction='mean')
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * self.entropy
                loss = loss / accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if (i // mini_batch_size + 1) % accumulation_steps == 0:
                    self.optimizer.step()

                batch_size_actual = batch_actions.size(0)
                total_policy_loss += policy_loss.item() * batch_size_actual
                total_value_loss += value_loss.item() * batch_size_actual
                total_entropy += self.entropy.item() * batch_size_actual
                total_loss += loss.item() * batch_size_actual
                total_batches += batch_size_actual

        total_policy_loss /= total_batches
        total_value_loss /= total_batches
        total_entropy /= total_batches
        total_loss /= total_batches

        self.entropy = total_entropy
        self.step_counter += len(states)
        self.progress = 0

        print(f"=== Learn Summary ===")
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
########################################################################################################################################################################

def update_actionStates():
    for action, key in key_map.items():
        actionStates[action] = int(is_pressed(key))  # 1 if pressed, else 0

def deduce_action():
    global previous_state

    if actionStates == previous_state:
        return action_size-1  # No change, maintain current inputs

    pressed_keys = [key for key in actionStates if actionStates[key] == 1 and previous_state[key] == 0]
    released_keys = [key for key in actionStates if actionStates[key] == 0 and previous_state[key] == 1]

    previous_state = actionStates.copy()  # Update state tracking

    if pressed_keys:
        return pressed_keys[0]  # Return first pressed key as action
    elif released_keys:
        return released_keys[0]  # Return first released key as action

    return action_size-1  # Default to no action

def execute_action(action):
    global actionStates
    if action < action_size-2:
        if action in opposite_actions:
            opposite = opposite_actions[action]
            actionStates[opposite] = 0
        # Allow only one shooting direction at a time
        elif action in shooting_actions and new_value == 1:
            for shoot_action in shooting_actions:
                actionStates[shoot_action] = 0
        # Apply the new action state
        actionStates[action] = 1
    elif action == action_size-1:
        for k in actionStates:
            actionStates[k] = 0


    # Send full state to Lua
    msg = " ".join(f"{k} {v}" for k, v in actionStates.items())
    with open("F:/IsaacInputs.txt", "w") as f:
        f.write(msg)
    while True:
        with open("F:/IsaacResponse.txt", "r") as file:
            if msg == file.read().strip():
                break  # Loop until msg is confirmed back

    return list(actionStates.values())  # Return full state list

def readGameData():
    global last_data, playerData, totalHP, itemArray, dataValues
    readFileLoop = True
    while readFileLoop:
        try:
            with open("F:/IsaacData.txt", "r") as file:
                data = file.read().strip()
            for item in data.split(","):
                if "=" in item:
                    key, value = item.split("=")
                    if key == "items":
                        try:
                            playerData[key] = list(map(int, value.strip("[]").split("|")))
                        except:
                            playerData["items"] = []
                    elif key in playerNormalization:
                        playerData[key] = float(value) / playerNormalization[key]
                        if playerData[key] > 1 or playerData[key] < -1:
                            print("Normalized Stat problem:",key,playerData[key])
                    elif key in playerData:
                        playerData[key] = int(value)

            totalHP = playerData["hp"]+playerData["soul_hp"]+playerData["black_hp"]+playerData["rotten_hp"]+playerData["bone_hp"]+playerData["eternal_hp"]+playerData["extra_lives"]
            dataValues = [v for k, v in playerData.items() if isinstance(v, (float))] #ignores lists
            readFileLoop = False
        except:
            pass

room_x_min, room_y_min = 20, 100
room_x_max, room_y_max = 1140, 740

def playerHeatmap(player_x, player_y, sigma=.5):
    overlay = np.zeros((16, 28), dtype=np.float32)

    # Normalize position
    norm_x = (player_x - room_x_min) / (room_x_max - room_x_min)
    norm_y = (player_y - room_y_min) / (room_y_max - room_y_min)
    # Convert to grid index
    grid_x = int(norm_x * 28)  # Ensure edges align
    grid_y = int(norm_y * 16)
    # Ensure indices are within bounds
    grid_x = np.clip(grid_x, 0, 27)
    grid_y = np.clip(grid_y, 0, 15)
    # Set the player's position on the heatmap
    overlay[grid_y, grid_x] = 1.0
    # Apply blur
    overlay = gaussian_filter(overlay, sigma=sigma)
    overlay = overlay / overlay.max() if overlay.max() > 0 else overlay
    return overlay

def create_entity_heatmaps(entitiesList, room_x_min=40, room_y_min=100, room_x_max=1100, room_y_max=640, numMaps=10):
    heatmaps = {i: np.zeros((16, 28), dtype=np.float32) for i in range(1, numMaps)}  # 9 heatmaps for types 1-9

    # Define sigma values per entity type
    sigma_values = {1:.5, 2:1, 3:.25, 4:.25, 5:.25, 6:.25, 7:.5, 8:.5, 9:.25, 10:.25}

    # Room width & height based on min/max positions
    room_width = room_x_max - room_x_min
    room_height = room_y_max - room_y_min

    for entity in entitiesList:
        entity_type, _, xpos, ypos, *_ = entity  # Extract type and position

        # Denormalize values
        entity_type = int(entity_type * 10)  # Convert back to 1-8
        entity_type = 9 if entity_type == 10 else entity_type

        xpos *= entitiesNormalization[2]
        ypos *= entitiesNormalization[3]

        if entity_type not in heatmaps:
            print("Entity heatmap problem:", entity_type, entity)
            continue  # Ignore types outside 1-8 range

        # Adjust for position offset
        norm_x = (xpos - room_x_min) / room_width
        norm_y = (ypos - room_y_min) / room_height
        # Convert to grid indices (Fix: Use *28 and *16)
        grid_x = int(norm_x * 28)
        grid_y = int(norm_y * 16)
        # Ensure within bounds
        grid_x = np.clip(grid_x, 0, 27)
        grid_y = np.clip(grid_y, 0, 15)
        # Apply influence
        heatmaps[entity_type][grid_y, grid_x] = 1.0

    # Apply Gaussian blur to each heatmap
    for i in range(1, numMaps):
        heatmaps[i] = gaussian_filter(heatmaps[i], sigma=sigma_values[i])
        # Normalize heatmap values (0-1)
        max_value = heatmaps[i].max()
        if max_value > 0:
            heatmaps[i] /= max_value
    # Stack into (8,16,28) tensor
    final_tensor = np.stack([heatmaps[i] for i in range(1, numMaps)], axis=0)

    return final_tensor  # Shape: (8,16,28)

def hsv_to_bgr(h, s, v):
    """Convert HSV values (0-360, 0-255, 0-255) to BGR color for OpenCV."""
    hsv_color = np.uint8([[[h // 2, s, v]]])  # OpenCV scales hue 0-180
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr_color)  # Convert to (B, G, R)

def drawOverlay():
    visualDataIndex = timer = 0
    lastVisualData = None
    while True:
        timer += 1
        sleep(1/60)
        overlay = np.zeros((400,250,3), dtype=np.uint8)
        tile_size = 4
        try:
            for y in range(roomY):  # Loop through grid height
                for x in range(roomX):  # Loop through grid width
                    tile = roomGrid_normalized[y][x]  # Get tile data
                    if tile[0] > 0:  # Draw only non-zero tiles
                        top_left = ((x * tile_size) + 2, (y * tile_size) + 10)
                        bottom_right = (((x + 1) * tile_size) + 2, ((y + 1) * tile_size) + 10)
                        hue = int(tile[0] * 180)
                        #print(tile)
                        color = hsv_to_bgr(hue, 200, int(50 + 160 * (tile[1])))
                        # Draw rectangle
                        cv2.rectangle(overlay, top_left, bottom_right, color, -1)

            for y in range(floorGrid_normalized.shape[0]):  # Iterate over rows (13)
                for x in range(floorGrid_normalized.shape[1]):  # Iterate over columns (13)
                    room_number, room_id, room_type, visited, cleared, current = floorGrid_normalized[y, x]  # Unpack values
                    if room_id > 0:
                        top_left = ((x * tile_size) + 120, (y * tile_size) + 10)
                        bottom_right = (((x + 1) * tile_size) + 120, ((y + 1) * tile_size) + 10)
                        if visited == 1 or cleared == 1:
                            outline = -1
                        else:
                            outline = 1
                        multiplier = 2 if current == 1 else 1.5
                        hue = int(room_type * 180)
                        color = hsv_to_bgr(hue, 150, 100*multiplier)

                        cv2.rectangle(overlay, top_left, bottom_right, color, outline)
            cv2.circle(overlay, (int(playerData["x"]*120), int(playerData["y"]*120)), 2, (0, 255, 0), -1)  # player display
            for entity in entitiesList:
                hue = int(entity[1] * 180)
                color = hsv_to_bgr(hue, 150, 150)
                cv2.circle(overlay, (int(entity[2]*120), int(entity[3]*120)), 1, color, -1) #entity display

            if len(targets) > 0:
                for door in targets:
                    x,y = door[2]
                    cv2.circle(overlay, (int(x/10),int(y/10)), 1, (255, 0, 0), -1)  # player display

            probs_list_values = agent.probs.squeeze().cpu().tolist()
            min_prob = min(probs_list_values)
            max_prob = max(probs_list_values)
            graphProbs = [(p - min_prob) / (max_prob - min_prob + 1e-6) * 100 for p in probs_list_values]
            entropy = -torch.sum(agent.probs * torch.log(agent.probs + 1e-6))
            max_entropy = np.log(action_size)
            randomness = (entropy / max_entropy) * 100
            randomness = min(max(randomness, 0), 100)

            #extra text
            reward_color = (0, 255, 0) if reward >= 0 else (0, 0, 255)  # Green if positive, Red if negative
            reward_color = (255,255,255) if reward == 0.0 else reward_color
            cv2.putText(overlay, f"Total Reward: {total_reward:.2f}, Reward:", (2, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            (text_width, _), _ = cv2.getTextSize(f"Total Reward: {total_reward:.2f}, Reward:", cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.putText(overlay, f"{reward:+.3f}", (2 + text_width, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, reward_color, 1)
            cv2.putText(overlay,f"Reset in {resetTimer+1-step_count} | Randomness: {randomness:.2f}%",(2,100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            #cv2.putText(overlay,f"{str(keyboardKeys)}",(2,110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(overlay,f"Episode: {agent.episode_counter}",(2,110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            bar_width = 4  # Width of each bard
            gap = 10  # Gap between bars
            max_value = max(graphProbs)  # Maximum value for scaling
            num_actions = len(graphProbs)  # Number of actions
            #actions = ["A", "D", "W", "S", "sL","sR","sU","sD","Bm","It","Cr","Dr","-","X"]  # Corresponding action letters
            actions = ["A", "D", "W", "S", "X", "-"]  # Corresponding action letters
            # Draw the bars
            if max_value != 0:
                for i, (value, letter) in enumerate(zip(graphProbs, actions)):
                    x1 = i * (bar_width + gap) + 3  # X-coordinate of the bar
                    y1 = 176  # Bottom of the bar (fixed)
                    bar_height = int((value / max_value) * 66)  # Correct height scaling
                    x2 = x1 + bar_width
                    y2 = y1 - bar_height  # Move the top of the bar upwards

                    # Draw the bar
                    color = (int(1*value), int(2*value), int(1*value))  # Green color for the bars
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Draw each bar
                    # Draw the letter below the bar
                    text_x = x1 + bar_width // 4  # Center the letter under the bar
                    text_y = y1 + 10  # Place slightly below the bar
                    cv2.putText(overlay, letter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.rectangle(overlay, (5,190),(5+int((len(states) / agent.n_steps) * 180),200), (200,200,200), -1)
            cv2.rectangle(overlay, (5,190),(5+int((agent.progress / 100) * 180),200), (50,200,50), -1)


            if is_pressed('7') and visualDataIndex > 0 and timer > 5:
                visualDataIndex -= 1
                timer = 0
                print("Graph Index:",visualDataIndex)
            if is_pressed('9') and visualDataIndex < len(agent.policy.visualData)-1 and timer > 5:
                visualDataIndex += 1
                timer = 0
                print("Graph Index:",visualDataIndex)

            if timer > 60:
                visualData = agent.policy.visualData[visualDataIndex].detach().cpu().clone()
                timer = 0
                lastVisualData = visualData
            else:
                visualData = lastVisualData
            if visualData is not None:
                if len(visualData.shape) == 4:
                    graph1 = visualData[0, 0, :, :].contiguous().to("cpu", non_blocking=True).numpy()  # Extract from first channel
                elif len(visualData.shape) == 2:
                    graph1 = visualData.contiguous().to("cpu", non_blocking=True).numpy()  # Extract from first channel
                else:
                    raise ValueError(f"Unexpected number of channels: {visualData.shape}")

                attn_normalized = cv2.normalize(graph1, None, 0, 200, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                attn_colored = cv2.applyColorMap(attn_normalized, cv2.COLORMAP_JET)
                attn_resized = cv2.resize(attn_colored, (graph1.shape[1]*6, graph1.shape[0]*6), interpolation=cv2.INTER_NEAREST)
                h_attn, w_attn, _ = attn_resized.shape
                overlay[205:205+h_attn, 5:5+w_attn] = attn_resized

            cv2.imshow("Overlay", overlay)
            cv2.waitKey(1)
        except Exception as e:
            sleep(.1)
            print("overlay:",e)

playerNormalization = {
    "x": 1200,
    "y": 1200,
    "vx": 30,
    "vy": 30,
    "hp": 24,  # Max red hearts
    "max_hp": 24,
    "soul_hp": 24,
    "black_hp": 24,
    "rotten_hp": 24,
    "bone_hp": 6,
    "eternal_hp": 2,
    "extra_lives": 10,  # Based on Dead Cat, Lazarus, etc.
    "coins": 99,
    "bombs": 99,
    "keys": 99,
    "golden_bomb": 1,
    "golden_key": 1,
    "active1": 733,  # Max item ID in Repentance
    "charge1": 12,  # Most active items max out at 12 charge
    "full_charge1": 1,
    "active2": 733,
    "charge2": 12,
    "full_charge2": 1,
    "trinket1": 198,  # Max trinket ID
    "trinket2": 198,
    "damage": 100,  # Extremely high damage (Soy Milk is 0.5, Brimstone is ~10)
    "fire_rate": 100,  # Higher = slower (default 10, Soy Milk ~2)
    "shot_speed": 2,
    "range": 500,
    "luck": 10,
    "speed": 2,
    "card": 97,  # Max card ID
    "pill": 13,  # Max pill ID
    "alive_enemies": 100,
    "room_type": 30,
    "first_visit": 1,
    "stage": 20
}

entitiesNormalization = {
    0: 10,       # Category (1 = enemy, 2 = bomb...)
    1: 1000,     # Entity id
    2: 1200, # X pos
    3: 1200, # Y pos
    4: 30,  # X vel
    5: 30,  # Y vel
    6: 10000,  # HP, explosion damage, coin value, etc.
    7: 10000,   # isInvincible, radius multiplier, etc.
    8: 10000,   # Collision damage, price, scale, etc.
    9: 10000,   # Size, scale, etc.
    10:100000000000000   # Flags,wtf #17592186044416
}

playerData = {key:0 for key in playerNormalization}
playerData["items"] = []
playerData["time_counter"] = 0
totalHP = playerData["hp"]+playerData["soul_hp"]+playerData["black_hp"]+playerData["rotten_hp"]+playerData["bone_hp"]+playerData["eternal_hp"]+playerData["extra_lives"]

print("Running...")
pm = Pymem('isaac-ng.exe')


actionStates = {i: 0 for i in range(12)}
previous_state = actionStates.copy()  # Store last known states
opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}
shooting_actions = {4, 5, 6, 7}  # Only one can be active
#map only for manual teaching
key_map = {0: 'a',  1: 'd',  2: 'w',  3: 's',  # Movement: Left, Right, Up, Down
           4: 'j',  5: 'l',  6: 'i',  7: 'k',  # Shooting: Left, Right, Up, Down
           8: 't',  9: 'f', 10: 'g', 11: 'h'}   # Bomb, Item, Card, Drop

itemArray = np.zeros(50, dtype=np.float32)
keyboardKeys = list(actionStates.values())
reset,done = False,True
lenEntitiesMemory = 200
numEntityValues = 11
num_additional_values = len(keyboardKeys)+len(playerNormalization)+len(itemArray)+(lenEntitiesMemory*numEntityValues)
action_size = 6

emptyRoomTensor = torch.empty((1,3,16,28), dtype=torch.float32, device="cuda")
emptyFloorTensor = torch.empty((1,6,13,13), dtype=torch.float32, device="cuda")
emptyPlayerTensor = torch.empty((1,1,16,28), dtype=torch.float32, device="cuda")
emptyEntitiesTensor = torch.empty((1,9,16,28), dtype=torch.float32, device="cuda")
emptyAdditionalValues = torch.empty((1,num_additional_values), dtype=torch.float32, device="cuda")
final_state = torch.empty((1,13,16,28), dtype=torch.float32, device="cuda")

overlay_thread = Thread(target=drawOverlay, daemon=True)
overlay_thread.start()


manualLearning = False
manualTesting = False


# PPO rollout storage
states = []
actions = []
rewards = []
next_states = []
dones = []
log_probs_list = []
values_list = []

#effect ids, still in progress, removed is ignored, allowed just doesn't get printed.
removedEffects = {2,3,4,5,7,11,12,13,14,15,16,17,20,21,27,33,38,43,58,59,63,64,65,66,68,79,86,99,133,146}
allowedEffects = {1,6,22,23,24,25,26,34,44,45,46,50,57,61,62}

# The loop
while True:
    modelBroken = False
    # Randomize hyperparameters
    """lr = random.uniform(0.0001, 0.001)  # Learning rate range
    gamma = random.choice([0.9, 0.95, 0.99])  # Discount factor options
    clip_param = random.uniform(0.15, 0.25)  # Clipping parameter range
    value_loss_coef = random.uniform(0.5, 2.0)  # Value loss coefficient range
    entropy_coef = random.uniform(0.001, 0.1)  # Entropy coefficient range
    print("lr=",lr,"gamma=",gamma,"clip_param=",clip_param,"value_loss_coef=",value_loss_coef,"entropy_coef=",entropy_coef)
    agent = PPOAgent(
        room_shape=(13, 16, 28),
        map_shape=(6, 13, 13),
        action_size=action_size,
        n_critical=len(keyboardKeys) + len(playerNormalization),
        n_entity_memory=len(itemArray) + (lenEntitiesMemory * numEntityValues),
        lr=lr,
        gamma=gamma,
        clip_param=clip_param,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef
    )"""
    agent = PPOAgent(room_shape=(13, 16, 28),map_shape=(6, 13, 13),action_size=action_size,n_critical=len(keyboardKeys) + len(playerNormalization),n_items=len(itemArray),n_entity_memory=(lenEntitiesMemory * numEntityValues))

    loadOnce = True
    if loadOnce:
        if not manualLearning:
            try:
                agent.load("F:/isaac_ppo_model.pth")
            except Exception as e:
                print("No model found to load...", e)
                try:
                    agent.load("F:/isaac_ppo_model_backup.pth")
                except Exception as e:
                    print("No model found to load...", e)
        else:
            try:
                agent.load("F:/isaac_manual_training.pth")
            except:
                print("No manual model found to load...")


        with open("F:/IsaacInputs.txt", "w") as f:
            pass
        loadOnce = False

    batchTotalRewards = []
    while not modelBroken: #or agent.episode_counter < 500 #Exit loop and randomize parameters again
        sleep(1/60)
        readGameData()
        currentFloor = playerData["stage"]

        if playerData["time_counter"] > 1 and not reset and not manualTesting:
            with open("F:/IsaacInputs.txt", "w") as f:
                f.write("reset")
            reset = True
            sleep(0.2)

        elif playerData["time_counter"] <= 1 or manualTesting:
            for k in actionStates:
                actionStates[k] = 0
            msg = " ".join(f"{k} {v}" for k, v in actionStates.items())
            with open("F:/IsaacInputs.txt", "w") as f:
                f.write(msg)
            reset = done = False

            # Initialize run
            readGameData()
            previousX = playerData["x"]
            previousY = playerData["y"]
            currentFloor = playerData["stage"]
            itemArray = np.zeros(50, dtype=np.float32)
            floorGrid_tensor = emptyFloorTensor
            additional_values_torch = emptyAdditionalValues

            wallTiles = set()
            doorTiles = set()
            trackedTears = set()

            itemsSum = len(playerData["items"])

            total_reward = previousItemsSum = totalEnemyHP = enemyDamage = 0

            door_min_distances = {}
            door_reward_emptied = {}

            previousHP, previousItemsSum, previousEnemyHP = totalHP, itemsSum, totalEnemyHP

            lastRoom = 84
            roomHP = previousHP = totalHP
            entitiesListFull = np.zeros((lenEntitiesMemory, numEntityValues), dtype=np.float32)

            visited_rooms = last_visited_rooms = 1
            step_count = enemyDamage = lastEnemyDamage = punishX = punishY = 0
            stepsInRoom = 0

            sleep(1)
            print("Enter Episode")
            resetTimer = 500
            while not done:

                step_count += 1

                action, log_prob, value = agent.act((final_state, floorGrid_tensor, additional_values_torch))
                log_probs_list.append(log_prob)
                values_list.append(value)

                if not manualTesting and not manualLearning:
                    keyboardKeys = execute_action(action)
                elif manualLearning:
                    action = deduce_action()
                    update_actionStates()
                    keyboardKeys = list(actionStates.values())

                # Update game state (your existing logic)
                readFileLoop = True
                while readFileLoop:
                    try:
                        with open("F:/IsaacTileData.txt", "r") as file:
                            first_line = file.readline()
                            tile_data = file.readlines()
                            room, roomX, roomY = map(int, first_line.split(","))
                            roomGrid = np.array([list(map(int, line.strip().split(","))) for line in tile_data], dtype=np.int32)
                            roomGrid = roomGrid.reshape((roomY, roomX, 3))
                            readFileLoop = False
                    except:
                        pass

                roomGrid_normalized = roomGrid / np.array([27, 5, 1000], dtype=np.float32)

                out_of_bounds = (roomGrid_normalized < -1) | (roomGrid_normalized > 1)
                if np.any(out_of_bounds):
                    print("RoomGrid Normalization Problem:", roomGrid_normalized[out_of_bounds])
                    print("grid",roomGrid)

                roomGrid_resized = np.full((16, 28, 3), 0, dtype=np.float32)
                roomGrid_resized[:roomY, :roomX, :] = roomGrid_normalized
                empty_mask = np.all(roomGrid_resized == [0, 0, 0], axis=-1)

                labeled_grid, num_features = label(empty_mask)
                for i in range(1, num_features + 1):
                    block_mask = labeled_grid == i  # Boolean mask for the current block
                    if np.any(block_mask[0, :]) or np.any(block_mask[-1, :]) or np.any(block_mask[:, 0]) or np.any(block_mask[:, -1]):
                        roomGrid_resized[block_mask] = [-.2, -.2, -.2]

                roomGrid_resized = np.transpose(roomGrid_resized, (2, 0, 1))
                roomGrid_tensor = emptyRoomTensor.copy_(torch.from_numpy(roomGrid_resized).unsqueeze(0))

                floorGrid = np.full((13, 13, 6), -.2, dtype=np.float32)
                with open("F:/IsaacFloorData.txt", "r") as file:
                    for line in file:
                        gridIndex, listIndex, roomType, seen, clear, current = map(int, line.strip().split(","))
                        row, col = divmod(gridIndex, 13)
                        if roomType != 0:
                            floorGrid[row, col] = [gridIndex, listIndex + 1, roomType, seen, clear, current]
                floorGrid_normalized = floorGrid / np.array([200, 50, 30, 1, 1, 1], dtype=np.float32)
                out_of_bounds = (floorGrid_normalized < -1) | (floorGrid_normalized > 1)
                if np.any(out_of_bounds):
                    print("FloorGrid Normalization Problem:", floorGrid_normalized[out_of_bounds])
                    print(floorGrid)
                floorGrid_resized = np.transpose(floorGrid_normalized, (2, 0, 1))
                next_floorGrid_tensor = emptyFloorTensor.copy_(torch.from_numpy(floorGrid_resized).unsqueeze(0))


                entitiesList = []
                with open("F:/IsaacEntityData.txt", "r") as file:
                    roomEntities = file.read().strip()
                if roomEntities:
                    for entry in roomEntities.split("|"):
                        fields = entry.split(",")
                        entity_data = list(map(float, fields))
                        if entity_data[0] != 8 or entity_data[8] not in removedEffects:
                            if entity_data[0] == 8 and entity_data[8] not in allowedEffects:
                                print(entity_data[0], entity_data[8])
                            normalized_entity = []
                            if len(entity_data) == 11:
                                for i, value in enumerate(entity_data):
                                    normalized_value = value / entitiesNormalization[i]
                                    normalized_entity.append(normalized_value)
                                    if normalized_value > 1 or normalized_value < -1:
                                        print("Normalized entity problem:",i,normalized_value)
                                entitiesList.append(normalized_entity)
                if len(entitiesList) > 0:
                    entitiesListFull[:len(entitiesList)] = entitiesList
                else:
                    entitiesListFull = np.zeros((lenEntitiesMemory, numEntityValues), dtype=np.float32)
                readGameData()
                itemArray = np.zeros(50, dtype=np.float32)
                if len(playerData["items"]) > 0:
                    normalized_values = [value / 800 for value in playerData["items"]]
                    itemArray[:len(normalized_values)] = normalized_values


                playerGrid = playerHeatmap(playerData["x"] * playerNormalization["x"], playerData["y"] * playerNormalization["y"])
                playerGrid = emptyPlayerTensor.copy_(torch.from_numpy(playerGrid).unsqueeze(0).unsqueeze(0))
                entitiesGrids = emptyEntitiesTensor.copy_(torch.from_numpy(create_entity_heatmaps(entitiesList)).unsqueeze(0))

                final_next_state = torch.cat([roomGrid_tensor, playerGrid, entitiesGrids], dim=1) # 0 to 2 are the tile id, collision and state. 3 is player heatmap. 4 to 12 are entities. Enemy,bomb,pickup,enemy proj,ally tear,familiar,laaser,effect,slot+beggar.

                # State preparation
                additional_values = np.concatenate([np.array(keyboardKeys, dtype=np.float32),np.array(dataValues, dtype=np.float32),itemArray, entitiesListFull.flatten()])
                next_additional_values_torch = emptyAdditionalValues.copy_(torch.from_numpy(additional_values).unsqueeze(0))
                readFileLoop = True
                while readFileLoop:
                    try:
                        with open("F:/IsaacEnemyDamage.txt", "r") as file:
                            currentEnemyDamage = float(file.read())
                        readFileLoop = False
                    except:
                        pass
                readFileLoop = True

                damageDifference = currentEnemyDamage - lastEnemyDamage
                stepsInRoom += 1

                if lastRoom != room:
                    stepsInRoom = 0

                itemsSum = len(playerData["items"])

                # Compute reward (same as your code)
                reward = (itemsSum - previousItemsSum)/2 + ((totalHP - previousHP)*10) + (damageDifference/20)

                if (actionStates[0] == 1 or actionStates[1] == 1) and previousX == playerData["x"]:
                    punishX += 1
                    if punishX > 2:
                        reward -= 0.1
                else:
                    punishX = 0
                if (actionStates[2] == 1 or actionStates[3] == 1) and previousY == playerData["y"]:
                    punishY += 1
                    if punishY > 2:
                        reward -= 0.1
                else:
                    punishY = 0

                """if actionStates[8] == 1 and playerData["bombs"] == 0 and playerData["golden_bomb"] == 0:
                    reward -= 0.1
                if actionStates[9] == 1 and playerData["full_charge1"] == 0:
                    reward -= 0.1
                if actionStates[10] == 1 and playerData["card"] == 0 and playerData["pill"] == 0:
                    reward -= 0.1
                if actionStates[11] == 1 and playerData["trinket1"] == 0:
                    reward -= 0.1"""

                visited_rooms = sum(1 for y in range(floorGrid_normalized.shape[0]) for x in range(floorGrid_normalized.shape[1]) if floorGrid_normalized[y, x][3] == 1)



                # Agent's current position (example values)
                agent_x = playerData["x"]*1000
                agent_y = playerData["y"]*1000

                targets = []
                # Determine which rooms are part of the current big room
                current_rooms = []
                for y in range(floorGrid_normalized.shape[0]):  # Iterate over rows (13)
                    for x in range(floorGrid_normalized.shape[1]):  # Iterate over columns (13)
                        room_number, room_id, room_type, visited, cleared, current = floorGrid_normalized[y, x]
                        if current == 1:
                            current_rooms.append((x, y))  # Store coordinates of rooms in the current big room

                # Process all doors in the grid
                for y in range(roomY):  # Loop through grid height
                    for x in range(roomX):  # Loop through grid width
                        tile = roomGrid[y][x]  # Get tile data
                        if tile[0] == 16 and tile[2] == 2:  # Door and open
                            # Calculate door position
                            door_x = (x + 1) * 40
                            door_y = (y + 3) * 40
                            # Determine the target room for the door
                            target_room = None
                            if x == 0:  # Left door
                                for (rx, ry) in current_rooms:
                                    if rx > 0 and floorGrid_normalized[ry, rx - 1, 5] == 0:  # Check if adjacent room is NOT part of the big room
                                        target_room = (rx - 1, ry)
                                        break
                            elif x == roomX - 1:  # Right door
                                for (rx, ry) in current_rooms:
                                    if rx < 12 and floorGrid_normalized[ry, rx + 1, 5] == 0:  # Check if adjacent room is NOT part of the big room
                                        target_room = (rx + 1, ry)
                                        break
                            elif y == 0:  # Up door
                                for (rx, ry) in current_rooms:
                                    if ry > 0 and floorGrid_normalized[ry - 1, rx, 5] == 0:  # Check if adjacent room is NOT part of the big room
                                        target_room = (rx, ry - 1)
                                        break
                            elif y == roomY - 1:  # Down door
                                for (rx, ry) in current_rooms:
                                    if ry < 12 and floorGrid_normalized[ry + 1, rx, 5] == 0:  # Check if adjacent room is NOT part of the big room
                                        target_room = (rx, ry + 1)
                                        break
                            if target_room is not None:
                                # Get the cleared status of the target room
                                cleared_status = floorGrid_normalized[target_room[1], target_room[0], 4]

                                # Add door to targets list
                                targets.append([(x, y), cleared_status, (door_x, door_y), target_room])  # Include target_room in the target data

                # Track whether the agent has gotten closer to any door
                closer_to_any_door = False

                # First, check if the agent is getting closer to any door
                for target in targets:
                    (xtile, ytile), cleared_status, (door_x, door_y), target_room = target
                    key = (xtile, ytile, target_room[0], target_room[1])  # Unique identifier for the door, including room coordinates

                    # Calculate distance to the door
                    distance = ((agent_x - door_x) ** 2 + (agent_y - door_y) ** 2) ** 0.5

                    # Initialize minimum distance if not already tracked
                    if key not in door_min_distances:
                        door_min_distances[key] = distance  # Set initial minimum distance

                    # If the agent is closer than the current minimum distance
                    if distance < door_min_distances[key]:
                        closer_to_any_door = True  # Agent is getting closer to at least one door

                        # Calculate the proportion of the distance covered
                        distance_covered = door_min_distances[key] - distance
                        total_distance = door_min_distances[key]

                        # Calculate the reward increment based on the inverse of the distance
                        if cleared_status == 0:
                            reward_increment = (1 / distance) * 200  # Full reward, inversely proportional to distance
                        else:
                            reward_increment = (1 / distance) * 20  # 1/5 of max reward, inversely proportional to distance

                        # Add reward and update minimum distance
                        reward += reward_increment
                        door_min_distances[key] = distance  # Update minimum distance

                        #print(f"Rewarded {reward_increment:.2f} for getting closer to door at ({xtile}, {ytile}). New minimum distance: {distance:.2f}")

                # Now, apply punishment only if the agent is not getting closer to any door
                if not closer_to_any_door:
                    for target in targets:
                        (xtile, ytile), cleared_status, (door_x, door_y), target_room = target
                        key = (xtile, ytile, target_room[0], target_room[1])  # Unique identifier for the door, including room coordinates

                        # Calculate distance to the door
                        distance = ((agent_x - door_x) ** 2 + (agent_y - door_y) ** 2) ** 0.5

                        # If the agent is moving away from the door and the minimum distance is greater than 10
                        if distance > door_min_distances[key] and door_min_distances[key] > 10:
                            # Apply a constant punishment
                            punishment = -.02  # You can adjust this value as needed
                            reward += punishment

                            #print(f"Punished {punishment} for moving away from door at ({xtile}, {ytile}). Current distance: {distance:.2f}, minimum distance: {door_min_distances[key]:.2f}")

                if visited_rooms > 1 and visited_rooms > last_visited_rooms:
                    roomHP = totalHP
                    reward += 100
                    resetTimer += 500
                if lastRoom != room:
                    reward += 1
                if playerData["alive_enemies"] > 0 and totalHP >= roomHP:
                    reward += 0.1

                lastEnemyDamage = currentEnemyDamage

                if currentFloor != playerData["stage"]:
                    readGameData()
                    if currentFloor != playerData["stage"]:
                        print("Floor changed...")
                        reward += 1

                currentFloor = playerData["stage"]
                total_reward += reward

                # Store rollout
                if not manualTesting:
                    states.append((final_state.detach(), floorGrid_tensor.detach(), additional_values_torch.detach()))
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append((final_next_state.detach(), next_floorGrid_tensor.detach(), next_additional_values_torch.detach()))
                    dones.append(done)

                    # Reset conditions
                    if totalHP == 0 or step_count > resetTimer:
                        for k in actionStates:
                            actionStates[k] = 0
                        msg = " ".join(f"{k} {v}" for k, v in actionStates.items()) + " reset"
                        with open("F:/IsaacInputs.txt", "w") as f:
                            f.write(msg)
                        done = True
                        print(f"Main loop step counter: {step_count}, Episode {agent.episode_counter}, Total Reward: {total_reward}")
                        batchTotalRewards.append(total_reward)
                        total_reward = 0


                last_visited_rooms = visited_rooms
                previousX = playerData["x"]
                previousY = playerData["y"]
                lastRoom = room
                previousHP, previousItemsSum, previousEnemyHP = totalHP, itemsSum, totalEnemyHP
                final_state = final_next_state
                floorGrid_tensor = next_floorGrid_tensor
                additional_values_torch = next_additional_values_torch


                if done and len(states) > agent.n_steps:
                    print(f"Episode {agent.episode_counter}, Raw Rewards: {rewards[:5]}, Mean: {np.mean(rewards)}, Std: {np.std(rewards)}")
                    agent.learn((states, actions, rewards, next_states, dones, log_probs_list, values_list))
                    # Explicitly clear lists to release references
                    states.clear()
                    actions.clear()
                    rewards.clear()
                    next_states.clear()
                    dones.clear()
                    log_probs_list.clear()
                    values_list.clear()
                    torch.cuda.empty_cache()  # Clear GPU memory after learn

                    agent.episode_counter += 1
                    if not manualLearning and not manualTesting:
                        agent.save("F:/isaac_ppo_model.pth")
                        if agent.episode_counter % 50 == 0:
                            agent.save("F:/isaac_ppo_model_backup.pth")

                    """if agent.entropy.item() < 0.01:
                        #modelBroken = True

    mean_reward = np.mean(batchTotalRewards)
    filename = f"ppo_lr{agentlr:.5f}_gamma{gamma:.2f}_clip{clip_param:.2f}_vlcoef{value_loss_coef:.2f}_entcoef{entropy_coef:.4f}_ep{agent.episode_counter}_reward{mean_reward:.2f}.pt"
    save_path = os.path.join("F:/RandomSaves/", filename)
    torch.save({
        'hyperparams': {'lr': lr, 'gamma': gamma, 'clip_param': clip_param, 'value_loss_coef': value_loss_coef, 'entropy_coef': entropy_coef},
        'total_rewards': batchTotalRewards,
        'final_policy_state_dict': agent.policy.state_dict(),
        'step_counter': agent.step_counter
    }, save_path)
    print(f"Saved results to {save_path}")"""
