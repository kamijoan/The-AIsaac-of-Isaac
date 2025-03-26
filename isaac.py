from time import sleep
from keyboard import is_pressed
from threading import Thread
from scipy.ndimage import label
from os import path,listdir,remove
import numpy as np
import cv2,torch
import torch.multiprocessing as mp
#torch.set_printoptions(profile="full")

from isaacPPO import PPOPolicy, PPOAgent
#from isaacPPOTransformer import PPOPolicy, PPOAgent

def run_instance(isaacNumber, control_dict, learn_lock):
    def execute_action(action, actionStates, actionCounter, isaacNumber):
        actionCounter += 1
        if action < action_size:
            if action in opposite_actions:
                opposite = opposite_actions[action]
                actionStates[opposite] = 0
            elif action in shooting_actions:
                for shoot_action in shooting_actions:
                    actionStates[shoot_action] = 0
            # Apply the new action state
            actionStates[action] = 1
        #elif action == action_size-1:
        #    for k in actionStates:
        #        actionStates[k] = 0

        msg = " ".join(f"{k} {v}" for k, v in actionStates.items()) + f" {actionCounter}"
        with open(f"F:/IsaacInputs{isaacNumber}.txt", "w") as f:
            f.write(msg)
        while True:
            with open(f"F:/IsaacResponse{isaacNumber}.txt", "r") as file:
                if msg == file.read().strip():
                    break  # Loop until msg is confirmed back

        return list(actionStates.values()),actionStates, actionCounter  # Return full state list

    def readGameData(isaacNumber, playerData):
        while True:
            try:
                with open(f"F:/IsaacData{isaacNumber}.txt", "r") as file:
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
                dataValues = [v for k, v in playerData.items() if isinstance(v, (float))] #ignores item list and timer int
                return playerData, totalHP, dataValues
            except:
                pass

    def precompute_gaussian_kernel(size, sigma):
        x = np.linspace(-(size // 2), size // 2, size)
        y = np.linspace(-(size // 2), size // 2, size)
        X, Y = np.meshgrid(x, y)
        kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        return kernel / kernel.max()

    def playerHeatmap(player_x, player_y, sigma=.25):
        grid_h, grid_w = 16, 28
        kernel = precompute_gaussian_kernel(5, sigma)  # Small kernel size
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
        norm_x = (player_x - room_x_min) / (room_x_max - room_x_min) * (grid_w - 1)
        norm_y = (player_y - room_y_min) / (room_y_max - room_y_min) * (grid_h - 1)
        x0, y0 = int(norm_x), int(norm_y)
        kh, kw = kernel.shape
        for i in range(-kh//2, kh//2 + 1):
            for j in range(-kw//2, kw//2 + 1):
                if 0 <= y0 + i < grid_h and 0 <= x0 + j < grid_w:
                    heatmap[y0 + i, x0 + j] += kernel[i + kh//2, j + kw//2]
        return heatmap / heatmap.max()

    def entityHeatmaps(entitiesList, numMaps=10):
        heatmaps = {i: np.zeros((16, 28), dtype=np.float32) for i in range(1, numMaps)}  # 9 heatmaps for types 1-9

        # Define sigma values per entity type
        sigma_values = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.25, 7: 0.25, 8: 0.25, 9: 0.25, 10: 0.25}

        # Room width & height based on min/max positions
        room_width = room_x_max - room_x_min
        room_height = room_y_max - room_y_min

        # Generate meshgrid for Gaussian calculations
        x = np.linspace(0, 27, 28)  # Grid width
        y = np.linspace(0, 15, 16)  # Grid height
        X, Y = np.meshgrid(x, y)

        for entity in entitiesList:
            entity_type, _, xpos, ypos, *_ = entity  # Extract type and position

            # Denormalize values
            entity_type = int(entity_type * 10)  # Convert back to 1-9
            entity_type = 9 if entity_type == 10 else entity_type  # Convert type 10 to 9

            xpos *= entitiesNormalization[2]
            ypos *= entitiesNormalization[3]

            if entity_type not in heatmaps:
                print("Entity heatmap problem:", entity_type, entity)
                continue  # Ignore types outside 1-9 range

            # Normalize position to grid space (float for subpixel precision)
            norm_x = (xpos - room_x_min) / room_width
            norm_y = (ypos - room_y_min) / room_height
            grid_x = norm_x * 27  # Scale to 0-27
            grid_y = norm_y * 15  # Scale to 0-15

            # Get entity-specific sigma
            sigma = sigma_values[entity_type]

            # Compute Gaussian at this entity's position
            entity_gaussian = np.exp(-(((X - grid_x) ** 2) / (2 * sigma ** 2) + ((Y - grid_y) ** 2) / (2 * sigma ** 2)))

            # Add the Gaussian to the heatmap (accumulate values instead of replacing)
            heatmaps[entity_type] += entity_gaussian

        # Normalize each heatmap individually
        for i in range(1, numMaps):
            max_value = heatmaps[i].max()
            if max_value > 0:
                heatmaps[i] /= max_value

        # Stack into (9,16,28) tensor
        finalNP = np.stack([heatmaps[i] for i in range(1, numMaps)], axis=0)

        return finalNP  # Shape: (9,16,28)

    def hsv_to_bgr(h, s, v):
        """Convert HSV values (0-360, 0-255, 0-255) to BGR color for OpenCV."""
        hsv_color = np.uint8([[[h // 2, s, v]]])  # OpenCV scales hue 0-180
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in bgr_color)  # Convert to (B, G, R)

    def drawOverlay():
        visualDataIndex = channelIndex = timer = 0
        while True:
            timer += 1
            sleep(1/30)
            overlay = np.zeros((500,250,3), dtype=np.uint8)
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
                cv2.circle(overlay, (int(playerData["x"]*playerNormalization["x"]/10), int(playerData["y"]*playerNormalization["y"]/10)), 2, (0, 255, 0), -1)  # player display
                for entity in entitiesList:
                    hue = int(entity[1] * 180)
                    color = hsv_to_bgr(hue, 150, 150)
                    cv2.circle(overlay, (int(entity[2]*entitiesNormalization[2]/10), int(entity[3]*entitiesNormalization[3]/10)), 1, color, -1) #entity display

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
                if action_size == 4:
                    actions = ["A","D","W","S"]
                elif action_size == 8:
                    actions = ["A","D","W","S","sL","sR","sU","sD"]

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
                #learn() progress bar
                cv2.rectangle(overlay, (0,190),(int((len(states) / agent.n_steps) * overlay.shape[1]),200), (200,200,200), -1)
                cv2.rectangle(overlay, (0,190),(int((agent.progress / 100) * overlay.shape[1]),200), (50,200,50), -1)

                if len(agent.policy.visualData) != 0:
                    if is_pressed('7') and visualDataIndex > 0 and timer > 10:
                        visualDataIndex -= 1
                        channelIndex = 0
                        timer = 0
                        print("Graph Index:",visualDataIndex)
                    if is_pressed('9') and visualDataIndex < len(agent.policy.visualData)-1 and timer > 5:
                        visualDataIndex += 1
                        channelIndex = 0
                        timer = 0
                        print("Graph Index:",visualDataIndex)

                    visualData = agent.policy.visualData[visualDataIndex].detach().clone().cpu()

                    # Convert visualData to numpy based on shape
                    if len(visualData.shape) == 4:
                        layers = visualData[0].to("cpu", non_blocking=True).numpy()  # First batch item
                    elif len(visualData.shape) == 3:
                        layers = visualData.to("cpu", non_blocking=True).numpy()
                    elif len(visualData.shape) == 2:
                        layers = visualData.unsqueeze(0).to("cpu", non_blocking=True).numpy()
                    else:
                        raise ValueError(f"Unexpected number of channels: {visualData.shape}")

                    num_layers, height, width = layers.shape

                    # Handle channel index changes
                    if is_pressed('4') and channelIndex > 0 and timer > 4:
                        channelIndex -= 1
                        timer = 0
                        print("Graph Channel Index:", channelIndex)
                    if is_pressed('6') and channelIndex < num_layers - 1 and timer > 4:
                        channelIndex += 1
                        timer = 0
                        print("Graph Channel Index:", channelIndex)

                    # Normalize layers to 0-255
                    layers = cv2.normalize(layers, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                    # Overlay dimensions
                    max_overlay_height = overlay.shape[0] - 205  # Space from y=205 to bottom
                    max_overlay_width = overlay.shape[1]

                    # Scale factor and base dimensions
                    scale_factor = 6
                    base_height = height * scale_factor
                    base_width = width * scale_factor

                    # Adjust scale to fit half the available height (for stacking)
                    if base_height > max_overlay_height // 2:
                        scale_adjust = (max_overlay_height // 2) / base_height
                        base_height = int(base_height * scale_adjust)
                        base_width = int(base_width * scale_adjust)

                    # Calculate offsets and total canvas size
                    y_offset = min(5, (max_overlay_height - base_height) // max(1, num_layers - 1))
                    total_layer_height = base_height + (num_layers - 1) * y_offset
                    if total_layer_height > max_overlay_height:
                        # Recalculate base_height to fit all layers
                        scale_adjust = max_overlay_height / total_layer_height
                        base_height = int(base_height * scale_adjust)
                        base_width = int(base_width * scale_adjust)
                        y_offset = int(y_offset * scale_adjust)
                        total_layer_height = base_height + (num_layers - 1) * y_offset

                    total_layer_width = base_width * num_layers
                    if total_layer_width <= max_overlay_width:
                        x_offset = (max_overlay_width - total_layer_width) // max(1, num_layers - 1)
                    else:
                        x_offset = (max_overlay_width - base_width) // max(1, num_layers - 1)

                    # Canvas dimensions
                    canvas_height = min(total_layer_height, max_overlay_height)
                    canvas_width = min(max_overlay_width, base_width + (num_layers - 1) * x_offset)
                    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)  # RGBA

                    # Render layers
                    for i in range(num_layers):
                        if i != channelIndex:  # Semi-transparent layers
                            layer = layers[i]
                            layer_colored = cv2.applyColorMap(layer, cv2.COLORMAP_JET)
                            layer_resized = cv2.resize(layer_colored, (base_width, base_height), interpolation=cv2.INTER_NEAREST)
                            layer_rgba = cv2.cvtColor(layer_resized, cv2.COLOR_BGR2BGRA)
                            layer_rgba[:, :, 3] = 25  # Semi-transparent

                            x_pos = i * x_offset
                            y_pos = (num_layers - 1 - i) * y_offset
                            overlay_x = x_pos
                            overlay_y = 205 + y_pos
                            y_max = min(overlay_y + base_height, 205 + canvas_height)
                            x_max = min(overlay_x + base_width, canvas_width)

                            if y_max > overlay_y and x_max > overlay_x:
                                canvas_roi = canvas[overlay_y-205:y_max-205, overlay_x:x_max]
                                layer_patch = layer_rgba[:y_max-overlay_y, :x_max-overlay_x]
                                alpha = layer_patch[:, :, 3] / 255.0
                                alpha_bg = (1 - alpha)
                                for c in range(3):
                                    canvas_roi[:, :, c] = (
                                        alpha * layer_patch[:, :, c] + alpha_bg * canvas_roi[:, :, c]
                                    ).astype(np.uint8)
                                canvas_roi[:, :, 3] = np.maximum(canvas_roi[:, :, 3], layer_patch[:, :, 3])

                    # Render opaque layer last
                    if channelIndex < num_layers:
                        layer = layers[channelIndex]
                        layer_colored = cv2.applyColorMap(layer, cv2.COLORMAP_JET)
                        layer_resized = cv2.resize(layer_colored, (base_width, base_height), interpolation=cv2.INTER_NEAREST)
                        layer_rgba = cv2.cvtColor(layer_resized, cv2.COLOR_BGR2BGRA)
                        layer_rgba[:, :, 3] = 255  # Fully opaque

                        x_pos = channelIndex * x_offset
                        y_pos = (num_layers - 1 - channelIndex) * y_offset
                        overlay_x = x_pos
                        overlay_y = 205 + y_pos
                        y_max = min(overlay_y + base_height, 205 + canvas_height)
                        x_max = min(overlay_x + base_width, canvas_width)

                        if y_max > overlay_y and x_max > overlay_x:
                            canvas_roi = canvas[overlay_y-205:y_max-205, overlay_x:x_max]
                            layer_patch = layer_rgba[:y_max-overlay_y, :x_max-overlay_x]
                            alpha = layer_patch[:, :, 3] / 255.0
                            alpha_bg = (1 - alpha)
                            for c in range(3):
                                canvas_roi[:, :, c] = (
                                    alpha * layer_patch[:, :, c] + alpha_bg * canvas_roi[:, :, c]
                                ).astype(np.uint8)
                            canvas_roi[:, :, 3] = np.maximum(canvas_roi[:, :, 3], layer_patch[:, :, 3])

                    # Convert to BGR and assign to overlay
                    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
                    overlay[205:205+canvas_height, 0:canvas_width] = canvas_bgr

            except Exception as e:
                sleep(.1)
                print("overlay:",e)

            cv2.imshow(f"Overlay{isaacNumber}", overlay)
            cv2.waitKey(1)

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
        "coins": 100,
        "bombs": 100,
        "keys": 100,
        "golden_bomb": 1,
        "golden_key": 1,
        "active1": 800,  # Max item ID in Repentance
        "charge1": 12,  # Most active items max out at 12 charge
        "full_charge1": 1,
        "active2": 800,
        "charge2": 12,
        "full_charge2": 1,
        "trinket1": 200,  # Max trinket ID
        "trinket2": 200,
        "damage": 100,  # Extremely high damage (Soy Milk is 0.5, Brimstone is ~10)
        "fire_rate": 100,  # Higher = slower (default 10, Soy Milk ~2)
        "shot_speed": 2,
        "range": 500,
        "luck": 10,
        "speed": 2,
        "card": 100,  # Max card ID
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

    door_mappings = {
        (0, 4): lambda rooms: (min(t[0] for t in rooms) - 1, min(t[1] for t in rooms)),  # left x1 or left up x2
        (0, 11): lambda rooms: (min(t[0] for t in rooms) - 1, max(t[1] for t in rooms)), # left down x2
        (7, 0): lambda rooms: (min(t[0] for t in rooms), min(t[1] for t in rooms) - 1),  # top x1 or top left x2
        (7, 7): lambda rooms: (min(t[0] for t in rooms), max(t[1] for t in rooms) - 1),  # mirror L room going up to top left
        (7, 8): lambda rooms: (min(t[0] for t in rooms), min(t[1] for t in rooms) + 1),  # bottom x1
        (7, 15): lambda rooms: (min(t[0] for t in rooms), max(t[1] for t in rooms) + 1), # bottom left x2
        (13, 4): lambda rooms: (max(t[0] for t in rooms) - 1, min(t[1] for t in rooms)), # # mirror L room going left to top left
        (14, 4): lambda rooms: (max(t[0] for t in rooms) + 1, min(t[1] for t in rooms)), # right x1
        (14, 11): lambda rooms: (min(t[0] for t in rooms) + 1, max(t[1] for t in rooms)),# right down x1.5
        (20, 0): lambda rooms: (max(t[0] for t in rooms), max(t[1] for t in rooms) - 1), # top right x2
        (20, 7): lambda rooms: (max(t[0] for t in rooms), max(t[1] for t in rooms) - 1), # L room going up to top right
        (20, 8): lambda rooms: (max(t[0] for t in rooms), min(t[1] for t in rooms) + 1), # bottom right x1.5
        (20, 15): lambda rooms: (max(t[0] for t in rooms), max(t[1] for t in rooms) + 1),# bottom right x2
        (27, 4): lambda rooms: (max(t[0] for t in rooms) + 1, min(t[1] for t in rooms)), # right up x2
        (27, 11): lambda rooms: (max(t[0] for t in rooms) + 1, max(t[1] for t in rooms)),# right down x2
    }

    playerData = {key:0 for key in playerNormalization}
    playerData["items"] = []
    playerData["time_counter"] = 0
    totalHP = playerData["hp"]+playerData["soul_hp"]+playerData["black_hp"]+playerData["rotten_hp"]+playerData["bone_hp"]+playerData["eternal_hp"]+playerData["extra_lives"]

    #heatmap calc
    room_x_min, room_y_min = 20, 100
    room_x_max, room_y_max = 1140, 740

    actionStates = {i: 0 for i in range(12)}
    previous_state = actionStates.copy()  # Store last known states
    opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}
    shooting_actions = {4, 5, 6, 7}  # Only one can be active
    #map only for manual teaching
    key_map = {0: 'a',  1: 'd',  2: 'w',  3: 's',  # Movement: Left, Right, Up, Down
               4: 'j',  5: 'l',  6: 'i',  7: 'k',  # Shooting: Left, Right, Up, Down
               8: 't',  9: 'f', 10: 'g', 11: 'h'}   # Bomb, Item, Card, Drop

    itemArray = np.zeros(50, dtype=np.float32)
    keyboardKeys = list(actionStates.values()) #12 keys
    reset,done = False,True
    lenEntitiesMemory = 200
    numEntityValues = 11
    num_additional_values = len(keyboardKeys)+len(playerNormalization)+len(itemArray)+(lenEntitiesMemory*numEntityValues)
    action_size = 4

    agent = PPOAgent(room_shape=(13, 16, 28),map_shape=(6, 13, 13),action_size=action_size,n_critical=len(keyboardKeys) + len(playerNormalization),n_items=len(itemArray),n_entity_memory=(lenEntitiesMemory * numEntityValues), isaacNumber=isaacNumber)

    emptyFloorTensor = torch.empty((1,6,13,13), dtype=torch.float32, device="cuda")
    emptyAdditionalValues = torch.empty((1,num_additional_values), dtype=torch.float32, device="cuda")
    emptyFinalState = torch.empty((1,13,16,28), dtype=torch.float32, device="cuda")

    roomX,roomY = 15,9
    overlay_thread = Thread(target=drawOverlay, daemon=True)
    overlay_thread.start()

    manualLearning = False
    manualTesting = False

    # PPO rollout storage
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs_list = []
    values_list = []
    hidden_states = []
    states.append((emptyFinalState.clone(), emptyFloorTensor.clone(), emptyAdditionalValues.clone()))

    #effect ids, still in progress, removed is ignored, allowed just doesn't get printed.
    removedEffects = {2,3,4,5,7,11,12,13,14,15,16,17,20,21,27,33,38,43,58,59,63,64,65,66,68,79,86,99,133,146}
    allowedEffects = {1,6,22,23,24,25,26,34,44,45,46,50,57,61,62}

    print(f"Running Isaac {isaacNumber}...")
    try:
        model_dir = "F:/"
        best_model_path = None
        best_reward_per_step = -float('inf')
        for filename in listdir(model_dir):
            if filename.startswith(f"isaacPPOModelIsaac{isaacNumber}RPS") and filename.endswith(".pth"):
                try:
                    start_rps = filename.index("RPS") + 3
                    start_ep = filename.index("EP")
                    file_rps = float(filename[start_rps:start_ep])
                    if file_rps > best_reward_per_step:
                        best_reward_per_step = file_rps
                        best_model_path = path.join(model_dir, filename)
                except (ValueError, IndexError) as e:
                    print(f"Skipping invalid filename {filename}: {e}")
                    continue
        if best_model_path:
            agent.load(best_model_path)  # Uses your load method
            print(f"Loaded best model for Isaac {isaacNumber}: {best_model_path} with reward/step {best_reward_per_step:.4f}")
        else:
            raise FileNotFoundError("No valid model files found")
    except Exception as e:
        print(f"No model found for Isaac {isaacNumber}: {e}")

    with open(f"F:/IsaacInputs{isaacNumber}.txt", "w") as f:
        f.write("")

    freedom = False

    episodeSteps = 0

    while control_dict.get(f"isaac_{isaacNumber}", True):
        sleep(1/60)
        playerData, totalHP, dataValues = readGameData(isaacNumber, playerData)
        currentFloor = playerData["stage"]

        if playerData["time_counter"] > 1 and not reset and not manualTesting:
            with open(f"F:/IsaacInputs{isaacNumber}.txt", "w") as f:
                f.write("reset")
            reset = True
            sleep(0.2)

        elif playerData["time_counter"] <= 1 or manualTesting:
            for k in actionStates:
                actionStates[k] = 0
            msg = " ".join(f"{k} {v}" for k, v in actionStates.items())
            with open(f"F:/IsaacInputs{isaacNumber}.txt", "w") as f:
                f.write(msg)
            reset = done = False

            # Initialize run
            playerData, totalHP, dataValues = readGameData(isaacNumber, playerData)
            previousX = playerData["x"]
            previousY = playerData["y"]
            currentFloor = playerData["stage"]
            floorGrid_tensor = emptyFloorTensor
            additional_values_tensor = emptyAdditionalValues
            final_state = emptyFinalState

            itemsSum = len(playerData["items"])
            previousPickups = pickups = playerData["bombs"] + playerData["coins"] + playerData["keys"]
            actionCounter = step_count = enemyDamage = lastEnemyDamage = punishX = punishY = total_reward = previousItemsSum = totalEnemyHP = previousEnemyHP = enemyDamage = 0
            roomHP = previousHP = totalHP
            visited_rooms = last_visited_rooms = 1
            lastRoom = 84
            resetTimer = 500

            door_min_distances = {}

            itemArray = np.zeros(50, dtype=np.float32)
            entitiesListFull = np.zeros((lenEntitiesMemory, numEntityValues), dtype=np.float32)
            totalRoomGrid = np.zeros((105, 183, 13), dtype=np.float32)
            totalRoomGrid[:, :, :3] = -0.2

            hidden_state = None

            sleep(1)

            while not done:

                step_count += 1

                action, log_prob, value, hidden_state = agent.act((final_state, floorGrid_tensor, additional_values_tensor),hidden_state)
                log_probs_list.append(log_prob)
                values_list.append(value)
                hidden_states.append(hidden_state)

                if not manualTesting and not manualLearning:
                    keyboardKeys, actionStates, actionCounter = execute_action(action, actionStates, actionCounter, isaacNumber)
                """elif manualLearning:
                    action = deduce_action()
                    update_actionStates()
                    keyboardKeys = list(actionStates.values())"""

                floorGrid = np.full((13, 13, 6), -.2, dtype=np.float32)
                with open(f"F:/IsaacFloorData{isaacNumber}.txt", "r") as file:
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
                current_rooms = []
                for y in range(floorGrid_normalized.shape[0]):
                    for x in range(floorGrid_normalized.shape[1]):
                        room_number, room_id, room_type, visited, cleared, current = floorGrid_normalized[y, x]
                        if current == 1:
                            current_rooms.append((x, y, room_number))  # Store coordinates of rooms in the current big room

                while True:
                    try:
                        with open(f"F:/IsaacTileData{isaacNumber}.txt", "r") as file:
                            first_line = file.readline()
                            tile_data = file.readlines()
                            room, roomX, roomY = map(int, first_line.split(","))
                            roomGrid = np.array([list(map(int, line.strip().split(","))) for line in tile_data], dtype=np.int32)
                            roomGrid = roomGrid.reshape((roomY, roomX, 3))
                            break
                    except:
                        pass
                roomGrid_normalized = roomGrid / np.array([27, 5, 1000], dtype=np.float32)
                out_of_bounds = (roomGrid_normalized < -1) | (roomGrid_normalized > 1)
                if np.any(out_of_bounds):
                    print("RoomGrid Normalization Problem:", roomGrid_normalized[out_of_bounds])
                    print("grid",roomGrid)
                empty_mask = np.all(roomGrid_normalized == [0, 0, 0], axis=-1)
                labeled_grid, num_features = label(empty_mask)
                for i in range(1, num_features + 1):
                    block_mask = labeled_grid == i  # Boolean mask for the current block
                    if np.any(block_mask[0, :]) or np.any(block_mask[-1, :]) or np.any(block_mask[:, 0]) or np.any(block_mask[:, -1]):
                        roomGrid_normalized[block_mask] = [-.2, -.2, -.2]

                entitiesList = []
                with open(f"F:/IsaacEntityData{isaacNumber}.txt", "r") as file:
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

                entitiesGrids = entityHeatmaps(entitiesList)
                if len(current_rooms) != 0 and any(room_number == room/200 for _, _, room_number in current_rooms):
                    minRoom_x = min(x for x, y, _ in current_rooms)
                    minRoom_y = min(y for x, y, _ in current_rooms)
                    target_x = 15 + (minRoom_x - 1) * 14 if minRoom_x > 0 else 0
                    target_y = 9 + (minRoom_y - 1) * 8 if minRoom_y > 0 else 0
                    valid_mask = np.any(roomGrid_normalized != -0.2, axis=-1)
                    totalRoomGrid[target_y:target_y + roomY, target_x:target_x + roomX, :3][valid_mask] = roomGrid_normalized[valid_mask]
                    entitiesGrids = np.transpose(entitiesGrids, (1, 2, 0))  # Shape: (16,28,9)
                    totalRoomGrid[target_y:target_y + roomY, target_x:target_x + roomX, 4:13][valid_mask] = entitiesGrids[:roomY, :roomX, :][valid_mask]

                roomGridSection = totalRoomGrid[target_y:target_y+16, target_x:target_x+28, :]
                playerData, totalHP, dataValues = readGameData(isaacNumber, playerData)
                agent_x = playerData["x"]*playerNormalization["x"]
                agent_y = playerData["y"]*playerNormalization["y"]
                playerGrid = playerHeatmap(agent_x, agent_y)
                roomGridSection[:, :, 3] = playerGrid
                roomGridSection = np.transpose(roomGridSection, (2, 0, 1))  #(13,16,28)

                itemArray = np.zeros(50, dtype=np.float32)
                if len(playerData["items"]) > 0:
                    normalized_values = [value / 800 for value in playerData["items"]]
                    itemArray[:len(normalized_values)] = normalized_values
                additional_values = np.concatenate([np.array(keyboardKeys, dtype=np.float32),np.array(dataValues, dtype=np.float32),itemArray, entitiesListFull.flatten()])

                # 0 to 2 are the tile id, collision and state. 3 is player heatmap. 4 to 12 are entities. Enemy,bomb,pickup,enemy proj,ally tear,familiar,laaser,effect,slot+beggar.
                final_state = emptyFinalState.copy_(torch.from_numpy(roomGridSection).unsqueeze(0))
                floorGrid_tensor = emptyFloorTensor.copy_(torch.from_numpy(floorGrid_resized).unsqueeze(0))
                additional_values_tensor = emptyAdditionalValues.copy_(torch.from_numpy(additional_values).unsqueeze(0))

                #Rewards
                while True:
                    try:
                        with open(f"F:/IsaacEnemyDamage{isaacNumber}.txt", "r") as file:
                            currentEnemyDamage = float(file.read())
                            break
                    except:
                        pass

                damageDifference = currentEnemyDamage - lastEnemyDamage

                itemsSum = len(playerData["items"])
                pickups = playerData["bombs"] + playerData["coins"] + playerData["keys"]

                reward = (itemsSum - previousItemsSum)/10# + ((totalHP - previousHP)) + (damageDifference/10)

                if pickups > previousPickups:
                    reward += .1 * pickups

                if (actionStates[0] == 1 or actionStates[1] == 1) and previousX == playerData["x"]:
                    punishX += 1
                    if punishX > 2:
                        reward -= .01
                else:
                    punishX = 0
                if (actionStates[2] == 1 or actionStates[3] == 1) and previousY == playerData["y"]:
                    punishY += 1
                    if punishY > 2:
                        reward -= .01
                else:
                    punishY = 0

                if actionStates[8] == 1 and playerData["bombs"] == 0 and playerData["golden_bomb"] == 0:
                    reward -= .01
                if actionStates[9] == 1 and playerData["full_charge1"] == 0:
                    reward -= .01
                if actionStates[10] == 1 and playerData["card"] == 0 and playerData["pill"] == 0:
                    reward -= .01
                if actionStates[11] == 1 and playerData["trinket1"] == 0:
                    reward -= .01

                targets = []
                for y in range(roomY):
                    for x in range(roomX):
                        tile = roomGrid[y][x]
                        if tile[0] == 16 and tile[2] == 2 and len(current_rooms) > 0:  # Door and open
                            door_x = (x + 1) * 40  # Position inside the room
                            door_y = (y + 3) * 40
                            if (x, y) in door_mappings:
                                target_room = door_mappings[(x, y)](current_rooms)
                            else:
                                print("Door not listed", x, y)
                            cleared_status = floorGrid_normalized[target_room[1], target_room[0], 4]
                            if target_room:
                                targets.append([(x, y), cleared_status, (door_x, door_y), target_room])
                closer_to_any_door = False
                for target in targets:
                    (xtile, ytile), cleared_status, (door_x, door_y), target_room = target
                    key = (xtile, ytile, target_room[0], target_room[1])
                    targetRoomType = floorGrid_normalized[target_room[1], target_room[0], 2]
                    targetMultiplier = 2 if targetRoomType == 0.13333334 else 1
                    distance = ((agent_x - door_x) ** 2 + (agent_y - door_y) ** 2) ** 0.5  # Distance in units

                    if key not in door_min_distances:
                        door_min_distances[key] = distance  # Initialize with current distance

                    # Reward for getting closer
                    distance_covered = door_min_distances[key] - distance
                    if distance_covered > 0:
                        closer_to_any_door = True
                        # Normalize by 200 units (5 tiles), cap at 1.0
                        reward_increment = min(1.0, distance_covered / 200) * (0.2 if cleared_status == 0 else 0.02)
                        reward += reward_increment * targetMultiplier

                    """elif closer_to_any_door == False and distance_covered < 0 and cleared_status == 0:
                        punishment = -0.001 * min(1.0, (distance - door_min_distances[key]) / 200)
                        reward += punishment * targetMultiplier"""
                    if distance < door_min_distances[key]:
                        door_min_distances[key] = distance

                visited_rooms = sum(1 for y in range(floorGrid_normalized.shape[0]) for x in range(floorGrid_normalized.shape[1]) if floorGrid_normalized[y, x][3] == 1)
                if visited_rooms > 1 and visited_rooms > last_visited_rooms:
                    roomHP = totalHP
                    reward += .2
                    resetTimer += 250
                if lastRoom != room:
                    reward += .05
                if playerData["alive_enemies"] > 0 and totalHP >= roomHP:
                    reward += .01
                    resetTimer += .5

                if currentFloor != playerData["stage"]:
                    playerData, totalHP, dataValues = readGameData(isaacNumber, playerData)
                    if currentFloor != playerData["stage"]:
                        print(f"{isaacNumber}. Floor changed...")
                        door_min_distances = {}
                        totalRoomGrid = np.zeros((105, 183, 13), dtype=np.float32)
                        totalRoomGrid[:, :, :3] = -0.2
                        reward += .5

                reward *= 100
                total_reward += reward

                if not manualTesting:
                    # Done conditions
                    if totalHP == 0 or step_count > resetTimer or (lastRoom != room and freedom == False):
                        for k in actionStates:
                            actionStates[k] = 0
                        msg = " ".join(f"{k} {v}" for k, v in actionStates.items()) + " reset"
                        with open(f"F:/IsaacInputs{isaacNumber}.txt", "w") as f:
                            f.write(msg)
                        done = True
                        emptyFinalState.zero_()
                        emptyFloorTensor.zero_()
                        emptyAdditionalValues.zero_()
                        #print(f"{isaacNumber}. Main loop step counter: {step_count}, Episode {agent.episode_counter}, Total Reward: {total_reward}, Per step: {(total_reward/step_count):.2f}")
                        #freedom = True if total_reward > 0 else False
                        total_reward = 0
                        episodeSteps += step_count

                    states.append((final_state.clone(), floorGrid_tensor.clone(), additional_values_tensor.clone()))
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(done)

                currentFloor = playerData["stage"]
                previousX = playerData["x"]
                previousY = playerData["y"]
                previousHP, previousItemsSum, previousEnemyHP, previousPickups, lastRoom, last_visited_rooms, lastEnemyDamage = totalHP, itemsSum, totalEnemyHP, pickups, room, visited_rooms, currentEnemyDamage

                if done and len(states) > (agent.n_steps - resetTimer):
                    total_reward_sum = sum(rewards)  # Total reward for the episode
                    reward_per_step = total_reward_sum / episodeSteps  # Calculate reward per step
                    print(f"Isaac {isaacNumber} - Episode {agent.episode_counter}, "
                          f"Total Reward: {total_reward_sum:.2f}, Reward/Step: {reward_per_step:.4f}")

                    with learn_lock:
                        agent.learn((states, actions, rewards, dones, log_probs_list, values_list, hidden_states))
                        states.clear()
                        actions.clear()
                        rewards.clear()
                        dones.clear()
                        log_probs_list.clear()
                        values_list.clear()
                        hidden_states.clear()
                        torch.cuda.empty_cache()

                    states.append((emptyFinalState.clone(), emptyFloorTensor.clone(), emptyAdditionalValues.clone()))

                    agent.episode_counter += 1
                    if not manualLearning and not manualTesting:
                        reward_per_step = total_reward_sum / episodeSteps
                        episodeSteps = 0
                        savesLimit = 3
                        model_dir = "F:/"
                        model_path = f"F:/isaacPPOModelIsaac{isaacNumber}RPS{reward_per_step:.4f}EP{agent.episode_counter}.pth"

                        own_files = []
                        all_files = []
                        for filename in listdir(model_dir):
                            if filename.startswith("isaacPPOModelIsaac") and filename.endswith(".pth"):
                                try:
                                    start_isaac = filename.index("Isaac") + 5
                                    start_rps = filename.index("RPS") + 3
                                    start_ep = filename.index("EP")
                                    file_isaac_num = int(filename[start_isaac:start_rps-3])
                                    file_rps = float(filename[start_rps:start_ep])
                                    full_path = path.join(model_dir, filename)
                                    if file_isaac_num == isaacNumber:
                                        own_files.append((full_path, file_rps))
                                    all_files.append((full_path, file_rps, file_isaac_num))
                                except (ValueError, IndexError) as e:
                                    print(f"Skipping invalid filename {filename}: {e}")
                                    continue

                        if len(own_files) >= savesLimit:
                            lowest_file, lowest_rps = min(own_files, key=lambda x: x[1])
                            if reward_per_step > lowest_rps:
                                remove(lowest_file)
                                print(f"Removed {lowest_file} with reward/step {lowest_rps:.4f}")
                            else:
                                print(f"New reward/step {reward_per_step:.4f} not better than lowest {lowest_rps:.4f}, skipping save")
                                model_path = None

                        if model_path:
                            agent.save(model_path)  # Uses your save method

                        if len(own_files) >= savesLimit and agent.episode_counter % 10 == 0:
                            current_reward_per_step = reward_per_step
                            best_model_path = model_path if model_path else own_files[-1][0]
                            best_reward_per_step = current_reward_per_step
                            best_instance_id = isaacNumber

                            for file_path, file_rps, file_isaac_num in all_files:
                                if file_rps > best_reward_per_step:
                                    best_reward_per_step = file_rps
                                    best_model_path = file_path
                                    best_instance_id = file_isaac_num

                            if best_reward_per_step > current_reward_per_step:
                                try:
                                    # Load the checkpoint and extract policy state dict
                                    checkpoint = torch.load(best_model_path, map_location=agent.device)
                                    best_state_dict = checkpoint['policy_state_dict']
                                    current_state_dict = agent.policy.state_dict()

                                    # Check for key mismatches
                                    best_keys = set(best_state_dict.keys())
                                    current_keys = set(current_state_dict.keys())
                                    common_keys = best_keys.intersection(current_keys)
                                    missing_in_best = current_keys - best_keys
                                    missing_in_current = best_keys - current_keys

                                    if missing_in_best or missing_in_current:
                                        print(f"Warning: Model mismatch for Isaac {isaacNumber} blending with {best_model_path}")
                                        if missing_in_best:
                                            print(f"Keys in current but not in best: {missing_in_best}")
                                        if missing_in_current:
                                            print(f"Keys in best but not in current: {missing_in_current}")
                                        if not common_keys:
                                            raise ValueError("No common parameters to blend; models are incompatible")

                                    # Calculate blending weight
                                    reward_diff = best_reward_per_step - current_reward_per_step
                                    alpha = min(0.9, reward_diff / (reward_diff + 0.1))

                                    # Get state dicts
                                    current_state_dict = agent.policy.state_dict()
                                    best_state_dict = checkpoint['policy_state_dict']
                                    common_keys = [key for key in current_state_dict.keys() if key in best_state_dict]

                                    # Define critic-specific parameters to skip blending
                                    critic_params = ['critic.weight', 'critic.bias']

                                    # Blend only non-critic parameters, keep critic parameters from current model
                                    new_state_dict = {}
                                    for key in common_keys:
                                        if key in critic_params:
                                            # Keep the current critic parameters unchanged
                                            new_state_dict[key] = current_state_dict[key]
                                        else:
                                            # Blend shared and actor parameters
                                            new_state_dict[key] = (1 - alpha) * current_state_dict[key] + alpha * best_state_dict[key]

                                    # Add any remaining parameters from current_state_dict that weren't in common_keys
                                    for key in current_state_dict.keys():
                                        if key not in new_state_dict:
                                            new_state_dict[key] = current_state_dict[key]

                                    # Load the blended policy state dict
                                    agent.policy.load_state_dict(new_state_dict)
                                    print(f"Isaac {isaacNumber} blended model with {best_model_path} "
                                          f"(Reward/Step: {best_reward_per_step:.4f}, Weight: {alpha:.2f}), "
                                          f"critic parameters unchanged from current model")

                                    # Update optimizer and counters
                                    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                                    agent.step_counter = max(agent.step_counter, checkpoint['step_counter'])
                                    agent.episode_counter = max(agent.episode_counter, checkpoint['episode_counter'])
                                    print(f"Updated optimizer, step_counter={agent.step_counter}, episode_counter={agent.episode_counter}")

                                except Exception as e:
                                    print(f"Isaac {isaacNumber} failed to blend model from {best_model_path}: {e}")

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Required for CUDA in multiprocessing
    manager = mp.Manager()
    control_dict = manager.dict()  # Controls whether instances should run
    processes = {}  # Store running processes
    learn_lock = mp.Lock()

    while True:
        try:
            cmd = input("Enter command (start <n> <m>..., stop <n>, quit): ").strip().split()
            if not cmd:
                continue
            action = cmd[0].lower()

            if action == "start" and len(cmd) > 1:
                for num in cmd[1:]:
                    try:
                        isaacNumber = int(num)
                        if f"isaac_{isaacNumber}" not in processes:
                            control_dict[f"isaac_{isaacNumber}"] = True
                            p = mp.Process(target=run_instance, args=(isaacNumber, control_dict, learn_lock))
                            p.start()
                            processes[f"isaac_{isaacNumber}"] = p
                            print(f"Started Isaac {isaacNumber}")
                        else:
                            print(f"Isaac {isaacNumber} is already running")
                    except ValueError:
                        print(f"Invalid number: {num}")

            elif action == "stop" and len(cmd) > 1:
                for num in cmd[1:]:
                    try:
                        isaacNumber = int(num)
                        key = f"isaac_{isaacNumber}"
                        if key in processes:
                            control_dict[key] = False
                            processes[key].join()
                            del processes[key]
                            print(f"Stopped Isaac {isaacNumber}")
                        else:
                            print(f"Isaac {isaacNumber} is not running")
                    except ValueError:
                        print(f"Invalid number: {num}")

            elif action == "quit":
                for key in list(processes.keys()):
                    control_dict[key] = False
                    processes[key].join()
                    del processes[key]
                print("All instances stopped. Exiting.")
                break

            else:
                print("Invalid command. Use: start <n> <m>..., stop <n> <m>..., quit")

        except EOFError:
            print("EOF detected. Use 'quit' to exit cleanly.")
        except Exception as e:
            print(f"Error: {e}")
