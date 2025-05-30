from time import sleep
from keyboard import is_pressed
from threading import Thread
from scipy.ndimage import label
from os import path,listdir,remove
import numpy as np
import cv2, torch
import torch.nn.functional as F
import torch.multiprocessing as mp

#torch.set_printoptions(profile="full")

from isaacPPO import PPOPolicy, PPOAgent

def run_instance(isaacNumber, initializeData, control_dict, shared_model, modelLock, agentCopy, rollout_queue):

    def readGameData(isaacNumber, playerData):
        with open(f"F:/IsaacData{isaacNumber}.txt", "r") as f:
            data = f.read().strip()
        for item in data.split(","):
            if "=" in item:
                key, value = item.split("=")
                if key == "items":
                    try:
                        playerData[key] = list(map(int, value.strip("[]").split("|")))
                    except:
                        playerData["items"] = []
                elif key in playerNormalization:
                    norm_value = float(value) / playerNormalization[key]
                    if key == "vx":
                        playerData["vxneg"] = max(0.0, -norm_value)
                        playerData["vx"] = max(0.0, norm_value)
                    elif key == "vy":
                        playerData["vyneg"] = max(0.0, -norm_value)
                        playerData["vy"] = max(0.0, norm_value)
                    else:
                        playerData[key] = norm_value
                        if playerData[key] > 1 or playerData[key] < 0:
                            print("Normalized Stat problem:",key,playerData[key],"formula:",playerData[key]*playerNormalization[key],"/",playerNormalization[key])
                elif key in playerData:
                    playerData[key] = int(value)

        totalHP = playerData["hp"]+playerData["soul_hp"]+playerData["black_hp"]+playerData["rotten_hp"]+playerData["bone_hp"]+playerData["eternal_hp"]+playerData["extra_lives"]
        dataValues = np.array([v for v in playerData.values() if isinstance(v, float)], dtype=np.float32) #ignores item list and timer int
        return playerData, totalHP, dataValues

    def playerHeatmap(player_x, player_y, sigma=.2):
        grid_h, grid_w = 16, 28
        x = np.linspace(0, 27, 28)  # Grid width
        y = np.linspace(0, 15, 16)  # Grid height
        X, Y = np.meshgrid(x, y)
        norm_x = (player_x - room_x_min) / room_width
        norm_y = (player_y - room_y_min) / room_height
        grid_x = norm_x * 27  # Scale to 0-27
        grid_y = norm_y * 15  # Scale to 0-15

        heatmap = np.exp(-(((X - grid_x) ** 2) / (2 * sigma ** 2) + ((Y - grid_y) ** 2) / (2 * sigma ** 2)))
        heatmap = heatmap / heatmap.max()
        heatmap[heatmap < 0.01] = 0

        return heatmap, int(grid_x), int(grid_y)

    def entityHeatmaps(entityData, numMaps=10):
        heatmaps = {i: np.zeros((16, 28), dtype=np.float32) for i in range(1, numMaps)}  # 9 heatmaps for types 1-9

        # Define sigma values per entity type
        sigma_values = {1:.2, 2:.2, 3:.2, 4:.2, 5:.2, 6:.2, 7:.2, 8:.2, 9:.2, 10:.2}

        x = np.linspace(0, 27, 28)
        y = np.linspace(0, 15, 16)
        X, Y = np.meshgrid(x, y)
        for entity in entityData:
            entity_type, _, xpos, ypos, *_ = entity
            entity_type = int(entity_type)
            entity_type = 9 if entity_type == 10 else entity_type

            if entity_type not in heatmaps:
                print("Entity heatmap problem:", entity_type, entity)
                continue

            # Convert world coordinates to grid space
            norm_x = (xpos - room_x_min) / room_width
            norm_y = (ypos - room_y_min) / room_height
            grid_x = norm_x * 27
            grid_y = norm_y * 15

            # Get sigma and apply Gaussian
            sigma = sigma_values[entity_type]
            entity_gaussian = np.exp(-(((X - grid_x) ** 2) / (2 * sigma ** 2) + ((Y - grid_y) ** 2) / (2 * sigma ** 2)))
            entity_gaussian /= entity_gaussian.max()
            entity_gaussian[entity_gaussian < 0.01] = 0

            heatmaps[entity_type] += entity_gaussian

        # Stack into (9,16,28) tensor
        finalNP = np.stack([heatmaps[i] for i in range(1, numMaps)], axis=0)

        return finalNP  # Shape: (9,16,28)

    def hsv_to_bgr(h, s, v):
        """Convert HSV values (0-360, 0-255, 0-255) to BGR color for OpenCV."""
        hsv_color = np.uint8([[[h // 2, s, v]]])  # OpenCV scales hue 0-180
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(int(c) for c in bgr_color)  # Convert to (B, G, R)

    def drawOverlay():
        channelIndex = timer = 0
        tile_size = 4
        while True:
            timer += 1
            sleep(1/30)
            overlay = np.zeros((500, 250, 3), dtype=np.uint8)
            try:
                roomChannelsToDraw = [2] + [5, 6, 7, 8, 9, 10, 11, 12, 13, 14][:useHeatmaps] if useHeatmaps > 0 else [2]
                non_zero_mask = np.any(roomGridSection[:, :, roomChannelsToDraw] > 0, axis=2)
                y_indices, x_indices = np.where(non_zero_mask)
                top_left_coords = [((x * tile_size) + 2, (y * tile_size) + 10) for y, x in zip(y_indices, x_indices)]
                bottom_right_coords = [(((x + 1) * tile_size) + 2, ((y + 1) * tile_size) + 10) for y, x in zip(y_indices, x_indices)]
                for idx, (y, x) in enumerate(zip(y_indices, x_indices)):
                    tile = roomGridSection[y, x]
                    top_left = top_left_coords[idx]
                    bottom_right = bottom_right_coords[idx]

                    # Handle grid entity (special case due to value computation)
                    if tile[3] > 0 or tile[4] > 0:  # Grid entity or collision
                        value = 0.5 * tile[3] + 0.5 * tile[4]
                        brightness = int(min(50 + 160 * value, 255))
                        color = hsv_to_bgr(int(tile[2] * 180), 200, brightness)
                        cv2.rectangle(overlay, top_left, bottom_right, color, -1)

                    if useHeatmaps > 0:
                        overlayTiles = [
                            (5, 120, tile[5]),  # Player heatmap (green)
                            (6, 0, tile[6]),    # Enemy (red)
                            (7, 30, tile[7]),   # Bomb (orange)
                            (8, 60, tile[8]),   # Pickup (yellow)
                            (9, 180, tile[9]),  # Enemy projectile (cyan)
                            (10, 240, tile[10]),  # Ally tear (blue)
                            (11, 270, tile[11]),  # Familiar (purple)
                            (12, 300, tile[12]),# Laser (magenta)
                            (13, 90, tile[13]), # Effect (light green)
                            (14, 150, tile[14]) # Slot + Beggar (teal)
                        ]
                        for index, hue, _ in overlayTiles:  # Skip grid entity
                            if tile[index] > 0:
                                brightness = int(min(50 + 160 * tile[index], 255))
                                color = hsv_to_bgr(hue, 200, brightness)
                                cv2.rectangle(overlay, top_left, bottom_right, color, -1)

                for gridIndex, room_data in floorGridDict.items():
                    if len(floorGridDict) > 1 and room_data["Type"] > 0:
                        row_idx, col = divmod(gridIndex, 13)
                        x_coord = col / 12.0
                        y_coord = row_idx / 12.0

                        top_left = (int(x_coord * 12 * tile_size) + 120, int(y_coord * 12 * tile_size) + 10)
                        bottom_right = (int((x_coord * 12 + 1) * tile_size) + 120, int((y_coord * 12 + 1) * tile_size) + 10)

                        if room_data["Visited"] == 1 or room_data["Clear"] == 1:
                            outline = -1  # Filled rectangle
                        else:
                            outline = 1   # Outline only
                        multiplier = 2 if room_data["Current"] == 1 else 1.5 #for brightness
                        hue = int((room_data["Type"]/29) * 180)
                        color = hsv_to_bgr(hue, 150, 100 * multiplier)
                        cv2.rectangle(overlay, top_left, bottom_right, color, outline)

                if not stateCenteredOnPlayer:
                    player_pixel_x = int(agent_x / 10)
                    player_pixel_y = int(agent_y / 10)
                    cv2.circle(overlay, (int(player_pixel_x), int(player_pixel_y)), 2, (0, 255, 0), -1)

                    if entityData.size > 0:
                        for entity in entityData:
                            hue = int(entity[1] * 180)
                            color = hsv_to_bgr(hue, 150, 150)
                            entity_pixel_x = int(entity[2] / 10)
                            entity_pixel_y = int(entity[3] / 10)
                            cv2.circle(overlay, (int(entity_pixel_x), int(entity_pixel_y)), 1, color, -1)

                    if len(targets) > 0:
                        for target in targets:
                            x, y = target[0]
                            x = (x + 1) * 40
                            y = (y + 3) * 40
                            cv2.circle(overlay, (int(x / 10), int(y / 10)), 1, (255, 0, 0), -1)

                randomness = (entropy / np.log(action_size)) * 100
                randomness = min(max(randomness, 0), 100)

                reward_color = (0, 255, 0) if reward >= 0 else (0, 0, 255)
                reward_color = (255, 255, 255) if reward == 0.0 else reward_color
                cv2.putText(overlay, f"Total Reward: {total_reward:.2f}, Reward:", (2, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                (text_width, _), _ = cv2.getTextSize(f"Total Reward: {total_reward:.2f}, Reward:", cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                cv2.putText(overlay, f"{reward:+.3f}", (2 + text_width, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.3, reward_color, 1)
                #cv2.putText(overlay, f"Reset in {resetTimer + 1 - step_count} | Randomness: {randomness:.2f}%", (2, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(overlay, f"Reset in {agent.n_steps+1 - len(states)} | Confidence: {(100-randomness):.2f}%", (2, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                cv2.putText(overlay, f"Episode: {agent.episode_counter}", (2, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                if pathfinding:
                    cv2.circle(overlay, (int(agent_target_x / 10), int(agent_target_y / 10)), 2, (2, 2, 255), 0)
                    prob_grid = np.array(graphProbs).reshape(16, 28)  # Your 16x28 grid
                    prob_offset_x = 2
                    prob_offset_y = 115
                    for y in range(16):  # prob_grid is 16x28
                        for x in range(28):
                            prob = prob_grid[y, x]
                            if prob > 0:  # Only draw non-zero probabilities
                                top_left = ((x * tile_size) + prob_offset_x, (y * tile_size) + prob_offset_y)
                                bottom_right = (((x + 1) * tile_size) + prob_offset_x, ((y + 1) * tile_size) + prob_offset_y)
                                hue = int(240 - (prob * 240 / 100))  # 240 (blue) to 0 (red)
                                color = hsv_to_bgr(hue, 255, 200)
                                cv2.rectangle(overlay, top_left, bottom_right, color, -1)
                                # Optional transparency
                                overlay[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = \
                                    cv2.addWeighted(overlay[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], 0.5,
                                                    np.full((tile_size, tile_size, 3), color, dtype=np.uint8), 0.5, 0)
                else:
                    bar_width = 4  # Width of each bar
                    gap = 10  # Gap between bars
                    #actions = ["A", "D", "W", "S", "sL","sR","sU","sD","Bm","It","Cr","Dr","-","X"]
                    if action_size == 4:
                        actions = ["A","D","W","S"]
                    elif action_size == 8:
                        actions = ["A","D","W","S","sL","sR","sU","sD"]
                    if max(graphProbs) != 0:
                        for i, (value, letter) in enumerate(zip(graphProbs, actions)):
                            x1 = i * (bar_width + gap) + 3  # X-coordinate of the bar
                            y1 = 176  # Bottom of the bar (fixed)
                            bar_height = int((value / max(graphProbs)) * 66)  # Correct height scaling
                            x2 = x1 + bar_width
                            y2 = y1 - bar_height  # Move the top of the bar upwards

                            color = (int(1*value), int(2*value), int(1*value))  # Green color for the bars
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # Draw each bar
                            # Draw the letter below the bar
                            text_x = x1 + bar_width // 4  # Center the letter under the bar
                            text_y = y1 + 10  # Place slightly below the bar
                            cv2.putText(overlay, letter, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                #learn() progress bar
                cv2.rectangle(overlay, (0,190),(int((len(states) / agent.n_steps) * overlay.shape[1]),200), (200,200,200), -1)
                cv2.rectangle(overlay, (0,190),(int((agent.progress / 100) * overlay.shape[1]),200), (50,200,50), -1)

                if is_pressed('7') and agent.policy.visualDataIndex > 0 and timer > 10:
                    agent.policy.visualDataIndex -= 1
                    channelIndex = 0
                    timer = 0
                    print("Graph Index:",agent.policy.visualDataIndex)
                if is_pressed('9') and agent.policy.visualDataIndex < agent.policy.numVisualData-1 and timer > 5:
                    agent.policy.visualDataIndex += 1
                    channelIndex = 0
                    timer = 0
                    print("Graph Index:",agent.policy.visualDataIndex)

                visualData = agent.policy.visualData[agent.policy.visualDataIndex].detach().to("cpu", non_blocking=True)

                # Convert visualData to numpy based on shape
                if len(visualData.shape) == 4:
                    layers = visualData[0].numpy()  # First batch item
                elif len(visualData.shape) == 3:
                    layers = visualData.numpy()
                elif len(visualData.shape) == 2:
                    layers = visualData.unsqueeze(0).numpy()
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
                sleep(1)
                print("overlay:",e)

            cv2.imshow(f"Overlay{isaacNumber}", overlay)
            cv2.waitKey(1)

    def loadSharedModel():
        with modelLock:
            state_dict = dict(shared_model)
            state_dict = {key: value.to(agent.device) for key, value in state_dict.items()}
            agent.policy.load_state_dict(state_dict)
            #print(f"Isaac {isaacNumber}: Updated Model.")

    playerNormalization,action_size,pathfinding,useHeatmaps,roomChannels,stateX,stateY = initializeData

    entitiesNormalization = np.array([
        10,        # 0: Category
        1000,      # 1: Entity ID
        1200,      # 2: X pos
        1200,      # 3: Y pos
        30,        # 4: X vel negative
        30,        # 5: X vel
        30,        # 6: Y vel negative
        30,        # 7: Y vel
        10000,     # 8: HP, explosion dmg
        10000,     # 9: isInvincible, etc.
        10000,     # 10: collision dmg, etc.
        10000,     # 11: size, scale, etc.
        1e14       # 12: flags
    ], dtype=np.float32)

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

    roomTypeValues = {
        1: 1, #default
        2: 1.4, #shop
        3: 2, #Error
        4: 2, #Treasure
        5: 1.9, #boss
        6: 1.3, #miniboss
        7: 1.5, #secret
        8: 1.6, #supersecret
        9: 1.1, #arcade
        10: .9, #curse
        11: .9, #challenge
        12: 2, #library
        13: .5, #sacrifice
        14: 2, #devil
        15: 2, #angel
        16: 1, #crawlspace
        17: 1, #bossrush
        18: 1,
        19: 1,
        20: 1,
        21: .9, #dice
        22: 1,
        23: 1,
        24: 2, #planetarium
        25: 1,
        26: 1,
        27: 1,
        28: 1,
        29: 2
    }

    opposite_actions = {0: 1, 1: 0, 2: 3, 3: 2}
    shooting_actions = {4, 5, 6, 7}  # Only one can be active

    playerData = {key:0 for key in playerNormalization}
    playerData["items"] = []
    playerData["time_counter"] = 0
    totalHP = playerData["hp"]+playerData["soul_hp"]+playerData["black_hp"]+playerData["rotten_hp"]+playerData["bone_hp"]+playerData["eternal_hp"]+playerData["extra_lives"]

    #heatmap calc
    room_x_min, room_y_min = 20, 100
    room_x_max, room_y_max = 1140, 740
    room_width = room_x_max - room_x_min
    room_height = room_y_max - room_y_min

    actionStates = {i: 0 for i in range(12)}

    itemArray = np.zeros(50, dtype=np.float32)
    keyboardKeys = list(actionStates.values()) #12 keys
    reset,done = False,True
    resetEpisode = True
    lenEntitiesMemory = 200
    numEntityValues = 13
    num_additional_values = len(keyboardKeys)+len(playerNormalization)+len(itemArray)+(lenEntitiesMemory*numEntityValues)

    stateCenteredOnPlayer = True #The overall state varies more, might help with "just the player moving" not being enough difference between states.

    agent = agentCopy
    agent.isaacNumber = isaacNumber
    loadSharedModel()

    roomX,roomY = 15, 9
    overlay_thread = Thread(target=drawOverlay, daemon=True)
    overlay_thread.start()

    manualTesting = False

    #effect ids, still in progress, removed ones don't reach the state, allowed ones just don't get printed.
    removedEffects = np.array([2,3,4,5,7,11,12,13,14,15,16,17,20,21,27,33,38,43,58,59,63,64,65,66,68,79,80,86,99,133,146], dtype=float)
    allowedEffects = np.array([1,6,22,23,24,25,26,30,34,44,45,46,50,57,61,62], dtype=float)

    # Initialize rollout storage
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs_list = []
    values_list = []
    hidden_states = []

    print(f"Running Isaac {isaacNumber}...")

    with open(f"F:/IsaacInputs{isaacNumber}.txt", "w") as f:
        f.write("")

    freedom = True

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

            itemsSum = len(playerData["items"])
            previousPickups = pickups = playerData["bombs"] + playerData["coins"] + playerData["keys"]
            step_count = enemyDamage = lastEnemyDamage = punishX = punishY = total_reward = previousItemsSum = totalEnemyHP = previousEnemyHP = enemyDamage = 0
            roomHP = previousHP = totalHP
            visited_rooms = last_visited_rooms = 1
            lastRoom = 84
            resetTimer = 500

            target_potential_reward = {}
            floorGridDict = {}
            floorGridDict[84] = {"Visits":1}

            itemArray = np.zeros(50, dtype=np.float32)
            entitiesListFull = np.zeros((lenEntitiesMemory, numEntityValues), dtype=np.float32)
            totalRoomGrid = np.zeros((105, 183, roomChannels), dtype=np.float32)
            totalRoomGrid[:, :, 3] = 1 #Fill with solid collision, overwrite after.
            global_x, global_y = np.meshgrid(np.linspace(0, 1, 183), np.linspace(0, 1, 105))
            totalRoomGrid[:, :, 0] = global_x  # Global x-coordinate, normalized [0, 1]
            totalRoomGrid[:, :, 1] = global_y  # Global y-coordinate, normalized [0, 1]

            hidden_state = None
            resetEpisode = False

            sleep(1)

            stepsInRoom = 0

            while not done and not resetEpisode:

                while step_count != 0:
                    with open(f"F:/IsaacResponse{isaacNumber}.txt", "r") as f:
                        if msg == f.read().strip():
                            break  # Loop until msg is confirmed back

                step_count += 1
                stepsInRoom += 1

                visited_rooms = 0
                floorGrid = np.zeros((13, 13, 6), dtype=np.float32)
                rows, cols = np.indices((13, 13))
                floorGrid[:, :, 0] = cols.astype(np.float32) / 12.0
                floorGrid[:, :, 1] = rows.astype(np.float32) / 12.0
                floorData = np.loadtxt(f"F:/IsaacFloorData{isaacNumber}.txt", delimiter=",", dtype=np.int32)
                for row in floorData:
                    gridIndex, listIndex, roomType, visited, clear, current = row
                    row_idx, col = divmod(gridIndex, 13)
                    if roomType != 0:
                        floorGridDict.setdefault(gridIndex, {}).update({"ID": listIndex, "Type": roomType, "Visited": visited, "Clear": clear, "Current": current})
                        baseRoomValue = roomTypeValues.get(roomType, 0)
                        floorGridDict[gridIndex].setdefault("Value", baseRoomValue)
                        if clear == 1:
                            floorGridDict[gridIndex]["Value"] = 0
                        floorGrid[row_idx, col, 2:] = [roomType, visited, clear, current]
                        visited_rooms += visited
                floorGrid_normalized = floorGrid / np.array([1, 1, 29, 1, 1, 1], dtype=np.float32) #first two are X and Y
                out_of_bounds = (floorGrid_normalized < -1) | (floorGrid_normalized > 1)
                if np.any(out_of_bounds):
                    print("FloorGrid Normalization Problem:", floorGrid_normalized[out_of_bounds])
                    print(floorGrid)
                floorGrid_resized = np.transpose(floorGrid_normalized, (2, 0, 1))

                with open(f"F:/IsaacTileData{isaacNumber}.txt", "r") as f:
                    first_line = f.readline()
                    tile_data = f.readlines()
                room, roomX, roomY = map(int, first_line.split(","))
                roomGrid = np.array([list(map(int, line.strip().split(","))) for line in tile_data], dtype=np.int32)
                roomGrid = roomGrid.reshape((roomY, roomX, 3))
                roomGrid_normalized = roomGrid / np.array([27, 1, 1000], dtype=np.float32)
                out_of_bounds = (roomGrid_normalized < -1) | (roomGrid_normalized > 1)
                if np.any(out_of_bounds):
                    print("RoomGrid Normalization Problem:", roomGrid_normalized[out_of_bounds])
                    print("grid",roomGrid)
                #make tiles outside the room -.2 instead of 0,0,0 (walkable floor)
                empty_mask = np.all(roomGrid_normalized == [0, 0, 0], axis=-1)
                labeled_grid, num_features = label(empty_mask)
                for i in range(1, num_features + 1):
                    block_mask = labeled_grid == i
                    if np.any(block_mask[0, :]) or np.any(block_mask[-1, :]) or np.any(block_mask[:, 0]) or np.any(block_mask[:, -1]):
                        roomGrid_normalized[block_mask] = [-.2, -.2, -.2]

                entityData = np.loadtxt(f"F:/IsaacEntityData{isaacNumber}.txt", delimiter=",", dtype=np.float32)
                if entityData.ndim == 1:
                    entityData = entityData.reshape(1, -1)
                if entityData.size > 0:
                    # Filter: keep if not category 8, or if category 8 and effect not in removedEffects
                    keep_mask = np.logical_or(
                        entityData[:, 0] != 8,
                        ~np.isin(entityData[:, 8], removedEffects)
                    )
                    entityData = entityData[keep_mask]
                    # Debug: check for unexpected effects and print
                    debug_mask = np.logical_and(
                        entityData[:, 0] == 8,
                        ~np.isin(entityData[:, 8], allowedEffects)
                    )
                    for entity in entityData[debug_mask]:
                        print("Unknown Effect:",entity[0], entity[8])

                    normalizedEntities = entityData / entitiesNormalization  # shape: (N, 13)

                    # Normalization check
                    bad_mask = (normalizedEntities < 0) | (normalizedEntities > 1)
                    for i, row in enumerate(normalizedEntities):
                        for j, val in enumerate(row):
                            if bad_mask[i, j]:
                                print("Normalized entity problem:", j, val, entityData[i])

                    # Pad or truncate to fit into fixed memory slot
                    entitiesListFull = np.zeros((lenEntitiesMemory, numEntityValues), dtype=np.float32)
                    n = min(len(normalizedEntities), lenEntitiesMemory)
                    entitiesListFull[:n, :13] = normalizedEntities[:n]
                else:
                    entitiesListFull = np.zeros((lenEntitiesMemory, numEntityValues), dtype=np.float32)

                if useHeatmaps > 1:
                    entitiesGrids = entityHeatmaps(entityData)

                current_rooms = []
                for gridIndex, data in floorGridDict.items():
                    if data["Current"] == 1:
                        y, x = divmod(gridIndex, 13)  # row, col
                        current_rooms.append((x, y, gridIndex))  # (x, y, gridIndex)

                if len(current_rooms) != 0 and any(room_number == room for _, _, room_number in current_rooms):
                    minRoom_x = min(x for x, y, _ in current_rooms)
                    minRoom_y = min(y for x, y, _ in current_rooms)
                    target_x = 15 + (minRoom_x - 1) * 14 if minRoom_x > 0 else 0
                    target_y = 9 + (minRoom_y - 1) * 8 if minRoom_y > 0 else 0
                    maskCurrentRoom = np.any(roomGrid_normalized != -0.2, axis=-1)
                    totalRoomGrid[target_y:target_y + roomY, target_x:target_x + roomX, 2:5][maskCurrentRoom] = roomGrid_normalized[maskCurrentRoom]
                    if useHeatmaps > 1:
                        entitiesGrids = np.transpose(entitiesGrids, (1, 2, 0))  # Shape: (16, 28, 9)
                        totalRoomGrid[target_y:target_y + roomY, target_x:target_x + roomX, roomChannels-useHeatmaps-1:roomChannels][maskCurrentRoom] = entitiesGrids[:roomY, :roomX, :][maskCurrentRoom]

                    playerData, totalHP, dataValues = readGameData(isaacNumber, playerData)
                    agent_x = playerData["x"] * playerNormalization["x"]
                    agent_y = playerData["y"] * playerNormalization["y"]
                    playerGrid, playerTileX, playerTileY = playerHeatmap(agent_x, agent_y)
                    if useHeatmaps > 0:
                        totalRoomGrid[:, :, 5] = 0  # Clear player heatmap channel
                        totalRoomGrid[target_y:target_y + roomY, target_x:target_x + roomX, 5] = playerGrid[:roomY, :roomX]
                        if pathfinding:
                            targetGrid, _, _ = playerHeatmap(agent_target_x, agent_target_y)
                            totalRoomGrid[:, :, 6] = 0  # Clear target heatmap channel
                            totalRoomGrid[target_y:target_y + roomY, target_x:target_x + roomX, 6] = targetGrid[:roomY, :roomX]

                    if stateCenteredOnPlayer:
                        section_left = target_x + playerTileX - stateX // 2
                        section_top = target_y + playerTileY - stateY // 2

                        gridHeight, gridWidth, roomChannels = totalRoomGrid.shape
                        start_y = max(0, section_top)  # Avoid negative indices
                        start_x = max(0, section_left)
                        end_y = min(gridHeight, section_top + stateY)  # Don't exceed grid height
                        end_x = min(gridWidth, section_left + stateX)  # Don't exceed grid width
                        roomGridSection = np.zeros((stateY, stateX, roomChannels), dtype=totalRoomGrid.dtype)

                        dest_start_y = max(0, -section_top)  # Offset if section_top is negative
                        dest_start_x = max(0, -section_left)  # Offset if section_left is negative
                        dest_end_y = stateY - max(0, (section_top + stateY) - gridHeight)  # Trim if exceeding height
                        dest_end_x = stateX - max(0, (section_left + stateX) - gridWidth)  # Trim if exceeding width

                        roomGridSection[dest_start_y:dest_end_y, dest_start_x:dest_end_x, :] = totalRoomGrid[start_y:end_y, start_x:end_x, :]

                    else:
                        roomGridSection = totalRoomGrid[target_y:target_y+stateY, target_x:target_x+stateX, :]

                    roomGridSectionF = np.transpose(roomGridSection, (2, 0, 1))  # Shape: (14, 16, 28)

                    itemArray = np.zeros(50, dtype=np.float32)
                    if len(playerData["items"]) > 0:
                        normalized_values = [value / 800 for value in playerData["items"]]
                        itemArray[:len(normalized_values)] = normalized_values
                    additional_values = np.concatenate([np.array(keyboardKeys, dtype=np.float32), dataValues, itemArray, entitiesListFull.flatten()])

                    # 0 and 1 are x y coordinates, 2 to 4 are the tile id, collision and state. 5 is player heatmap. 6 to 14 are entities. Enemy,bomb,pickup,enemy proj,ally tear,familiar,laser,effect,slot+beggar.
                    stateNumpy = roomGridSectionF
                    floorGridNumpy = floorGrid_resized
                    additionalValuesNumpy = additional_values

                    with torch.no_grad():
                        logits, value, hidden_state = agent.policy(stateNumpy, floorGridNumpy, additionalValuesNumpy, hidden_state)
                        probs = F.softmax(logits, dim=-1)
                        action = torch.multinomial(probs, 1).item()
                        log_prob = probs.log().gather(1, torch.tensor([[action]], device=agent.device))
                        entropy = -torch.sum(probs * torch.log(probs + 1e-6))
                        min_prob = probs.min()
                        max_prob = probs.max()
                        probs = ((probs - min_prob) / (max_prob - min_prob + 1e-6)) * 100
                        graphProbs = probs.squeeze().tolist()

                    if not manualTesting:
                        if pathfinding:
                            agent_target_x = 40.0 + (action % 28) * 40
                            agent_target_y = 120.0 + (action // 28) * 40
                            execute = (agent_target_x, agent_target_y)
                        else:
                            execute = action
                    else:
                        execute = action_size

                    if isinstance(execute, int):
                        if execute < action_size:
                            if execute in opposite_actions:
                                opposite = opposite_actions[execute]
                                actionStates[opposite] = 0
                            elif execute in shooting_actions:
                                for shoot_action in shooting_actions:
                                    actionStates[shoot_action] = 0
                            actionStates[execute] = 1
                            msg = " ".join(f"{k} {v}" for k, v in actionStates.items()) + f" {step_count}"
                        if execute == action_size:
                            msg = f"{step_count}"
                    else:
                        msg = f"target_position:{execute[0]},{execute[1]} {step_count}"

                    with open(f"F:/IsaacInputs{isaacNumber}.txt", "w") as f:
                        f.write(msg)

                    keyboardKeys = list(actionStates.values())
                    ######################################Rewards##########################################
                    with open(f"F:/IsaacEnemyDamage{isaacNumber}.txt", "r") as f:
                        currentEnemyDamage = float(f.read())

                    damageDifference = currentEnemyDamage - lastEnemyDamage

                    itemsSum = len(playerData["items"])
                    pickups = playerData["bombs"] + playerData["coins"] + playerData["keys"]

                    reward = 0.0
                    reward += (itemsSum - previousItemsSum)*10 + (totalHP - previousHP)*10 + (damageDifference / 10)

                    if pickups > previousPickups:
                        reward += 2 * pickups * 100 #pickups are normalized to 100
                        previousPickups = pickups

                    if lastRoom != room:
                        visits = floorGridDict[room].get("Visits", 0) + 1
                        floorGridDict[room]["Visits"] = visits
                        reward += floorGridDict[room].get("Value", 0)/(1+visits)
                        lastRoom = room
                        stepsInRoom = 0
                        if not freedom:
                            done = True

                    if visited_rooms > last_visited_rooms and visited_rooms > 1:
                        roomHP = totalHP
                        reward += 10
                        resetTimer += 250
                        last_visited_rooms = visited_rooms

                    """if stepsInRoom > 200 and room == 84:
                        reward += -.1"""

                    if (actionStates[0] == 1 or actionStates[1] == 1) and previousX == playerData["x"]:
                        punishX += 1
                        if punishX > 30:
                            reward += -.01
                    else:
                        punishX = 0
                    if (actionStates[2] == 1 or actionStates[3] == 1) and previousY == playerData["y"]:
                        punishY += 1
                        if punishY > 30:
                            reward += -.01
                    else:
                        punishY = 0

                    if actionStates[8] == 1 and playerData["bombs"] == 0 and playerData["golden_bomb"] == 0:
                        reward += -1
                    if actionStates[9] == 1 and playerData["full_charge1"] == 0:
                        reward += -1
                    if actionStates[10] == 1 and playerData["card"] == 0 and playerData["pill"] == 0:
                        reward += -1
                    if actionStates[11] == 1 and playerData["trinket1"] == 0:
                        reward += -1

                    if currentFloor != playerData["stage"]:
                        playerData, totalHP, dataValues = readGameData(isaacNumber, playerData)
                        if currentFloor != playerData["stage"]:
                            print(f"{isaacNumber}. Floor changed...")
                            target_potential_reward = {}
                            totalRoomGrid = np.zeros((105, 183, roomChannels), dtype=np.float32)
                            totalRoomGrid[:, :, 3] = 1 #Fill with solid collision, overwrite after.
                            global_x, global_y = np.meshgrid(np.linspace(0, 1, 183), np.linspace(0, 1, 105))
                            totalRoomGrid[:, :, 0] = global_x  # Global x-coordinate, normalized [0, 1]
                            totalRoomGrid[:, :, 1] = global_y  # Global y-coordinate, normalized [0, 1]
                            reward += 50

                    targets = []

                    if playerData["extra_lives"] != 0:#-------------------------------
                        print("lives:",playerData["extra_lives"])


                    if pathfinding:
                        if entityData.size > 0:
                            for entity in entityData:
                                if entity[0] == 3:
                                    target_room = current_rooms[0][2]
                                    x = (entity[2] / 40) - 1
                                    y = (entity[3] / 40) - 3
                                    targets.append([(x, y), [target_room % 13, target_room // 13], target_room])

                        for y in range(roomY):
                            for x in range(roomX):
                                tile = roomGrid[y][x]
                                if tile[0] == 16 and tile[2] == 2:  # Door and open
                                    if (x, y) in door_mappings:
                                        target_room = door_mappings[(x, y)](current_rooms)
                                    else:
                                        print("Door not listed", x, y)
                                        continue
                                    if target_room:
                                        targets.append([(x, y), target_room, current_rooms[0][2]])
                                elif tile[0] in [17, 23, 20]:  # Non-door targets
                                    target_room = current_rooms[0][2]
                                    targets.append([(x, y), [target_room % 13, target_room // 13], target_room])

                        # Process targets and scale base rewards
                        for target in targets:
                            (xtile, ytile), target_target_room, target_current_room = target
                            target_target_room_idx = target_target_room[1] * 13 + target_target_room[0]
                            key = (target_current_room, xtile, ytile, target_target_room_idx)
                            base_reward = floorGridDict[target_target_room_idx]["Value"]

                            target_total_reward = base_reward
                            for existing_key in target_potential_reward:
                                if existing_key[0] == target_target_room_idx:
                                    if len(existing_key) >= 4 and existing_key[3] == target_current_room:
                                        continue
                                    target_total_reward += target_potential_reward[existing_key] * 0.9
                            floorGridDict.setdefault(target_target_room_idx, {})
                            visit_count = floorGridDict[target_target_room_idx].get("Visits", 0)
                            decay_factor = max(0.2, 1.0 - 0.2 * (visit_count - 1))

                            target_potential_reward[key] = target_total_reward * decay_factor

                        # Reward calculation for actions

                        action_x = action % 28
                        action_y = action // 28
                        if action_x < roomX and action_y < roomY:
                            target_tile = roomGrid_normalized[action_y][action_x]
                            if -0.2 not in target_tile:
                                tile_collision = target_tile[1]

                                # Find the highest reward for the current room
                                max_reward = float('-inf')
                                for key in target_potential_reward:
                                    current_room, xtile, ytile, target_room_idx = key
                                    if current_room == room:
                                        max_reward = max(max_reward, target_potential_reward[key])

                                # Only reward if the action matches a target with the highest reward
                                for key in target_potential_reward:
                                    current_room, xtile, ytile, target_room_idx = key
                                    if xtile == action_x and ytile == action_y and current_room == room:
                                        if target_potential_reward[key] == max_reward:  # Only reward if it's the highest
                                            reward += target_potential_reward[key] * 1  # Scaled: 0.01 * 100 = 1
                                        break
                                else:  # No matching target found
                                    if tile_collision == 1:
                                        reward += -0.01
                                    else:  # No collision
                                        reward += 0.01
                            else:  # Target is unwalkable tile
                                reward += -0.01

                    total_reward += reward

                if not manualTesting:
                    # Done conditions
                    #if totalHP == 0 or step_count > resetTimer :
                    if totalHP == 0 or done or step_count>agent.n_steps:
                        for k in actionStates:
                            actionStates[k] = 0
                        msg = " ".join(f"{k} {v}" for k, v in actionStates.items()) + " reset"
                        resetEpisode = True
                        with open(f"F:/IsaacInputs{isaacNumber}.txt", "w") as f:
                            f.write(msg)
                        done = True
                        #print(f"{isaacNumber}. Main loop step counter: {step_count}, Episode {agent.episode_counter}, Total Reward: {total_reward}, Per step: {(total_reward/step_count):.2f}")
                        #freedom = True
                        total_reward = 0
                        episodeSteps += step_count

                    states.append((stateNumpy, floorGridNumpy, additionalValuesNumpy))
                    actions.append(action)
                    log_probs_list.append(log_prob.item())
                    values_list.append(value.item())
                    hidden_states.append((hidden_state[0].cpu(), hidden_state[1].cpu()))
                    rewards.append(reward)
                    dones.append(done)
                else:
                    if playerData["time_counter"] <= 1:
                        done = True

                currentFloor = playerData["stage"]
                previousX = playerData["x"]
                previousY = playerData["y"]
                previousHP, previousItemsSum, previousEnemyHP, lastEnemyDamage = totalHP, itemsSum, totalEnemyHP, currentEnemyDamage

                #if done and len(states) > (agent.n_steps - resetTimer):
                if done or len(states) > agent.n_steps:
                    if len(states) == agent.n_steps+2:
                        episodeSteps = 1
                        if sum(rewards[:-1]) != 0:
                            rollout_queue.put((states, actions[:-1], rewards[:-1], dones[:-1], log_probs_list[:-1], values_list[:-1], hidden_states[:-1]))
                            agent.episode_counter += 1
                            print(f"Isaac {isaacNumber}: Sent rollout, Episode completed.")
                        states = [states[-1]]
                        actions = [actions[-1]]
                        rewards = [rewards[-1]]
                        dones = [dones[-1]]
                        log_probs_list = [log_probs_list[-1]]
                        values_list = [values_list[-1]]
                        hidden_states = [hidden_states[-1]]
                    else:
                        loadSharedModel()
                        episodeSteps = 0

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Required for CUDA in multiprocessing
    manager = mp.Manager()
    control_dict = manager.dict()  # Controls whether instances should run
    shared_model = manager.dict()
    processes = {}  # Store running processes
    modelLock = mp.Lock()
    rollout_queue = mp.Queue()

    playerNormalization = {
        "x": 1200,
        "y": 1200,
        "vxneg": 30,
        "vx": 30,
        "vyneg": 30,
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
    action_size = 8
    pathfinding = True if action_size > 100 else False
    useHeatmaps = 0 #0 or 10, maybe 1 for only player
    roomChannels = 6+useHeatmaps if pathfinding else 5+useHeatmaps
    stateX,stateY = 28, 16

    initializeData = (playerNormalization,action_size,pathfinding,useHeatmaps,roomChannels,stateX,stateY)

    learnerAgent = PPOAgent(
        room_shape=(roomChannels, stateY, stateX),
        map_shape=(6, 13, 13),action_size=action_size,
        n_critical=12+len(playerNormalization),
        n_items=50,
        n_entity_memory=200*13,
        isaacNumber=0,
        shared_model=shared_model,
        modelLock=modelLock,
        rollout_queue=rollout_queue)

    # Load the best model and initialize shared_model with CPU tensors
    try:
        learnerAgent.load("F:/isaacPPOModel.pth")
        print(f"Loaded best model and initialized shared_model.")
    except Exception as e:
        print(f"No model found: {e}")

    with modelLock:
        cpu_state_dict = {key: value.cpu() for key, value in learnerAgent.policy.state_dict().items()}
        shared_model.clear()
        shared_model.update(cpu_state_dict)

    # Start learner process
    learner_process = mp.Process(target=learnerAgent.run)
    learner_process.start()

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
                            p = mp.Process(
                                target=run_instance,
                                args=(isaacNumber, initializeData, control_dict, shared_model, modelLock, learnerAgent, rollout_queue)
                            )
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
