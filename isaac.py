from time import sleep
from keyboard import is_pressed
from threading import Thread
from scipy.ndimage import label
import numpy as np
import cv2,torch
torch.set_printoptions(profile="full")

from isaacPPO import PPOPolicy, PPOAgent

def update_actionStates():#not updated
    for action, key in key_map.items():
        actionStates[action] = int(is_pressed(key))  # 1 if pressed, else 0

def deduce_action():#not updated
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
    global actionStates, actionCounter
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


    # Send full state to Lua
    msg = " ".join(f"{k} {v}" for k, v in actionStates.items()) + f" {actionCounter}"
    with open(f"F:/IsaacInputs{isaacNumber}.txt", "w") as f:
        f.write(msg)
    while True:
        with open(f"F:/IsaacResponse{isaacNumber}.txt", "r") as file:
            if msg == file.read().strip():
                break  # Loop until msg is confirmed back

    return list(actionStates.values())  # Return full state list

def readGameData():
    global last_data, playerData, totalHP, itemArray, dataValues
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
            break
        except:
            pass

def playerHeatmap(player_x, player_y, sigma=.25):
    grid_h, grid_w = 16, 28
    overlay = np.zeros((grid_h, grid_w), dtype=np.float32)

    # Normalize player position to grid space (floating point)
    norm_x = (player_x - room_x_min) / (room_x_max - room_x_min)
    norm_y = (player_y - room_y_min) / (room_y_max - room_y_min)

    grid_x = norm_x * (grid_w - 1)
    grid_y = norm_y * (grid_h - 1)

    # Generate meshgrid for all (x, y) positions
    x = np.linspace(0, grid_w - 1, grid_w)
    y = np.linspace(0, grid_h - 1, grid_h)
    X, Y = np.meshgrid(x, y)

    # Compute 2D Gaussian centered at (grid_x, grid_y)
    heatmap = np.exp(-(((X - grid_x) ** 2) / (2 * sigma ** 2) + ((Y - grid_y) ** 2) / (2 * sigma ** 2)))

    # Normalize to [0, 1]
    heatmap /= heatmap.max()

    return heatmap

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
    visualDataIndex = timer = 0
    while True:
        timer += 1
        sleep(1/60)
        overlay = np.zeros((400,256,3), dtype=np.uint8)
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

            cv2.rectangle(overlay, (5,190),(5+int((len(states) / agent.n_steps) * 180),200), (200,200,200), -1)
            cv2.rectangle(overlay, (5,190),(5+int((agent.progress / 100) * 180),200), (50,200,50), -1)

            if agent.policy.visualData is not None:
                #Getting these out of the GPU into the CPU costs the CPU some work, so careful.
                if is_pressed('7') and visualDataIndex > 0 and timer > 10:
                    visualDataIndex -= 1
                    timer = 0
                    print("Graph Index:",visualDataIndex)
                if is_pressed('9') and visualDataIndex < len(agent.policy.visualData)-1 and timer > 5:
                    visualDataIndex += 1
                    timer = 0
                    print("Graph Index:",visualDataIndex)

                visualData = agent.policy.visualData[visualDataIndex].detach().clone().cpu()

                if len(visualData.shape) == 4:
                    graph = visualData[0].sum(dim=0).to("cpu", non_blocking=True).numpy()
                elif len(visualData.shape) == 3:
                    graph = visualData.sum(dim=0).to("cpu", non_blocking=True).numpy()
                elif len(visualData.shape) == 2:
                    graph = visualData.contiguous().to("cpu", non_blocking=True).numpy()  # Extract from first channel
                else:
                    raise ValueError(f"Unexpected number of channels: {visualData.shape}")

                graph = cv2.normalize(graph, None, 0, 200, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                graph = cv2.applyColorMap(graph, cv2.COLORMAP_JET)
                graph = cv2.resize(graph, (graph.shape[1]*6, graph.shape[0]*6), interpolation=cv2.INTER_NEAREST)
                graphH, graphW, _ = graph.shape
                overlay[205:205+graphH, 5:5+graphW] = graph

            cv2.imshow(f"Overlay{isaacNumber}", overlay)
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

isaacNumber = int(input("Number of Isaac: "))

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

clipParam = 0.2*isaacNumber
value_loss_coef = 0.2*isaacNumber
agent = PPOAgent(room_shape=(13, 16, 28),map_shape=(6, 13, 13),clip_param=clipParam,value_loss_coef=value_loss_coef,action_size=action_size,n_critical=len(keyboardKeys) + len(playerNormalization),n_items=len(itemArray),n_entity_memory=(lenEntitiesMemory * numEntityValues), isaacNumber=isaacNumber)

emptyFloorTensor = torch.empty((1,6,13,13), dtype=torch.float32, device="cuda")
emptyAdditionalValues = torch.empty((1,num_additional_values), dtype=torch.float32, device="cuda")
emptyFinalState = torch.empty((1,13,16,28), dtype=torch.float32, device="cuda")

overlay_thread = Thread(target=drawOverlay, daemon=True)


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
hidden_states = []
hidden_state = None

#effect ids, still in progress, removed is ignored, allowed just doesn't get printed.
removedEffects = {2,3,4,5,7,11,12,13,14,15,16,17,20,21,27,33,38,43,58,59,63,64,65,66,68,79,86,99,133,146}
allowedEffects = {1,6,22,23,24,25,26,34,44,45,46,50,57,61,62}

print("Running...")
overlay_thread.start()
if not manualLearning:
    try:
        agent.load(f"F:/isaac_ppo_model{isaacNumber}.pth")
    except Exception as e:
        print("No model found to load...", e)
        try:
            agent.load(f"F:/isaac_ppo_model_backup{isaacNumber}.pth")
        except Exception as e:
            print("No backup model found to load...", e)


with open(f"F:/IsaacInputs{isaacNumber}.txt", "w") as f:
    pass

positiveReward = 0
freedom = False

while True:
    sleep(1/60)
    readGameData()
    currentFloor = playerData["stage"]

    print(playerData["time_counter"])
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
        readGameData()
        previousX = playerData["x"]
        previousY = playerData["y"]
        currentFloor = playerData["stage"]
        floorGrid_tensor = emptyFloorTensor
        additional_values_torch = emptyAdditionalValues
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

        sleep(1)
        print("Enter Episode")
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
            readGameData()
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

            final_next_state = emptyFinalState.copy_(torch.from_numpy(roomGridSection).unsqueeze(0)) # 0 to 2 are the tile id, collision and state. 3 is player heatmap. 4 to 12 are entities. Enemy,bomb,pickup,enemy proj,ally tear,familiar,laaser,effect,slot+beggar.
            next_floorGrid_tensor = emptyFloorTensor.copy_(torch.from_numpy(floorGrid_resized).unsqueeze(0))
            next_additional_values_tensor = emptyAdditionalValues.copy_(torch.from_numpy(additional_values).unsqueeze(0))

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

            """if (actionStates[0] == 1 or actionStates[1] == 1) and previousX == playerData["x"]:
                punishX += 1
                if punishX > 2:
                    reward -= .1
            else:
                punishX = 0
            if (actionStates[2] == 1 or actionStates[3] == 1) and previousY == playerData["y"]:
                punishY += 1
                if punishY > 2:
                    reward -= .1
            else:
                punishY = 0"""

            if actionStates[8] == 1 and playerData["bombs"] == 0 and playerData["golden_bomb"] == 0:
                reward -= .1
            if actionStates[9] == 1 and playerData["full_charge1"] == 0:
                reward -= .1
            if actionStates[10] == 1 and playerData["card"] == 0 and playerData["pill"] == 0:
                reward -= .1
            if actionStates[11] == 1 and playerData["trinket1"] == 0:
                reward -= .1

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
                key = (xtile, ytile, target_room[0], target_room[1])  # Unique identifier for the door
                targetRoomType = floorGrid_normalized[target_room[1], target_room[0], 2]
                #print(targetRoomType)
                targetMultiplier = 2 if targetRoomType == 0.13333334 else 1
                distance = ((agent_x - door_x) ** 2 + (agent_y - door_y) ** 2) ** 0.5
                if key not in door_min_distances:
                    door_min_distances[key] = distance
                if distance < door_min_distances[key]:
                    closer_to_any_door = True
                    distance_covered = door_min_distances[key] - distance
                    total_distance = door_min_distances[key]
                    if cleared_status == 0:
                        reward_increment = (1 / distance) * 1  # Full reward
                    else:
                        reward_increment = (1 / distance) * 0.2
                    reward += reward_increment*targetMultiplier
                    door_min_distances[key] = distance  #Update door distance

                """if not closer_to_any_door:
                    (xtile, ytile), cleared_status, (door_x, door_y), target_room = target
                    key = (xtile, ytile, target_room[0], target_room[1])  # Unique identifier for the door, including room coordinates
                    distance = ((agent_x - door_x) ** 2 + (agent_y - door_y) ** 2) ** 0.5
                    if distance > door_min_distances[key] and door_min_distances[key] > 20 and cleared_status == 0:
                        punishment = -.01
                        reward += punishment*targetMultiplier"""

            visited_rooms = sum(1 for y in range(floorGrid_normalized.shape[0]) for x in range(floorGrid_normalized.shape[1]) if floorGrid_normalized[y, x][3] == 1)
            if visited_rooms > 1 and visited_rooms > last_visited_rooms:
                roomHP = totalHP
                reward += 100
                resetTimer += 250
            if lastRoom != room:
                reward += .01
            if playerData["alive_enemies"] > 0 and totalHP >= roomHP:
                reward += .01
                resetTimer += .5

            if currentFloor != playerData["stage"]:
                readGameData()
                if currentFloor != playerData["stage"]:
                    print("Floor changed...")
                    door_min_distances = {}
                    totalRoomGrid = np.zeros((105, 183, 13), dtype=np.float32)
                    totalRoomGrid[:, :, :3] = -0.2
                    reward += 10

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
                    print(f"Main loop step counter: {step_count}, Episode {agent.episode_counter}, Total Reward: {total_reward}")
                    positiveReward += 1 if total_reward > 0 else -1
                    freedom = True if positiveReward > 20 else False
                    total_reward = 0
                states.append((final_state.clone().detach(), floorGrid_tensor.clone().detach(), additional_values_torch.clone().detach()))
                actions.append(action)
                rewards.append(reward)
                next_states.append((final_next_state.clone().detach(), next_floorGrid_tensor.clone().detach(), next_additional_values_tensor.clone().detach()))
                dones.append(done)

            currentFloor = playerData["stage"]
            previousX = playerData["x"]
            previousY = playerData["y"]
            previousHP, previousItemsSum, previousEnemyHP, previousPickups, lastRoom, last_visited_rooms, lastEnemyDamage = totalHP, itemsSum, totalEnemyHP, pickups, room, visited_rooms, currentEnemyDamage
            final_state = final_next_state
            floorGrid_tensor = next_floorGrid_tensor
            additional_values_torch = next_additional_values_tensor

            if done and len(states) > agent.n_steps:
                print(f"Episode {agent.episode_counter}, Rewards: Mean: {np.mean(rewards)}, Std: {np.std(rewards)}")
                agent.learn((states, actions, rewards, next_states, dones, log_probs_list, values_list))
                states.clear()
                actions.clear()
                rewards.clear()
                next_states.clear()
                dones.clear()
                log_probs_list.clear()
                values_list.clear()
                hidden_states.clear()
                hidden_state = None  # Reset hidden state at episode boundary

                torch.cuda.empty_cache()

                agent.episode_counter += 1
                if not manualLearning and not manualTesting:
                    agent.save(f"F:/isaac_ppo_model{isaacNumber}.pth")
                    if agent.episode_counter % 50 == 0:
                        agent.save(f"F:/isaac_ppo_model_backup{isaacNumber}.pth")
