local mod = RegisterMod("AIsaac_Combined_Mod", 1)

-- ====================== VARIABLES ======================
local game = Game()
local inputs = {}
local lastInputData = ""  -- Track the last processed input
local newInputData = nil  -- Store new input data temporarily
local totalDamageDealt = 0  -- Total damage dealt to enemies
local enemyHPBefore = {}  -- Store previous HP of enemies

local sprite = Sprite()
local hasReset = false  -- Track if the game was reset
local totalDamageDealt = 0
local enemyHPBefore = {}
local instanceNumber = 0       -- Current active instance number
local pendingNumber = 1        -- Pending instance number (not applied yet)
local pendingChanges = false   -- Whether there are pending changes to apply

-- ====================== PATHFINDING VARIABLES ======================
local targetPosition = nil
local lastTargetPosition = nil

-- ====================== FILE PATHS ======================
local function GetFilePaths()
    -- If instance is 0, return dummy paths that won't be used
    if instanceNumber == 0 then
        return {
            playerData = nil,
            roomTileData = nil,
            damageData = nil,
            floorData = nil,
            entityData = nil,
            input = nil,
            response = nil
        }
    end

    -- Normal path generation for instances > 0
    return {
        playerData = "F:/IsaacData" .. instanceNumber .. ".txt",
        roomTileData = "F:/IsaacTileData" .. instanceNumber .. ".txt",
        damageData = "F:/IsaacEnemyDamage" .. instanceNumber .. ".txt",
        floorData = "F:/IsaacFloorData" .. instanceNumber .. ".txt",
        entityData = "F:/IsaacEntityData" .. instanceNumber .. ".txt",
        input = "F:/IsaacInputs" .. instanceNumber .. ".txt",
        response = "F:/IsaacResponse" .. instanceNumber .. ".txt"
    }
end

-- ====================== PATHFIND ======================
local waypoints = {}  -- Persistent waypoints

local function HandlePathfinding()
    if not targetPosition then
        waypoints = {}
        inputs[ButtonAction.ACTION_LEFT] = 0
        inputs[ButtonAction.ACTION_RIGHT] = 0
        inputs[ButtonAction.ACTION_UP] = 0
        inputs[ButtonAction.ACTION_DOWN] = 0
        return
    end

    local player = Isaac.GetPlayer(0)
    local room = game:GetRoom()
    local gridWidth = room:GetGridWidth()
    local gridSize = room:GetGridSize()
    local tileSize = 40

    local playerPos = player.Position
    local playerGridIndex = room:GetGridIndex(playerPos)
    local targetGridIndex = room:GetGridIndex(targetPosition)

    -- Validate target
    if targetGridIndex < 0 or targetGridIndex >= gridSize then
        waypoints = {}
        inputs[ButtonAction.ACTION_LEFT] = 0
        inputs[ButtonAction.ACTION_RIGHT] = 0
        inputs[ButtonAction.ACTION_UP] = 0
        inputs[ButtonAction.ACTION_DOWN] = 0
        return
    end
    local targetEntity = room:GetGridEntity(targetGridIndex)
    if targetEntity and (
        targetEntity.CollisionClass == GridCollisionClass.COLLISION_PIT or
        targetEntity.CollisionClass == GridCollisionClass.COLLISION_SOLID or
        targetEntity.CollisionClass == GridCollisionClass.COLLISION_WALL
    ) then
        waypoints = {}
        inputs[ButtonAction.ACTION_LEFT] = 0
        inputs[ButtonAction.ACTION_RIGHT] = 0
        inputs[ButtonAction.ACTION_UP] = 0
        inputs[ButtonAction.ACTION_DOWN] = 0
        return
    end

    -- Recalculate path if needed
    if #waypoints == 0 or waypoints[#waypoints]:Distance(targetPosition) > tileSize then
        waypoints = {}
        local openSet = { playerGridIndex }
        local cameFrom = {}
        local gScore = { [playerGridIndex] = 0 }
        local fScore = { [playerGridIndex] = playerPos:Distance(targetPosition) }
        local closedSet = {}

        while #openSet > 0 do
            local currentIndex = openSet[1]
            local lowestF = fScore[currentIndex]
            for i, index in ipairs(openSet) do
                if fScore[index] < lowestF then
                    currentIndex = index
                    lowestF = fScore[index]
                end
            end

            if currentIndex == targetGridIndex then
                local gridPath = {}
                local tempIndex = currentIndex
                while tempIndex do
                    table.insert(gridPath, 1, tempIndex)
                    tempIndex = cameFrom[tempIndex]
                end
                for _, gridIndex in ipairs(gridPath) do
                    table.insert(waypoints, room:GetGridPosition(gridIndex))
                end
                break
            end

            for i, index in ipairs(openSet) do
                if index == currentIndex then
                    table.remove(openSet, i)
                    break
                end
            end
            closedSet[currentIndex] = true

            -- Check all 8 neighbors (cardinal + diagonal)
            local neighbors = {
                currentIndex + 1,              -- Right
                currentIndex - 1,              -- Left
                currentIndex + gridWidth,      -- Down
                currentIndex - gridWidth,      -- Up
                currentIndex + 1 + gridWidth,  -- Down-Right
                currentIndex - 1 + gridWidth,  -- Down-Left
                currentIndex + 1 - gridWidth,  -- Up-Right
                currentIndex - 1 - gridWidth   -- Up-Left
            }
            for _, neighbor in ipairs(neighbors) do
                if neighbor >= 0 and neighbor < gridSize and not closedSet[neighbor] then
                    local gridEntity = room:GetGridEntity(neighbor)
                    if not gridEntity or (
                        gridEntity.CollisionClass == GridCollisionClass.COLLISION_NONE or
                        gridEntity.CollisionClass == GridCollisionClass.COLLISION_OBJECT or
                        gridEntity.CollisionClass == GridCollisionClass.COLLISION_WALL_EXCEPT_PLAYER
                    ) then
                        local isDiagonal = (neighbor == currentIndex + 1 + gridWidth or
                                           neighbor == currentIndex - 1 + gridWidth or
                                           neighbor == currentIndex + 1 - gridWidth or
                                           neighbor == currentIndex - 1 - gridWidth)
                        local cost = isDiagonal and tileSize * 1.414 or tileSize  -- Diagonal cost is sqrt(2) * tileSize
                        local tentativeG = gScore[currentIndex] + cost
                        if not gScore[neighbor] or tentativeG < gScore[neighbor] then
                            cameFrom[neighbor] = currentIndex
                            gScore[neighbor] = tentativeG
                            fScore[neighbor] = gScore[neighbor] + room:GetGridPosition(neighbor):Distance(targetPosition)
                            if not table.contains(openSet, neighbor) then
                                table.insert(openSet, neighbor)
                            end
                        end
                    end
                end
            end

            if #openSet > 50 then break end
        end

        if #waypoints == 0 then
            table.insert(waypoints, targetPosition)
        end
    end

    -- Move toward next waypoint with smoother inputs
    if #waypoints > 0 then
        local nextWaypoint = waypoints[1]
        local directionToNext = (nextWaypoint - playerPos):Normalized()
        local distanceToNext = playerPos:Distance(nextWaypoint)

        -- Smooth input values (0 to 1) based on direction
        local inputThreshold = 0.05  -- Lowered for finer control
        inputs[ButtonAction.ACTION_LEFT] = directionToNext.X < -inputThreshold and math.min(1, -directionToNext.X) or 0
        inputs[ButtonAction.ACTION_RIGHT] = directionToNext.X > inputThreshold and math.min(1, directionToNext.X) or 0
        inputs[ButtonAction.ACTION_UP] = directionToNext.Y < -inputThreshold and math.min(1, -directionToNext.Y) or 0
        inputs[ButtonAction.ACTION_DOWN] = directionToNext.Y > inputThreshold and math.min(1, directionToNext.Y) or 0

        -- Remove waypoint if close enough
        if distanceToNext < 15 then  -- Adjusted threshold
            table.remove(waypoints, 1)
        end
    else
        inputs[ButtonAction.ACTION_LEFT] = 0
        inputs[ButtonAction.ACTION_RIGHT] = 0
        inputs[ButtonAction.ACTION_UP] = 0
        inputs[ButtonAction.ACTION_DOWN] = 0
    end
end

function table.contains(tbl, value)
    for _, v in ipairs(tbl) do
        if v == value then return true end
    end
    return false
end
-- ====================== SPRITES SETUP ======================
-- Sprite filenames
local spriteFiles = {}
for i = 1, 51 do
    table.insert(spriteFiles, "keys" .. i .. ".anm2")
end
local currentSpriteFile = spriteFiles[math.random(#spriteFiles)]
sprite:Load(currentSpriteFile, true)

-- Key actions
local keyActions = {
    { action = 0, anim = "left", xOffset = 0, yPos = 380 },
    { action = 1, anim = "right", xOffset = 2, yPos = 380 },
    { action = 2, anim = "up", xOffset = 1, yPos = 350 },
    { action = 3, anim = "down", xOffset = 1, yPos = 380 },
    { action = 4, anim = "shoot_left", xOffset = 4, yPos = 380 },
    { action = 5, anim = "shoot_right", xOffset = 6, yPos = 380 },
    { action = 6, anim = "shoot_up", xOffset = 5, yPos = 350 },
    { action = 7, anim = "shoot_down", xOffset = 5, yPos = 380 },
    { action = 8, anim = "bomb", xOffset = 8, yPos = 380 },
    { action = 9, anim = "item", xOffset = 9, yPos = 380 },
    { action = 10, anim = "card", xOffset = 10, yPos = 380 },
    { action = 11, anim = "drop", xOffset = 11, yPos = 380 }
}

-- Grid entity types to ignore
local ignoredTypes = {
    [GridEntityType.GRID_DECORATION] = true,
    -- Add more as needed
}

-- Flag to enable/disable enemy removal (NoEnemies functionality)
local removeEnemies = true -- Set to true to enable enemy removal

-- ====================== HELPER FUNCTIONS ======================
local function ClearFiles()
    -- Clear/reset all data files
    local paths = GetFilePaths()
    local files = {
        paths.playerData,
        paths.roomTileData,
        paths.damageData,
        paths.floorData,
        paths.entityData,
        paths.response
    }

    for _, filepath in ipairs(files) do
        local file = io.open(filepath, "w")
        if file then
            file:write("")
            file:close()
        end
    end

    -- Reset variables
    totalDamageDealt = 0
    enemyHPBefore = {}
end

local function WriteFile(filepath, data)
    -- Skip file operations if instance is 0 or filepath is nil
    if instanceNumber == 0 or not filepath then
        return false
    end

    local file = io.open(filepath, "w")
    if file then
        file:write(data)
        file:close()
        return true
    end
    return false
end

-- Define resetGame before it's used
local function resetGame()
    currentSpriteFile = spriteFiles[math.random(#spriteFiles)]
    sprite:Load(currentSpriteFile, true)
    hasReset = true
    Isaac.ExecuteCommand("restart")
end

-- ====================== INPUT READING ======================
local function ReadInputsFromFile()
    local paths = GetFilePaths()
    local file = io.open(paths.input, "r")
    if not file then return false end

    local data = file:read("*a")
    file:close()

    if data:find("reset") then
        resetGame()
        lastInputData = data
        newInputData = data
        return true
    else
        if data == lastInputData then return false end
    end

    -- Reset inputs
    inputs = {}

    -- Parse target position
    local x, y = data:match("target_position:(%d+%.?%d*),(%d+%.?%d*)")
    if x and y then
        x, y = tonumber(x), tonumber(y)
        local room = game:GetRoom()
        targetPosition = Vector(x, y)
    else
        print("Failed to parse target position from: " .. data)
    end

    -- Store new input but don’t write response yet
    newInputData = data
    return true
end
-- ====================== FEATURE: DAMAGE TRACKING ======================
local function WriteDamageToFile()
    WriteFile(GetFilePaths().damageData, tostring(totalDamageDealt))
end

-- ====================== FEATURE: ENEMY REMOVAL ======================
local function RemoveEnemies()
    if not removeEnemies then return end

    for _, entity in ipairs(Isaac.GetRoomEntities()) do
        if entity:IsEnemy() then
            entity:Remove()
        end
    end
end

-- ====================== FEATURE: ROOM TILE LOGGING ======================
local function LogRoomTiles()
    local paths = GetFilePaths()
    local room = game:GetRoom()
    local level = game:GetLevel()
    local gridSize = room:GetGridSize()
    local roomIndex = level:GetCurrentRoomIndex()
    local roomX = room:GetGridWidth()
    local roomY = room:GetGridHeight()

    local tileData = {}

    -- Add room metadata to the first line
    table.insert(tileData, string.format("%d,%d,%d", roomIndex, roomX, roomY))

    for i = 0, gridSize - 1 do
        local gridEntity = room:GetGridEntity(i)

        local tileType = 0  -- Default for empty spaces
        local collisionClass = 0
        local state = 0

        if gridEntity then
            local entityType = gridEntity:GetType()

            -- If entity type is in the ignore list, set values to 0
            if not ignoredTypes[entityType] then
                tileType = entityType
                collisionClass = gridEntity.CollisionClass
                state = gridEntity.State
            end
        end

        -- Convert to string for comparison
        local tileString = string.format("%d, %d, %d", tileType, collisionClass, state)
        table.insert(tileData, tileString)
    end

    WriteFile(paths.roomTileData, table.concat(tileData, "\n"))
end

-- ====================== FEATURE: PLAYER DATA LOGGING ======================
local function WritePlayerData()
    local paths = GetFilePaths()
    local player = Isaac.GetPlayer(0)
    local level = game:GetLevel()
    local room = game:GetRoom()

    -- Get collected items safely
    local collectedItems = {}
    for i = 1, CollectibleType.NUM_COLLECTIBLES - 1 do
        if player:HasCollectible(i) then
            table.insert(collectedItems, tostring(i))
        end
    end
    local collectedItemsStr = "[" .. table.concat(collectedItems, "|") .. "]"

    -- Active items
    local active1 = player:GetActiveItem(ActiveSlot.SLOT_PRIMARY)
    local charge1 = player:GetActiveCharge(ActiveSlot.SLOT_PRIMARY)
    local NeedsCharge1 = player:NeedsCharge(ActiveSlot.SLOT_PRIMARY) and 1 or 0

    local active2 = player:GetActiveItem(ActiveSlot.SLOT_SECONDARY)
    local charge2 = player:GetActiveCharge(ActiveSlot.SLOT_SECONDARY)
    local NeedsCharge2 = player:NeedsCharge(ActiveSlot.SLOT_SECONDARY) and 1 or 0

    -- Room Data
    local firstVisit = room:IsFirstVisit() and 1 or 0
    local aliveEnemies = room:GetAliveEnemiesCount()
    local roomType = room:GetType()
    local stage = level:GetStage()

    -- Game Timer
    local timeCounter = game.TimeCounter

    -- Format the data
    local data = string.format(
        "x=%.2f,y=%.2f,vx=%.2f,vy=%.2f,hp=%d,max_hp=%d,soul_hp=%d,black_hp=%d,rotten_hp=%d,bone_hp=%d,eternal_hp=%d,extra_lives=%d,coins=%d,bombs=%d,keys=%d,golden_bomb=%d,golden_key=%d,active1=%d,charge1=%d,full_charge1=%d,active2=%d,charge2=%d,full_charge2=%d,items=%s,trinket1=%d,trinket2=%d,damage=%.2f,fire_rate=%.2f,shot_speed=%.2f,range=%.2f,luck=%.2f,speed=%.2f,card=%d,pill=%d,alive_enemies=%d,room_type=%d,first_visit=%d,stage=%d,time_counter=%d",
        player.Position.X, player.Position.Y, player.Velocity.X, player.Velocity.Y,
        player:GetHearts(), player:GetMaxHearts(), player:GetSoulHearts(), player:GetBlackHearts(),
        player:GetRottenHearts(), player:GetBoneHearts(), player:GetEternalHearts(),
        player:GetExtraLives(),
        player:GetNumCoins(), player:GetNumBombs(), player:GetNumKeys(),
        player:HasGoldenBomb() and 1 or 0, player:HasGoldenKey() and 1 or 0,
        active1, charge1, NeedsCharge1,
        active2, charge2, NeedsCharge2,
        collectedItemsStr, player:GetTrinket(0), player:GetTrinket(1),
        player.Damage, player.MaxFireDelay, player.ShotSpeed, player.TearRange, player.Luck, player.MoveSpeed,
        player:GetCard(0), player:GetPill(0),
        aliveEnemies, roomType, firstVisit, stage, timeCounter)

    WriteFile(paths.playerData, data)
end

-- ====================== FEATURE: FLOOR LAYOUT LOGGING ======================
local function ScanRooms()
    local paths = GetFilePaths()
    local level = game:GetLevel()
    local roomData = {}

    local rooms = level:GetRooms()  -- Get the list of generated rooms
    local currentRoomIndex = level:GetCurrentRoomDesc().SafeGridIndex  -- Get the current room index

    for i = 0, rooms.Size - 1 do
        local roomDesc = rooms:Get(i)
        local index = roomDesc.SafeGridIndex
        local listIndex = roomDesc.ListIndex
        local roomType = roomDesc.Data and roomDesc.Data.Type or 0
        local visited = roomDesc.VisitedCount > 0 and 1 or 0
        local isClear = roomDesc.Clear and 1 or 0
        local shape = roomDesc.Data and roomDesc.Data.Shape or 1  -- Default to normal room

        -- Identify all possible indices belonging to this room
        local indices = { index }
        if shape == 4 or shape == 5 then  -- Vertical (2-room)
            table.insert(indices, index + 13)
        elseif shape == 6 or shape == 7 then  -- Horizontal (2-room)
            table.insert(indices, index + 1)
        elseif shape >= 8 and shape <= 12 then  -- Large rooms (including L-shapes)
            local i1, i2, i3, i4 = index, index + 1, index + 13, index + 14

            -- Adjust indices for Shape 9 (missing top-left)
            if shape == 9 then
                i1, i2, i3, i4 = index - 1, index, index + 12, index + 13
            end

            indices = { i1, i2, i3, i4 }

            -- Replace the missing tile with "0,0,0,0,0,0"
            if shape == 9 then  -- Missing top-left
                indices[1] = 0
            elseif shape == 10 then  -- Missing top-right
                indices[2] = 0
            elseif shape == 11 then  -- Missing bottom-left
                indices[3] = 0
            elseif shape == 12 then  -- Missing bottom-right
                indices[4] = 0
            end
        end

        -- Determine if ANY part of this room is the current room
        local isCurrent = 0
        for _, idx in ipairs(indices) do
            if idx == currentRoomIndex then
                isCurrent = 1
                break
            end
        end

        -- Log all parts of the room with the same isCurrent value
        for _, idx in ipairs(indices) do
            if idx == 0 then
                table.insert(roomData, "0,0,0,0,0,0")  -- Fill missing room tiles with zeros
            else
                table.insert(roomData, string.format("%d,%d,%d,%d,%d,%d", idx, listIndex, roomType, visited, isClear, isCurrent))
            end
        end
    end
    WriteFile(paths.floorData, table.concat(roomData, "\n"))
end

-- ====================== FEATURE: ENTITY SCANNING ======================
local function ScanEntities()
    local paths = GetFilePaths()
    local room = game:GetRoom()
    local entities = {}

    for _, entity in pairs(Isaac.GetRoomEntities()) do
        local entityType = entity.Type
        local x, y = entity.Position.X, entity.Position.Y
        local vx, vy = entity.Velocity.X, entity.Velocity.Y
        local flags = entity:GetEntityFlags() or 0

        local isPlayer = entity:ToPlayer() ~= nil
        local isFamiliar = entity:ToFamiliar() ~= nil

        if not isPlayer and not isFamiliar then
            local npc = entity:ToNPC()
            if npc then
                local hp = npc.HitPoints
                local isInvincible = npc:IsInvincible() and 1 or 0
                local collisionDamage = npc.CollisionDamage
                local size = npc.Size
                table.insert(entities, string.format("1,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%d",
                    entityType, x, y, vx, vy, hp, isInvincible, collisionDamage, size, flags))

            elseif entity.Type == EntityType.ENTITY_EFFECT then
                local effect = entity:ToEffect()
                local variant = effect.Variant
                local scale = effect.Scale
                local timeout = effect.Timeout
                table.insert(entities, string.format("8,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,0,%d",
                    entityType, x, y, vx, vy, scale, timeout, variant, flags))

            elseif entity.Type == EntityType.ENTITY_BOMB then
                local bomb = entity:ToBomb()
                local explosionDamage = bomb.ExplosionDamage
                local radiusMultiplier = bomb.RadiusMultiplier
                table.insert(entities, string.format("2,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,0,0,%d",
                    entityType, x, y, vx, vy, explosionDamage, radiusMultiplier, flags))

            elseif entity.Type == EntityType.ENTITY_PICKUP then
                local pickup = entity:ToPickup()
                local coinValue = pickup:GetCoinValue()
                local isShopItem = pickup:IsShopItem() and 1 or 0
                local price = pickup.Price
                table.insert(entities, string.format("3,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f,0,%d",
                    entityType, x, y, vx, vy, coinValue, isShopItem, price, flags))

            elseif entity.Type == EntityType.ENTITY_PROJECTILE then
                local projectile = entity:ToProjectile()
                local height = projectile.Height
                local damage = projectile.CollisionDamage
                local scale = projectile.SpriteScale.X
                table.insert(entities, string.format("4,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,0,%d",
                    entityType, x, y, vx, vy, height, damage, scale, flags))

            elseif entity.Type == EntityType.ENTITY_TEAR then
                local tear = entity:ToTear()
                local baseDamage = tear.CollisionDamage
                local scale = tear.SpriteScale.X
                local height = tear.Height
                table.insert(entities, string.format("5,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,0,%d",
                    entityType, x, y, vx, vy, baseDamage, scale, height, flags))

            elseif entity.Type == EntityType.ENTITY_LASER then
                local laser = entity:ToLaser()
                local parent = laser.Parent
                local parentType = parent and parent.Type or -1
                local angle = laser.Angle
                local distance = laser.Distance
                local damage = laser.CollisionDamage
                local scale = laser.SpriteScale.X
                local timeout = laser.Timeout
                table.insert(entities, string.format("7,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d",
                    entityType, x, y, vx, vy, angle, distance, damage, scale, timeout, flags))

            elseif entity.Type == EntityType.ENTITY_SLOT then
                local variant = entity.Variant
                local hitPoints = entity.HitPoints
                local timeout = entity.Timeout or 0

                table.insert(entities, string.format("9,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,0,0,%d",
                    entityType, x, y, vx, vy, hitPoints, timeout, variant, flags))

            elseif entity.Type == EntityType.ENTITY_BEGGAR then
                local variant = entity.Variant
                local hitPoints = entity.HitPoints
                local timeout = entity.Timeout or 0

                table.insert(entities, string.format("10,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,0,0,%d",
                    entityType, x, y, vx, vy, hitPoints, timeout, variant, flags))
            end

        elseif entity:ToFamiliar() ~= nil or (entity.IsFriendly and entity:IsFriendly()) then
            local familiar = entity:ToFamiliar()
            if familiar then
                local player = familiar.Player
                local playerIndex = player and player.ControllerIndex or -1
                local fireCooldown = familiar.FireCooldown
                local canShoot = familiar.CanShoot and 1 or 0
                local spriteScale = familiar.SpriteScale.X
                local hp = familiar.HitPoints
                local collisionDamage = familiar.CollisionDamage
                local variant = familiar.Variant

                table.insert(entities, string.format("6,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%d",
                    entityType, x, y, vx, vy, hp, canShoot, collisionDamage, spriteScale, variant, flags))
            end
        end
    end
    WriteFile(paths.entityData, table.concat(entities, "|"))
end

-- ====================== CALLBACKS ======================
mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, function(_, entity, amount, damageFlags, damageSource) --track damage to enemies
    if entity:IsVulnerableEnemy() then
        local id = entity.InitSeed  -- Unique ID for the enemy
        local prevHP = enemyHPBefore[id] or entity.HitPoints

        -- Only log damage if HP actually decreases
        if entity.HitPoints < prevHP then
            totalDamageDealt = totalDamageDealt + (prevHP - entity.HitPoints)
            -- Do not call WriteDamageToFile() here; defer to MC_POST_UPDATE
        end

        -- Update stored HP
        enemyHPBefore[id] = entity.HitPoints
    end
end)

local function WriteDamageToFile()
    WriteFile(GetFilePaths().damageData, tostring(totalDamageDealt))
end

-- For game start/reset
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, isContinued)
    if not isContinued and hasReset then
        local player = Isaac.GetPlayer(0)
        local room = game:GetRoom()
        local roomCenter = room:GetCenterPos()
        local range = 100
        player.Position = Vector(roomCenter.X + math.random(-range, range), roomCenter.Y + math.random(-range, range))
        hasReset = false
    end

    ClearFiles()
    totalDamageDealt = 0
    WriteDamageToFile()
    enemyHPBefore = {}
end)

-- ====================== POST UPDATE ======================
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    if Input.IsButtonTriggered(Keyboard.KEY_F1, 0) then
        pendingNumber = math.max(0, pendingNumber - 1)
        pendingChanges = true
    end
    if Input.IsButtonTriggered(Keyboard.KEY_F2, 0) then
        pendingNumber = pendingNumber + 1
        pendingChanges = true
    end
    if Input.IsButtonTriggered(Keyboard.KEY_F3, 0) and pendingChanges then
        if pendingNumber > 0 then
            instanceNumber = pendingNumber
            ClearFiles()
            totalDamageDealt = 0  -- Reset damage on instance change
            enemyHPBefore = {}
        else
            instanceNumber = 0
        end
        pendingChanges = false
        lastInputData = ""
        newInputData = nil
    end

    if instanceNumber > 0 then
        local inputChanged = ReadInputsFromFile()

        if inputChanged then
            HandlePathfinding()
            RemoveEnemies()
            ScanEntities()
            ScanRooms()
            LogRoomTiles()
            WritePlayerData()
            WriteDamageToFile()  -- Write damage here, after other updates

            if newInputData then
                WriteFile(GetFilePaths().response, newInputData)
                lastInputData = newInputData
                newInputData = nil
            end
        end
    end
end)

mod:AddCallback(ModCallbacks.MC_POST_RENDER, function() --rendering stuff in screen
    local screenWidth = Isaac.GetScreenWidth()
    local screenHeight = Isaac.GetScreenHeight()
    local scaleX = screenWidth / 800
    local scaleY = screenHeight / 600
    local scale = math.min(scaleX, scaleY)
    local xOffset = 30 * scale
    local yOffset = screenHeight - 30
    local spacing = 30 * scale
    local startX = xOffset
    local maxY = 0
    for _, key in ipairs(keyActions) do
        if key.yPos > maxY then
            maxY = key.yPos
        end
    end
    for _, key in ipairs(keyActions) do
        local xPos = startX + key.xOffset * spacing
        local yPos = yOffset - ((maxY - key.yPos) * scale)
        sprite.Scale = Vector(scale, scale)
        if inputs[key.action] == 1 then
            sprite:SetFrame(key.anim, 1)
        else
            sprite:SetFrame(key.anim, 0)
        end
        sprite:Render(Vector(xPos, yPos), Vector(0, 0), Vector(0, 0))
    end
    local font = Font()
    font:Load("font/terminus.fnt")
    local x = 3
    local y = screenHeight - 10
    local displayText
    if instanceNumber == 0 then
        if pendingChanges then
            displayText = "Instance: Disabled - " .. pendingNumber .. " (F3 to apply)"
        else
            displayText = "Instance: Disabled (F1/F2 to change)"
        end
    else
        if pendingChanges then
            displayText = "Instance: " .. instanceNumber .. " - " .. pendingNumber .. " (F3 to apply)"
        else
            displayText = "Instance: " .. instanceNumber
        end
    end
    Isaac.RenderScaledText(displayText, x, y, 1, 1, 1, 1, 1, 1)
end)


-- ====================== MODIFIED INPUT HANDLING ======================
mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, _, buttonAction)
    -- Fall back to original input handling
    local inputValue = inputs[buttonAction]
    if inputValue == 0 then
        inputValue = nil
    end
    return inputValue
end)
