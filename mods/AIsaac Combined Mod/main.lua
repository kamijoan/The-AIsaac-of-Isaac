local mod = RegisterMod("AIsaac_Combined_Mod", 1)

-- ====================== VARIABLES ======================
local game = Game()
local inputs = {}  -- Store inputs
local sprite = Sprite()
local hasReset = false  -- Track if the game was reset
local previousEntityData = ""
local previousRoomData = ""
local totalDamageDealt = 0
local enemyHPBefore = {}

-- ====================== FILE PATHS ======================
local playerDataPath = "F:/IsaacData1.txt"
local roomTileDataPath = "F:/IsaacTileData1.txt"
local damageDataPath = "F:/IsaacEnemyDamage1.txt"
local floorDataPath = "F:/IsaacFloorData1.txt"
local entityDataFile = "F:/IsaacEntityData1.txt"
local inputFile = "F:/IsaacInputs1.txt"
local responseFile = "F:/IsaacResponse1.txt"

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
local removeEnemies = false -- Set to true to enable enemy removal

-- ====================== HELPER FUNCTIONS ======================
local function ClearFiles()
    -- Clear/reset all data files
    local files = {
        playerDataPath,
        roomTileDataPath,
        damageDataPath,
        floorDataPath,
        entityDataFile,
        responseFile
    }

    for _, filepath in ipairs(files) do
        local file = io.open(filepath, "w")
        if file then
            file:write("")
            file:close()
        end
    end

    -- Reset variables
    previousEntityData = ""
    previousRoomData = ""
    totalDamageDealt = 0
    enemyHPBefore = {}
end

local function WriteFile(filepath, data)
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

local function ReadInputsFromFile()
    local file = io.open(inputFile, "r")
    if file then
        local data = file:read("*a")
        file:close()
        if data then
            if data:find("reset") then
                resetGame()
                return
            end
            inputs = {}
            for action, value in string.gmatch(data, "(%d+) (%d+)") do
                inputs[tonumber(action)] = tonumber(value)
            end
            WriteFile(responseFile, data)
        end
    end
end

-- ====================== FEATURE: DAMAGE TRACKING ======================
local function WriteDamageToFile()
    WriteFile(damageDataPath, tostring(totalDamageDealt))
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

    WriteFile(roomTileDataPath, table.concat(tileData, "\n"))
end

-- ====================== FEATURE: PLAYER DATA LOGGING ======================
local function WritePlayerData()
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

    WriteFile(playerDataPath, data)
end

-- ====================== FEATURE: FLOOR LAYOUT LOGGING ======================
local function ScanRooms()
    local level = game:GetLevel()
    local roomData = {}

    local rooms = level:GetRooms()  -- Get the list of generated rooms
    local currentRoomIndex = level:GetCurrentRoomDesc().SafeGridIndex  -- Get the current room index

    for i = 0, rooms.Size - 1 do
        local roomDesc = rooms:Get(i)
        local index = roomDesc.SafeGridIndex
        local listIndex = roomDesc.ListIndex
        local roomType = roomDesc.Data and roomDesc.Data.Type or 0
        local seen = roomDesc.VisitedCount > 0 and 1 or 0
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
                table.insert(roomData, string.format("%d,%d,%d,%d,%d,%d", idx, listIndex, roomType, seen, isClear, isCurrent))
            end
        end
    end

    -- Convert table to a single string for comparison
    local currentRoomData = table.concat(roomData, "\n")

    -- Write only if data has changed
    if currentRoomData ~= previousRoomData then
        WriteFile(floorDataPath, currentRoomData)
        previousRoomData = currentRoomData
    end
end

-- ====================== FEATURE: ENTITY SCANNING ======================
local function ScanEntities()
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

    local currentEntityData = table.concat(entities, "|")

    if currentEntityData ~= previousEntityData then
        WriteFile(entityDataFile, currentEntityData)
        previousEntityData = currentEntityData
    end
end

-- ====================== CALLBACKS ======================

-- For entity damage tracking
mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, function(_, entity, amount, damageFlags, damageSource)
    if entity:IsVulnerableEnemy() then
        local id = entity.InitSeed  -- Unique ID for the enemy
        local prevHP = enemyHPBefore[id] or entity.HitPoints

        -- Only log damage if HP actually decreases
        if entity.HitPoints < prevHP then
            totalDamageDealt = totalDamageDealt + (prevHP - entity.HitPoints)
            WriteDamageToFile()
        end

        -- Update stored HP
        enemyHPBefore[id] = entity.HitPoints
    end
end)

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

-- For regular updates (most functions)
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    ReadInputsFromFile()
    RemoveEnemies()
    ScanEntities()
    ScanRooms()
end)

-- For rendering input feedback - Updated with dynamic scaling
mod:AddCallback(ModCallbacks.MC_POST_RENDER, function()
    -- Get current screen size
    local screenWidth = Isaac.GetScreenWidth()
    local screenHeight = Isaac.GetScreenHeight()

    -- Calculate scaling factor
    local scaleX = screenWidth / 800  -- Scale relative to a reference width
    local scaleY = screenHeight / 600  -- Scale relative to a reference height
    local scale = math.min(scaleX, scaleY)  -- Maintain aspect ratio

    -- Offsets for positioning
    local xOffset = 30 * scale  -- Adjust left/right position
    local yOffset = screenHeight - 30  -- Position near the bottom

    -- Adjust spacing dynamically
    local spacing = 30 * scale  -- Scale spacing between keys
    local startX = xOffset  -- Position from the left side

    -- Find max yPos to flip rows correctly
    local maxY = 0
    for _, key in ipairs(keyActions) do
        if key.yPos > maxY then
            maxY = key.yPos
        end
    end

    -- Input display
    for _, key in ipairs(keyActions) do
        local xPos = startX + key.xOffset * spacing
        local yPos = yOffset - ((maxY - key.yPos) * scale)  -- Flip Y ordering

        sprite.Scale = Vector(scale, scale)  -- Apply sprite scaling

        if inputs[key.action] == 1 then
            sprite:SetFrame(key.anim, 1)  -- Fully visible frame
        else
            sprite:SetFrame(key.anim, 0)  -- Transparent frame
        end

        sprite:Render(Vector(xPos, yPos), Vector(0, 0), Vector(0, 0))
    end

    -- Log room tiles and player data in post render for consistent timing
    LogRoomTiles()
    WritePlayerData()
end)

-- For input handling
mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, _, buttonAction)
    local inputValue = inputs[buttonAction]
    if inputValue == 0 then
        inputValue = nil
    end
    return inputValue
end)
