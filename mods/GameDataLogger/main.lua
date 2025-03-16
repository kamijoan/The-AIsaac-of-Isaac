-- IsaacNetFileWriter: A Binding of Isaac mod that writes player data to a file
local mod = RegisterMod("IsaacNetFileWriter", 1)
local filePath = "F:/IsaacData1.txt"  -- Change the path if needed

local function ClearData()
    data = ""
    local file = io.open("F:/IsaacData1.txt", "w")
    file:write("")  -- Clear the file
    file:close()
end

-- Function to write player data to a file
function mod:WritePlayerData()
    local player = Isaac.GetPlayer(0)
    local game = Game()
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

    -- **Write to a file**
    local file = io.open(filePath, "w")  -- Overwrites file each frame
    if file then
        file:write(data)
        file:close()
    end
end

-- Callback for when a new game/run starts
local function OnGameStart()
    ClearData()  -- Clear data on reset
end

-- Callback to update the file every frame
mod:AddCallback(ModCallbacks.MC_POST_RENDER, function() mod:WritePlayerData() end)
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, OnGameStart)
