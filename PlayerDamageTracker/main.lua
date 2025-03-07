local mod = RegisterMod("PlayerDamageTracker", 1)
local totalDamageDealt = 0
local fileName = "F:/IsaacEnemyDamage.txt"

local function WriteDamageToFile()
    local file = io.open(fileName, "w")
    if file then
        file:write(totalDamageDealt)
        file:close()
    end
end

-- Store enemy HP before taking damage
local enemyHPBefore = {}

local function OnEntityTakeDamage(_, entity, amount, damageFlags, damageSource, damageCountdownFrames)
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
end

-- Reset damage tracking at the start of a new game
local function OnGameStart(_, continued)
    totalDamageDealt = 0
    enemyHPBefore = {}  -- Clear stored HP values
    WriteDamageToFile()
end

-- Register callbacks
mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, OnEntityTakeDamage)
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, OnGameStart)
