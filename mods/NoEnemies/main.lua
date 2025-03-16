local mod = RegisterMod("NoEnemies", 1)
local game = Game()

function mod:PreventEnemies()
    -- Iterate through all entities in the room
    for _, entity in ipairs(Isaac.GetRoomEntities()) do
        -- Check if the entity is an enemy
        if entity:IsEnemy() then
            -- Remove the enemy
            entity:Remove()
        end
    end
end

-- Attach the function to the game's update callback
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, mod.PreventEnemies)
