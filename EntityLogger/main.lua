local mod = RegisterMod("EntityLogger", 1)

local previousEntityData = ""

-- Function to clear entity data
local function ClearData()
    previousEntityData = ""
    currentEntityData = ""
    local file = io.open("F:/IsaacEntityData.txt", "w")
    file:write("")  -- Clear the file
    file:close()
end

local function ScanEntities()
    local game = Game()
    local room = game:GetRoom()
    local entities = {}

    for _, entity in pairs(Isaac.GetRoomEntities()) do
        local entityType = entity.Type
        local x, y = entity.Position.X, entity.Position.Y
        local vx, vy = entity.Velocity.X, entity.Velocity.Y
        local flags = entity:GetEntityFlags() or 0  -- Get flags, default to 0 if not available

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
                local variant = entity.Variant  -- Variant determines the type of slot machine
                local hitPoints = entity.HitPoints
                local timeout = entity.Timeout or 0  -- Some entities don't have Timeout

                table.insert(entities, string.format("9,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,0,0,%d",
                    entityType, x, y, vx, vy, hitPoints, timeout, variant, flags))

            elseif entity.Type == EntityType.ENTITY_BEGGAR then
                local variant = entity.Variant  -- Variant determines beggar type
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
        local file = io.open("F:/IsaacEntityData.txt", "w")
        file:write(currentEntityData)
        file:close()
        previousEntityData = currentEntityData
    end
end

-- Callback for when a new game/run starts
local function OnGameStart()
    ClearData()  -- Clear data on reset
end

-- Register callbacks
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, ScanEntities)
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, OnGameStart)
