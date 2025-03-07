local mod = RegisterMod("RoomTileLogger", 1)

local lastRoomIndex = 0  -- Store the last room index
local lastTiles = {}       -- Store previous tile states

-- GridEntityTypes to ignore
local ignoredTypes = {
    [GridEntityType.GRID_DECORATION] = true,
    -- Add more as needed
}

local function LogRoomTiles()
    local room = Game():GetRoom()
    local level = Game():GetLevel()
    local gridSize = room:GetGridSize()
    local roomIndex = level:GetCurrentRoomIndex()
    local roomX = room:GetGridWidth()
    local roomY = room:GetGridHeight()

    -- If the room changed, reset tile memory
    if roomIndex ~= lastRoomIndex then
        lastTiles = {}
        lastRoomIndex = roomIndex
    end

    local hasChanges = false
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

        -- Check if the tile changed
        if lastTiles[i] ~= tileString then
            lastTiles[i] = tileString
            hasChanges = true
        end

        table.insert(tileData, tileString)
    end

    -- Only write if there's a change
    if hasChanges then
        local file = io.open("F:/IsaacTileData.txt", "w")
        if file then
            file:write(table.concat(tileData, "\n"))
            file:close()
        end
    end
end

mod:AddCallback(ModCallbacks.MC_POST_RENDER, LogRoomTiles)
