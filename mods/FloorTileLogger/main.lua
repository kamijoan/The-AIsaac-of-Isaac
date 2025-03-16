local mod = RegisterMod("FloorTileLogger", 1)
local json = require("json")

local previousRoomData = ""

local function ClearData()
    currentRoomData = ""
    previousRoomData = ""
    local file = io.open("F:/IsaacFloorData1.txt", "w")
    file:write("")  -- Clear the file
    file:close()
end

function ScanRooms()
    local level = Game():GetLevel()
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
        local file = io.open("F:/IsaacFloorData1.txt", "w")
        file:write(currentRoomData)
        file:close()
        previousRoomData = currentRoomData
    end
end

-- Callback for when a new game/run starts
local function OnGameStart()
    ClearData()  -- Clear data on reset
end

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, ScanRooms)
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, OnGameStart)
