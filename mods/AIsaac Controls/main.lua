local mod = RegisterMod("AIsaac's Controls", 1)

local inputs = {}  -- Store inputs
local sprite = Sprite()
local hasReset = false  -- Track if the game was reset

-- Generate sprite filenames dynamically
local spriteFiles = {}
for i = 1, 51 do  -- Change 51 to the number of keys
    table.insert(spriteFiles, "keys" .. i .. ".anm2")
end

local currentSpriteFile = spriteFiles[math.random(#spriteFiles)]
sprite:Load(currentSpriteFile, true)

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

local startX = 30  -- Starting X position
local spacing = 30 -- Distance between each key icon

local responseFile = "F:/IsaacResponse1.txt"  -- File to write responses

local function writeResponse(response)
    local file = io.open(responseFile, "w")  -- Open file in write mode
    if not file then
        print("[Lua] Warning: Failed to open response file")
        return
    end
    file:write(response)  -- Write the response
    file:close()  -- Close the file
end

local function resetGame()
    currentSpriteFile = spriteFiles[math.random(#spriteFiles)]
    sprite:Load(currentSpriteFile, true)
    hasReset = true  -- Mark that we need to reposition Isaac
    Isaac.ExecuteCommand("restart")  -- Force restart
end

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, isContinued)
    if not isContinued and hasReset then
        local game = Game()
        local level = game:GetLevel()
        local player = Isaac.GetPlayer(0)
        local room = game:GetRoom()
        local roomCenter = room:GetCenterPos()

        -- Define a range around the center (adjust if needed)
        local range = 100
        local randomX = roomCenter.X + math.random(-range, range)
        local randomY = roomCenter.Y + math.random(-range, range)
        player.Position = Vector(randomX, randomY)

        hasReset = false  -- Ensure it happens only once
    end
end)

local function readInputsFromFile()
    local file = io.open("F:/IsaacInputs1.txt", "r")  -- Open file in read mode
    if not file then
        print("[Lua] Warning: Failed to open input file")
        return
    end

    local data = file:read("*a")  -- Read entire file
    file:close()  -- Close file after reading

    if data then
        -- Check if we need to reset
        if data:find("reset") then
            resetGame()
            return
        end

        -- Process input data
        inputs = {}  -- Clear previous inputs
        for action, value in string.gmatch(data, "(%d+) (%d+)") do
            action = tonumber(action)
            value = tonumber(value)
            inputs[action] = value
        end

        -- Send acknowledgment to Python
        writeResponse(data)
    end
end

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
    readInputsFromFile()  -- Read inputs every frame
end)

mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, _, buttonAction)
    local inputValue = inputs[buttonAction]

    if inputValue == 0 then
        inputValue = nil
    end

    return inputValue
end)

mod:AddCallback(ModCallbacks.MC_POST_RENDER, function()
    for _, key in ipairs(keyActions) do
        local xPos = startX + key.xOffset * spacing

        if inputs[key.action] == 1 then
            sprite:SetFrame(key.anim, 1) -- Fully visible frame
        else
            sprite:SetFrame(key.anim, 0) -- Transparent frame
        end

        sprite:Render(Vector(xPos, key.yPos), Vector(0, 0), Vector(0, 0))
    end
end)
