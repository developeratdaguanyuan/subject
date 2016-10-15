require 'utils'


local DataLoader = torch.class('DataLoader')

function DataLoader:__init(dataPath, batchSize)
  self.batchSize = batchSize == nil and 10 or batchSize
  self.maxIndex, self.word_ids_list, self.pos_mark_entity_list, self.neg_mark_entity_list
    = unpack(self:createData(dataPath))
  self.dataSize = #self.word_ids_list
  self.numBatch = math.floor(self.dataSize/self.batchSize)
  self.headList, self.cntList = unpack(self:groupData())
end

function DataLoader:nextBatch()
  if self.currentIndex == nil or self.currentIndex + self.batchSize - 1 > self.dataSize then
    self.currentIndex = 1
    self.indices = self:rerank(self.headList, self.cntList, self.dataSize)
  end

  -- select data indices
  local dataIndex = self.indices:narrow(1, self.currentIndex, self.batchSize)
  self.currentIndex = self.currentIndex + self.batchSize

  -- construct current batch data
  -- word_ids_list
  local maxStep = self.word_ids_list[dataIndex[1]]:size(1)
  local currentDataBatch = torch.LongTensor(self.batchSize, maxStep):fill(1)
  for i = 1, self.batchSize, 1 do
    currentDataBatch[{{i}, {maxStep - self.word_ids_list[dataIndex[i]]:size(1) + 1, maxStep}}]
      = self.word_ids_list[dataIndex[i]]
  end
  -- pos_mark_entity_list
  local currentPosMarkBatch = torch.LongTensor(self.batchSize)
  local currentPosEntityBatch = torch.DoubleTensor(self.batchSize, 256)
  for i = 1, self.batchSize, 1 do
    currentPosMarkBatch[{i}] = self.pos_mark_entity_list[dataIndex[i]][1]
    currentPosEntityBatch[{{i}}] = self.pos_mark_entity_list[dataIndex[i]][2]
  end
  -- neg_mark_entity_list
  local currentNegMarkBatch = torch.LongTensor(self.batchSize)
  local currentNegEntityBatch = torch.DoubleTensor(self.batchSize, 256)
  for i = 1, self.batchSize, 1 do
    if #self.neg_mark_entity_list[dataIndex[i]] > 0 then
      local random_id = getRandomInteger(1, #self.neg_mark_entity_list[dataIndex[i]])
      currentNegMarkBatch[{i}] = self.neg_mark_entity_list[dataIndex[i]][random_id][1]
      currentNegEntityBatch[{{i}}] = self.neg_mark_entity_list[dataIndex[i]][random_id][2]
    elseif #self.neg_mark_entity_list[dataIndex[i]] == 0 then
      currentNegMarkBatch[{i}] = 0
      currentNegEntityBatch[{{i}}]:uniform(-1, 1)-- = torch.rand(256)
      local norm_2 = torch.norm(currentNegEntityBatch[{{i}}])
      currentNegEntityBatch[{{i}}]:div(norm_2)
    end
  end

  return {cudacheck(currentDataBatch),
          cudacheck(currentPosMarkBatch),
          cudacheck(currentPosEntityBatch),
          cudacheck(currentNegMarkBatch),
          cudacheck(currentNegEntityBatch)}
end

function DataLoader:rerank(headList, cntList, dataSize)
  local list = torch.LongTensor(dataSize):zero()
  for i = 1, #headList, 1 do
    list[{{headList[i], headList[i] + cntList[i] - 1}}] = torch.LongTensor.torch.randperm(cntList[i]):add(headList[i] - 1)
  end
  return list
end

function DataLoader:createData(path)
  local maxIndex = 0
  local word_ids_list = {}
  local pos_mark_entity_list = {}
  local neg_mark_entity_list = {}

  local file = io.open(path, 'r')
  local buffer = {}
  for line in file:lines() do
    if line ~= "" then
      table.insert(buffer, line)
    else
      if #buffer ~= 0 then
        local word_ids = torch.LongTensor(split(buffer[1], " "))
        word_ids_list[#word_ids_list + 1] = word_ids
        maxIndex = math.max(maxIndex, torch.max(word_ids))

        local tokens = split(buffer[2], ";")
        pos_mark_entity_list[#pos_mark_entity_list + 1]
          = {tokens[1], torch.DoubleTensor(split(tokens[2], ","))}

        local neg_mark_entity = {}
        for i = 3, #buffer do
          local mark, entity = unpack(split(buffer[i], ";"))
          neg_mark_entity[#neg_mark_entity + 1] = {mark, torch.DoubleTensor(split(entity, ","))}
        end
        neg_mark_entity_list[#neg_mark_entity_list + 1] = neg_mark_entity

        buffer = {}
      end
    end
  end

  if #buffer ~= 0 then
    local word_ids = torch.LongTensor(split(buffer[1], " "))
    word_ids_list[#word_ids_list + 1] = word_ids
    maxIndex = math.max(maxIndex, torch.max(word_ids))

    local tokens = split(buffer[2], ";")
    pos_mark_entity_list[#pos_mark_entity_list + 1]
      = {tokens[1], torch.DoubleTensor(split(tokens[2], ","))}

    local neg_mark_entity = {}
    for i = 3, #buffer do
      local mark, entity = unpack(split(buffer[i], ";"))
      neg_mark_entity[#neg_mark_entity + 1] = {mark, torch.DoubleTensor(split(entity, ","))}
    end
    neg_mark_entity_list[#neg_mark_entity_list + 1] = neg_mark_entity

    buffer = {}
  end

  return {maxIndex, word_ids_list, pos_mark_entity_list, neg_mark_entity_list}
end

function DataLoader:groupData()
  local headList = {}
  local cntList = {}

  local head, cnt = 1, 1
  for i = 2, #self.word_ids_list, 1 do
    if self.word_ids_list[i]:size(1) == self.word_ids_list[head]:size(1) then
      cnt = cnt + 1
    else
      headList[#headList + 1] = head
      cntList[#cntList + 1] = cnt
      head = i
      cnt = 1
    end
  end
  headList[#headList + 1] = head
  cntList[#cntList + 1] = cnt

  return {headList, cntList}
end

