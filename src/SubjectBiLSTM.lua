require 'rnn'
require 'DataLoader'
require 'utils'
require 'nngraph'
require 'logroll'

local SubjectBiLSTM = torch.class("SubjectBiLSTM")

function mergeModule()
  local question = nn.Identity()()
  local entity = nn.Identity()()
  local relation = nn.Identity()()

  local sum = nn.CAddTable()({nn.DotProduct()({question, entity}), nn.Linear(1, 1, false)(relation)})

  return nn.gModule({question, entity, relation}, {sum})
end

function SubjectBiLSTM:__init(opt)
  self.log = logroll.file_logger('logs/bilstm_info.log')

  if opt.useGPU == 1 then
    require 'cunn'
    require 'cutorch'
    torch.setdefaulttensortype('torch.CudaTensor')
  end

  self.trainDataPath = opt.trainDataFile
  self.validDataPath = opt.validDataFile
  self.modelDirectory = opt.modelDirectory

  self.learningRate = opt.learningRate
  self.costMargin = opt.costMargin
  self.vocabularySize = opt.vocabularySize
  self.vocabularyDim = opt.vocabularyDim
  self.hiddenSize = self.vocabularyDim
  self.entityDim = opt.entityDim
  self.batchSize = opt.batchSize

  self.maxEpochs = opt.maxEpochs
  self.printEpoch = opt.printEpoch

  self.dataLoader = DataLoader(self.trainDataPath, self.batchSize, self.vocabularySize)
  self.validDataLoader = DataLoader(self.validDataPath, 1, self.vocabularySize)

  self.wordEmbedding = nn.LookupTable(self.vocabularySize, self.vocabularyDim)
  self.forwardEncoder = nn.Sequential()
            :add(nn.Sequencer(self.wordEmbedding))
            :add(nn.Sequencer(nn.LSTM(self.hiddenSize, self.hiddenSize)))
            :add(nn.SelectTable(-1))

  self.backwardEncoder = nn.Sequential()
            :add(nn.ReverseTable())
            :add(nn.Sequencer(self.wordEmbedding))
            :add(nn.Sequencer(nn.LSTM(self.hiddenSize, self.hiddenSize)))
            :add(nn.SelectTable(-1))

  self.concat = nn.ConcatTable()
            :add(self.forwardEncoder)
            :add(self.backwardEncoder)

  self.encoder = nn.Sequential()
            :add(self.concat)
            :add(nn.JoinTable(1, 1))
            :add(nn.Linear(self.hiddenSize * 2, self.entityDim))

  self.encoderModel = cudacheck(self.encoder)
  self.positiveEncoder, self.negativeEncoder
    = unpack(cloneModulesWithSharedParameters(self.encoderModel, 2))

  self.mergeModel = cudacheck(mergeModule())
  self.positiveMerge, self.negativeMerge
    = unpack(cloneModulesWithSharedParameters(self.mergeModel, 2))

  self.criterion = cudacheck(nn.MarginRankingCriterion(self.costMargin))


end

function SubjectBiLSTM:train()
  local epochLoss, accumLoss = 0, 0
  local maxIter = self.dataLoader.numBatch * self.maxEpochs

  for i = 1, maxIter do
    xlua.progress(i, maxIter)

    self.encoderModel:zeroGradParameters()
    self.mergeModel:zeroGradParameters()

    local question, posMark, posEntity, negMark, negEntity = unpack(self.dataLoader:nextBatch())

    local inputSeq = question:t():split(1, 1)
    for j = 1, #inputSeq, 1 do
      inputSeq[j] = torch.squeeze(inputSeq[j])
    end
    local pos_question = self.positiveEncoder:forward(inputSeq)
    local neg_question = self.negativeEncoder:forward(inputSeq)

    local pos_score = self.positiveMerge:forward({pos_question, posEntity, posMark})
    local neg_score = self.negativeMerge:forward({neg_question, negEntity, negMark})

    local loss =
      self.criterion:forward({pos_score, neg_score}, torch.Tensor(self.batchSize):fill(1))
    epochLoss = epochLoss + loss
    accumLoss = accumLoss + loss

    local d_loss =
      self.criterion:backward({pos_score, neg_score}, torch.Tensor(self.batchSize):fill(1))

    local d_pos_triple = self.positiveMerge:backward({pos_question, posEntity, posMark}, d_loss[1])
    local d_neg_triple = self.negativeMerge:backward({neg_question, negEntity, negMark}, d_loss[2])

    self.positiveEncoder:backward(inputSeq, d_pos_triple[1])
    self.negativeEncoder:backward(inputSeq, d_neg_triple[1])

    self.encoderModel:updateParameters(self.learningRate)
    self.mergeModel:updateParameters(self.learningRate)

    if i % self.printEpoch == 0 then
      self.log.info(string.format("[Iter %d]: %f", i, accumLoss / self.printEpoch))
      accumLoss = 0
    end

    -- evaluate and save model
    if i % self.dataLoader.numBatch == 0 then
      local epoch = i / self.dataLoader.numBatch
      local correct_rate = self:evaluate()
      self.log.info(
        string.format("[Epoch %d]: [training error %f]: [evaluating error %f]",
          epoch, epochLoss / self.dataLoader.numBatch, correct_rate))
      torch.save(self.modelDirectory.."/BiLSTM_"..epoch, {self.encoderModel, self.mergeModel})
      epochLoss = 0
    end
  end

end

function SubjectBiLSTM:evaluate()
  local count = 0
  for i = 1, self.validDataLoader.dataSize, 1 do
    local question, posMark, posEntity, negMark, negEntity = unpack(self.validDataLoader:nextBatch())

    self.encoderModel:zeroGradParameters()
    self.mergeModel:zeroGradParameters()

    local inputSeq = question:t():split(1, 1)
    for j = 1, #inputSeq, 1 do
      inputSeq[j] = torch.LongTensor(1):fill(torch.squeeze(inputSeq[j]))
    end

    local embed_question = self.encoderModel:forward(inputSeq)
    local pos_score = self.positiveMerge:forward({embed_question, posEntity, posMark})
    local neg_score = self.negativeMerge:forward({embed_question, negEntity, negMark})
    count = count + (pos_score[1] > neg_score[1] and 1 or 0)
  end
  return count / self.validDataLoader.dataSize
end

