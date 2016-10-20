require 'nn'
require 'rnn'
require 'DataLoader'
require 'SubjectLSTM'
require 'SubjectBiLSTM'

local cmd = torch.CmdLine()
cmd:option('-algorithm', 'bilstm', 'Algorithm Options')
cmd:option('-trainDataFile', '../data/question_subjectEmbed_train.txt', 'training data file')
cmd:option('-validDataFile', '../data/question_subjectEmbed_valid.txt', 'validation data file')
cmd:option('-wordEmbeddingFile', '../data/embedding.txt')
cmd:option('-modelDirectory', "../model")
cmd:option('-useEmbed', 1, 'whether to use word embedding')
cmd:option('-useGPU', 1, 'whether to use GPU')
cmd:option('-learningRate', 0.1, 'learning rate')
cmd:option('-costMargin', 1, 'margin of hinge loss')
cmd:option('-vocabularySize', 313939, 'vocabulary size')
cmd:option('-vocabularyDim', 300, 'word embedding size')
cmd:option('-entityDim', 256, 'entity embedding size')
cmd:option('-batchSize', 10, 'number of data in a batch')
cmd:option('-maxEpochs', 100, 'number of full passes through training data')
cmd:option('-printEpoch', 10, 'print training loss every printEpoch iterations')

local opt = cmd:parse(arg)

if opt.algorithm == 'lstm' then
  local subjectLSTM = SubjectLSTM(opt)
  subjectLSTM:train()
end
if opt.algorithm == 'bilstm' then
  local subjectBiLSTM = SubjectBiLSTM(opt)
  subjectBiLSTM:train()
end
if opt.algorithm == '2bilstm' then

end
