require 'nn'
require 'rnn'
require 'DataLoader'
require 'SubjectLSTM'

local cmd = torch.CmdLine()
cmd:option('-algorithm', 'lstm', 'Algorithm Options')
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

local subjectLSTM = SubjectLSTM(opt)
subjectLSTM:train()

