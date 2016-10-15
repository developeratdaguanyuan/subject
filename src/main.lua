require 'DataLoader'

require 'cunn'
require 'cutorch'
torch.setdefaulttensortype('torch.CudaTensor')

local dataLoader = DataLoader("../data/question_subjectEmbed_train.txt")
local data, posMark, posEntity, negMark, negEntity = unpack(dataLoader:nextBatch())
--print(data)
--print(posMark)
--print(torch.cumsum(torch.pow(posEntity, 2), 2))
--print(negMark)
--print(torch.cumsum(torch.pow(negEntity, 2), 2))


for i = 1, 1000 do
  print(i)
  dataLoader:nextBatch()
end
