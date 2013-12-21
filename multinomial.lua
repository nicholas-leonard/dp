
local DEBUG = false

-- Normalize each row of a matrix to sum to one
local function sum_to_one(prob_dist)
    local sums = prob_dist:sum(2)
    local sums = torch.expand(sums, prob_dist:size())
    return prob_dist:cdiv(sums)
end

-- Make cumulative probability distribution for each row of a matrix.
-- If inplace is True, use the prob_dist to store the cumulative dist.
local function cumulative(prob_dist, inplace)
    local cum_dist = prob_dist
    if not inplace then
        cum_dist = prob_dist:clone()
    end
    for i=2,cum_dist:size(2) do
       cum_dist:select(2,i):add(cum_dist:select(2,i-1))
    end
    if DEBUG then
        print("cum_dist")
        print(cum_dist) 
    end
    
    return cum_dist
end
    
-- Sample from multinomial matrix with replacement
local function multinomial_with_replacement(prob_dist, num_samples)    
    --Make cumulative distribution out of multinomial (sum-to-one) dist:
    local cum_dist = cumulative(sum_to_one(prob_dist:clone()), true)
    
    --Sample from uniform:
    local u_samples = torch.rand(cum_dist:size(1), num_samples)
    if DEBUG then
        print("u_samples")
        print(u_samples) 
    end
    --Used to store multinomial samples:
    local m_samples = torch.IntTensor(cum_dist:size(1), num_samples)
    
    for i=1,num_samples do
        for j=1,cum_dist:size(2) do
            local a, b, indices
            local u_sample = u_samples:select(2,i)
            if j < cum_dist:size(2) then
                a = torch.lt(u_sample, cum_dist:select(2,j))
            end
            if j > 1 then
                b = torch.ge(u_sample, cum_dist:select(2,j-1))
            end
            if a and b then
                indices = torch.eq(a,b)
            elseif not b then
                indices = a
            else
                indices = b
            end
            m_samples:select(2, i)[indices] = j
        end
    end    
    return m_samples
end

-- Sample without replacement from multinomial probability distribution 
-- matrix, where each row is its own probability distribution. 
local function multinomial_without_replacement(prob_dist, num_samples)
    --Used to store multinomial samples:
    local m_samples = torch.IntTensor(prob_dist:size(1), num_samples)
        
    local function multinomial(prob_dist, m_samples, i)
        --Make cumulative distribution out of multinomial (sum-to-one) dist:
        local cum_dist = cumulative(sum_to_one(prob_dist:clone()), true)
        
        --Sample from uniform:
        local u_samples = torch.rand(cum_dist:size(1))
        
        if DEBUG then
            print("u_samples")
            print(u_samples) 
        end
        
        for j=1,cum_dist:size(2) do
            local a, b, indices
            if j < cum_dist:size(2) then
                a = torch.lt(u_samples, cum_dist:select(2,j))
            end
            if j > 1 then
                b = torch.ge(u_samples, cum_dist:select(2,j-1))
            end
            if a and b then
                indices = torch.eq(a,b)
            elseif not b then
                indices = a
            else
                indices = b
            end
            m_samples:select(2, i)[indices] = j
            --Zero the probability of the sampled nominal
            --i.e. the "without replacement":
            if DEBUG then
                print("indices")
                print(indices) 
            end
            prob_dist:select(2, j)[indices] = 0
        end
        if i > 1 then 
            if DEBUG then
                print("prob_dist", i)
                print(prob_dist) 
            end
            return multinomial(prob_dist, m_samples, i-1)
        end
        return m_samples
    end
    return multinomial(prob_dist:clone(), m_samples, num_samples)
end

------------------------------------------------------------------------
--[[ dp.multinomial ]]--
-- function
-- Sample from a multinomial distribution with or without replacement
-- Returns an index for each sample. 
-- When input is a matrix, output is a matrix where each row contains 
-- num_samples indices
-- When input is a vector, output is a vector where each variable 
-- contains num_samples indices
------------------------------------------------------------------------

function dp.multinomial(prob_dist, num_samples, without_replacement)
   num_samples = num_samples or 1
   local new_size, old_size
   if prob_dist:dim() == 1 then
      old_size = prob_dist:size()
      new_size = _.concat({1}, old_size:totable())
      prob_dist:resize(unpack(new_size))
   end
   assert(prob_dist:dim() == 2)
   local res
   if without_replacement then
      res = multinomial_without_replacement(prob_dist, num_samples)
   end
   res = multinomial_with_replacement(prob_dist, num_samples)
   if new_size then
      prob_dist:resize(old_size)
      res:resize(old_size)
   end
   return res
end

------------------------------------------------------------------------
-- TEST --
------------------------------------------------------------------------

function test_multinomial()
    p_dist = torch.rand(4,5)
    t = torch.Timer()
    print"with replacement"
    print(multinomial_with_replacement(p_dist, 3))
    print(t:time().real)

    t = torch.Timer()
    print"without replacement"
    print(multinomial_without_replacement(p_dist, 3))
    print(t:time().real)
end
