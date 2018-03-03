### Convert CSV with numeric labels (i.e. int from 0-9) to list of binary labels (10 binary values, one for each possible int)

possible_vals = (0..9).to_a

require 'csv'
old = CSV.read("/Users/danielruskin/src/simple-neural-network/mnist_data/training_labels.csv")
new = []

old.each do |row|
  val = Float(row[0])

  if !possible_vals.include?(val)
    raise NotImplementedError
  end

  new_e = []
  (0..(possible_vals.size - 1)).each do |possible_val|
    if possible_val == val
      new_e.push(1)
    else
      new_e.push(0)
    end
  end

  new.push(new_e)
end

CSV.open("/Users/danielruskin/src/simple-neural-network/mnist_data/training_labels_new.csv", "w") do |csv|
  new.each { |n| csv << n }
end
