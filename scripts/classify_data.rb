### Convert file of DISCRETE features to file of numeric features via classification
require 'csv'

file_name = "foo.csv"
data_types = [:float, :discrete, :discrete, :discrete, :discrete, :float, :discrete, :discrete, :discrete, :float, :discrete, :float, :float, :float, :float, :discrete, :discrete]

data = File.read(file_name)
data = data.split("\n")
data = data.map { |d| d.split(",").map { |e| e.gsub("\"", "") } }
data.shift # remove header

data_possible_vals = {}
data_new = []
data.each do |dp|
  new_dp = []

  dp.each_with_index do |e, i|
    if data_types[i] == :float
      new_dp.push(e)
    elsif data_types[i] == :discrete
      if !data_possible_vals.has_key?(i)
        data_possible_vals[i] = []
      end
      idx = data_possible_vals[i].find_index(e)
      if !idx
        idx = data_possible_vals[i].size
	data_possible_vals[i].push(e)
      end
      new_dp.push(idx) 
    else
      raise "invalid data type: " + data_types[i].to_s
    end
  end

  data_new.push(new_dp)
end

CSV.open(file_name + ".new", "w") do |csv|
  data_new.each do |dn|
    csv << dn
  end
end
