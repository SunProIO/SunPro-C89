#! /bin/sh
exec ruby -S -x "$0" "$@"
#! ruby

# Usage: ./tex2png.rb [equation str] [outfile]

require "tex2png"

equation = ARGV[0]
outfile = ARGV[1]

converter = Tex2png::Converter.new(equation)

p converter.png.path
