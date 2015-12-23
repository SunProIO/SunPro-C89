#! /bin/sh
exec ruby -S -x "$0" "$@"
#! ruby

# Usage: ./tex2png.rb [equation str] [outfile]

require "tex2png"
require "fileutils"

equation = ARGV[0]
outfile = ARGV[1]

converter = Tex2png::Converter.new(equation)

FileUtils.mkdir_p File.dirname outfile
FileUtils.cp converter.png.path, outfile
