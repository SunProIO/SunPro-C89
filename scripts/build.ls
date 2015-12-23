require! {
  fs
  glob
  async
  crypto
  'prelude-ls': {each, map, filter}
  child_process: {spawn}
}

process.setMaxListeners 0

used-equations = Object.create null

source = process.argv[2]
dest = process.argv[3]

fs.read-file source, (error, data) ->
  return console.error error if error

  text = data.toString!

  text .= replace /@<m>\{((\\}|[^}])*)\}/g (string, equation) ->
    equation .= replace /\\}/g '}'
    hash = crypto.create-hash \md5 .update equation .digest \hex

    used-equations[hash] = equation

    return "@<icon>{math-#{hash}}"

  text .= replace /\/\/texequation{([\s\S]+?)\/\/}/g (string, equation) ->
    equation .= replace /\\}/g '}'
    hash = crypto.create-hash \md5 .update equation .digest \hex

    used-equations[hash] = equation

    return "//indepimage[math-#{hash}]"

  fs.write-file dest, text

  for hash, equation of used-equations
    tex2png = spawn \ruby [\scripts/tex2png.rb equation, "src/images/math-#{hash}.png"]
    tex2png.stdout.pipe process.stdout
    tex2png.stderr.pipe process.stderr
