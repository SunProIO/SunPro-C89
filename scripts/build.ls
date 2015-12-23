require! {
  fs
  glob
  async
  crypto
  'prelude-ls': {each, map, filter, keys}
  child_process: {spawn}
}

used-equations = Object.create null

source = process.argv[2]
dest = process.argv[3]

fs.read-file source, (error, data) ->
  return console.error error if error

  text = data.toString!

  text .= replace /@<m>\{((\\}|[^}])*)\}/g (string, equation) ->
    equation .= replace /\\}/g '}'
    equation .= trim!
    hash = crypto.create-hash \md5 .update equation .digest \hex

    used-equations[hash] = equation

    return "@<icon>{math-#{hash}}"

  text .= replace /\/\/texequation{([\s\S]+?)\/\/}/g (string, equation) ->
    equation .= replace /\\}/g '}'
    equation .= trim!
    hash = crypto.create-hash \md5 .update equation .digest \hex

    used-equations[hash] = equation

    return "//indepimage[math-#{hash}]"

  fs.write-file dest, text

  async.each-limit (used-equations |> keys), 5, (hash, done) ->
    equation = used-equations[hash]
    imgfile = "images/math-#{hash}.png"

    fs.access imgfile, fs.F_OK, (not-exists) ->
      if not-exists
        tex2png = spawn \ruby [\scripts/tex2png.rb equation, imgfile]
        tex2png.stdout.pipe process.stdout
        tex2png.stderr.pipe process.stderr
        tex2png.on \close (code) -> done!
      else
        done!
