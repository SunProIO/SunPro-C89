require! {
  fs
  glob
  async
  'prelude-ls': {each, map, filter}
}

sources = glob.sync \src/**/*.re

used-equations = []

async.each source, (source, done) ->
  dest = source.replace /\.re$/ \.build-re

  fs.read-file source, (error, data) ->
    return console.error error if error

    text = data.toString!

    text .= replace /@<m>\{((\\}|[^}])*)(\})/g (string, equation) ->
      equation .= replace /\\}/g '}'

      return string
