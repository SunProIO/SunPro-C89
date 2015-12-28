require! {
  \fs
  \gulp
  \gulp-less
  \gulp-browserify
  \js-yaml
}

gulp.task \css ->
  gulp.src \assets/*.less base: \.
  .pipe gulp-less!
  .pipe gulp.dest \src

gulp.task \js ->
  gulp.src \assets/*.js base: \.
  .pipe gulp-browserify!
  .pipe gulp.dest \src

gulp.task \catalog (done) ->
  fs.read-file \assets/catalog.yml \utf8 (error, text) ->
    return done error if error

    try
      data = js-yaml.safe-load text
    catch
      return done e

    json = JSON.stringify data, null '  '

    fs.write-file \src/assets/catalog.json json, \utf8 (error, text) ->
      if error
        return done error
      else
        return done!

gulp.task \assets <[js css catalog]>

gulp.task \default [\assets]
