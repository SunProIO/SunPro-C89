require! {
  \fs
  \mkdirp
  \gulp
  \querystring
  \gulp-less
  \gulp-jade
  \gulp-rename
  \gulp-postcss
  \gulp-browserify
  \js-yaml
  \autoprefixer
}

var catalog-data

gulp.task \css ->
  gulp.src \assets/style.less base: \.
  .pipe gulp-less!
  .pipe gulp-postcss [autoprefixer browsers: ['> 1%']]
  .pipe gulp.dest \src

gulp.task \css-index ->
  gulp.src \assets/index.less base: \.
  .pipe gulp-less!
  .pipe gulp-postcss [autoprefixer browsers: ['> 1%']]
  .pipe gulp.dest \.

gulp.task \js ->
  gulp.src \assets/*.js base: \.
  .pipe gulp-browserify!
  .pipe gulp.dest \src

gulp.task \html <[catalog]> ->
  gulp.src \assets/*.jade
  .pipe gulp-jade locals:
    catalog: catalog-data
    qs: querystring
  .pipe gulp.dest \.

gulp.task \catalog (done) ->
  fs.read-file \assets/catalog.yml \utf8 (error, text) ->
    return done error if error

    try
      data = js-yaml.safe-load text
    catch
      return done e

    catalog-data := data

    json = JSON.stringify data, null '  '

    mkdirp \src/assets (err) ->
      return done error if error

      fs.write-file \src/assets/catalog.json json, \utf8 (error, text) ->
        if error
          return done error
        else
          return done!

gulp.task \dist-pub <[assets]> ->
  gulp.src <[src/**/*.@(html|js|css|svg|png|jpeg|jpg|json|pdf) !src/**/*.web.* !**/math/*]> base: \src
  .pipe gulp.dest \dist/pub

gulp.task \dist-webimage <[dist-pub]> ->
  gulp.src <[src/**/*.web.@(png|jpeg|jpg)]> base: \src
  .pipe gulp-rename -> it.basename .= replace /\.web$/ ''
  .pipe gulp.dest \dist/pub

gulp.task \dist-root <[assets]> ->
  gulp.src <[index.html assets/*.@(css|eot|svg|ttg|woff) images/*]> base: \.
  .pipe gulp.dest \dist

gulp.task \assets <[js css css-index html catalog]>

gulp.task \dist <[dist-pub dist-webimage dist-root]>

gulp.task \default [\assets]
