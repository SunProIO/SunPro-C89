require! {
  \gulp
  \gulp-less
  \gulp-browserify
}

gulp.task \css ->
  gulp.src \assets/*.less base: \.
  .pipe gulp-less!
  .pipe gulp.dest \src

gulp.task \js ->
  gulp.src \assets/*.js base: \.
  .pipe gulp-browserify!
  .pipe gulp.dest \src

gulp.task \assets <[js css]>

gulp.task \default [\assets]
