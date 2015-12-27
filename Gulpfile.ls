require! {
  \gulp
  \gulp-less
}

gulp.task \assets ->
  gulp.src \assets/*.less base: \.
  .pipe gulp-less!
  .pipe gulp.dest \src

gulp.task \default, [\assets]
