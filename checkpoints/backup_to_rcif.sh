#!/bin/bash

FILES="*_cropped_[0-9]*"
DRY_RUN=false

for file in $FILES; do
  # Ignore if already a symlink
  if [[ -L $file ]]; then
    continue
  fi 

  if [ "$DRY_RUN" = true ]; then
    echo $file
    continue 
  fi

  mv $file /share/rcifdata/awilkins/checkpoints_backup/$file

  ln -s /share/rcifdata/awilkins/checkpoints_backup/$file $file

  echo "$file backup up"
done

