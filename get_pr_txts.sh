#!/usr/bin/env bash

OUTPUT_DIR=pullreq-txts
mkdir -p $OUTPUT_DIR

cat pr-data.csv |
while read line; do
  owner=`echo $line|cut -f2 -d','|tr -d '"'|cut -f1 -d'/'`;
  repo=`echo $line|cut -f2 -d','|tr -d '"'|cut -f2 -d'/'`;
  num=`echo $line|cut -f4 -d','`;
  fname="$owner@$repo@$num.txt"
 
  echo "Processing $fname"

  title="print(db.pull_requests.find({owner:'$owner', repo:'$repo', number: $num})[0].title);"
  mongo github --quiet --eval "$title" > $OUTPUT_DIR/$fname
  echo >> $OUTPUT_DIR/$fname

  body="print(db.pull_requests.find({owner:'$owner', repo:'$repo', number: $num})[0].body);"
  mongo github --quiet --eval "$body" >> $OUTPUT_DIR/$fname

done

