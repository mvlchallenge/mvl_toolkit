#!/usr/bin/env bash
usage() { echo "$0 USAGE:" && grep " .)\ #" $0; exit 0;}
[ $# -eq 0 ] && usage
while getopts ":hd:i:" arg; do
  case $arg in
    d) # Zip directory 
      ZIP_DIR=${OPTARG}
      ;;
    i) # GoogleDrive ID remote directory
      ID_DIR=${OPTARG}
      ;;
    h | *) # Display help.
      usage
      exit 0
      ;;
  esac
done

if [ -z "$ZIP_DIR" ]
then
      usage
fi

if [ -z "$ID_DIR" ]
then
      usage
fi

for filename in "$ZIP_DIR"/*.zip; do
    echo "$filename"
    gdrive upload -p "$ID_DIR" "$filename"
done

gdrive list --query " '$ID_DIR' in parents" > "$ZIP_DIR"/ids_"$ID_DIR".csv
gdrive upload -p "$ID_DIR" "$ZIP_DIR"/ids_"$ID_DIR".csv