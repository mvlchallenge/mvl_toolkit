#!/bin/bash

usage() { echo "$0 USAGE:" && grep " .)\ #" $0; exit 0;}
[ $# -eq 0 ] && usage
while getopts ":hd:o:" arg; do
  case $arg in
    d) # Zip directory 
      ZIP_DIR=${OPTARG}
      ;;
    o) # Output directory
      OUTPUT_DIR=${OPTARG}
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

if [ -z "$OUTPUT_DIR" ]
then
      usage
fi

for filename in "$ZIP_DIR"/*.zip; do
    echo "$filename"
    unzip "$filename" -d "$OUTPUT_DIR"
done
