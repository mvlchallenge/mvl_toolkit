#!/bin/bash

ZIP_DIR=/media/NFS/kike/360_Challenge/mvl_toolkit/mvl_challenge/assets/tmp/zip
ID_DIR=1IrH4sj1V8F9KMlANYDDU0MGR0ftoSrKc

for filename in $ZIP_DIR/*.zip; do
    echo $filename
    gdrive upload -p $ID_DIR $filename
done
