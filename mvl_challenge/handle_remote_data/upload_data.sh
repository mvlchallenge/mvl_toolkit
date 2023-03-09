#!/bin/bash

# ZIP_DIR=/media/NFS/kike/360_Challenge/mvl_toolkit/mvl_challenge/assets/tmp/zip
# ID_DIR=1IrH4sj1V8F9KMlANYDDU0MGR0ftoSrKc

ZIP_DIR=/media/NFS/kike/360_Challenge/mvl_toolkit/mvl_challenge/assets/tmp/zip_files__labels
ID_DIR=1ORpSP60h34TIOZlYstkuKhQBDaloPR5M

for filename in $ZIP_DIR/*.zip; do
    echo $filename
    gdrive upload -p $ID_DIR $filename
done

gdrive list --query " '$ID_DIR' in parents" > $ZIP_DIR/ids.csv