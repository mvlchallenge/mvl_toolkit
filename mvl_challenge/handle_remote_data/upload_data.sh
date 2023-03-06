#!/bin/bash

ZIP_DIR=$1
ID_DIR=1j40k7MBg-yJVwIQX5DVL4qEBFD25baBo

for filename in $ZIP_DIR/*.zip; do
    echo $filename
    gdrive upload -p $ID_DIR $filename
done
