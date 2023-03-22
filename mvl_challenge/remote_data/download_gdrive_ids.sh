#!/usr/bin/env bash

URL='https://www.googleapis.com/drive/v3/files/'
URL_END='?alt=media'
FOLDER_PATH=$2
FILE_CSV=$1
TOKEN=ya29.a0AVvZVsrrazc3-2UxSCZGZzMRp9r-At9nEVDO17t_GrjJt6bpx2TIkhDb8cEDErIZqPCO2TfJCvEnP-vXEyLjK77ZlbCuqT_ADWepSdBVH6MB5rpLVetfKRzIvJT7h1Wg5QKdmjDomgK8nf3jjox6O8FVkLtzaCgYKATYSARASFQGbdwaI0_9SwZXaCoshZEUyg_-F9w0163

while read line; do
  # echo "$line";
  firstItem="$(echo $line | cut -d' ' -f1)"
  secondItem="$(echo $line | cut -d' ' -f2)"
  echo $firstItem $secondItem 
  URL_SCENE=$URL$firstItem$URL_END
  OUTPUT=$FOLDER_PATH$secondItem 
  CODE=$(curl -H "Authorization: Bearer $TOKEN" $URL_SCENE -o $OUTPUT)
  echo $CODE
done < $FILE_CSV