#!/usr/bin/env bash

URL='https://www.googleapis.com/drive/v3/files/'
URL_END='?alt=media'
FOLDER_PATH=$2
ID=$1
TOKEN=ya29.a0AVvZVsrrazc3-2UxSCZGZzMRp9r-At9nEVDO17t_GrjJt6bpx2TIkhDb8cEDErIZqPCO2TfJCvEnP-vXEyLjK77ZlbCuqT_ADWepSdBVH6MB5rpLVetfKRzIvJT7h1Wg5QKdmjDomgK8nf3jjox6O8FVkLtzaCgYKATYSARASFQGbdwaI0_9SwZXaCoshZEUyg_-F9w0163

URL_SCENE=$URL$ID$URL_END
OUTPUT=$FOLDER_PATH 
CODE=$(curl -H "Authorization: Bearer $TOKEN" $URL_SCENE -o $OUTPUT)
