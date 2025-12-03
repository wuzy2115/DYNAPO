#!/bin/bash
FILE="alltracker_reference.tar.gz"
echo "downloading ${FILE} from dropbox"
wget --max-redirect=20 -O ${FILE} https://www.dropbox.com/scl/fi/ng66ceortfy07bgie3r54/alltracker_reference.tar.gz?rlkey=o781im2v0sl7035hy8fcuv1d5&st=u5mcttcx&dl=1
