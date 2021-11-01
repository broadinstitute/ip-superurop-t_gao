#!/bin/bash
#
# chtc-test.sh
#
echo "Beginning CHTC test Job $1 running on `whoami`@`hostname`"
#
python3 chtc-test.py
#
# keep this job running for a few minutes so you'll see it in the queue:
# sleep 60
