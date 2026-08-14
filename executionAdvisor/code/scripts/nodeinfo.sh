#!/bin/bash
echo "$(hostname -s) | $(cat /etc/redhat-release) | glibc $(ldd --version|head -1|awk '{print $NF}') | $(nproc) cpu | $(lscpu|awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}')"
