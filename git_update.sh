#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'paper accepted by IOTJ'

git push origin master

echo '------- update complete --------'