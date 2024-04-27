#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'add a timer to zip outputs'

git push origin master

echo '------- update complete --------'