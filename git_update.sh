#!/bin/bash

echo '------- update git and remote --------'

git add .

git commit . -m 'correct pp_weight'

git push origin master

echo '------- update complete --------'