#!/bin/bash

for file in $(find /projets/twitter/alltweets/2017/01/08/* -type f ); 
	do cat "$file" >> output.csv; 
done
