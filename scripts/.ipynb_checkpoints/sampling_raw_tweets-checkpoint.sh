#!/bin/bash

for file in $(find /projets/twitter/alltweets/2017/01/08/* -type f ); 
	do cat "$file" >> output.csv; 
done

for file in $(find /projets/twitter/alltweets/2017/01/09/* -type f ); 
	do cat "$file" >> output.csv; 
done

for file in $(find /projets/twitter/alltweets/2017/01/10/* -type f ); 
	do cat "$file" >> output.csv; 
done

for file in $(find /projets/twitter/alltweets/2017/01/11/* -type f );
	do cat "$file" >> output.csv; 
done

for file in $(find /projets/twitter/alltweets/2017/01/12/* -type f );
	do cat "$file" >> output.csv; 
done

for file in $(find /projets/twitter/alltweets/2017/01/13/* -type f );
	do cat "$file" >> output.csv; 
done

for file in $(find /projets/twitter/alltweets/2017/01/14/* -type f ); 
	do cat "$file" >> output.csv; 
done
