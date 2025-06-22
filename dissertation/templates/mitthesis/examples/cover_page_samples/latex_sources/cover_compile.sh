#! /bin/bash

# first run

for i in *.tex; 
	do 
		if [ "$i" != "abstract.tex" ] ; then 
			pdflatex $i; 
		fi; 
	done

# second run

for i in *.tex; 
	do 
		if [ "$i" != "abstract.tex" ] ; then 
			pdflatex $i; 
		fi; 
	done

# clean up

mkdir pdffiles
mv *.pdf pdffiles
rm *.aux
rm *.gz
rm *.log




