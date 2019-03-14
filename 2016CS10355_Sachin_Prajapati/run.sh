export PYTHONPATH="/home/sachin/libsvm-3.23/python:${PYTHONPATH}"
#!/bin/bash
a=1
b=2
zero=0
# c=3
# d=4
aa=a
bb=b
cc=c
dd=d
ee=e
ff=f
gg=g


#Question 1(Naive Bayes)
if [ $1 == $a ]; then
	if [ $4 == $aa ]; then 
    	echo "Running for Train V/s Test"
    	python3 bayes_a.py $2 $3
    	echo "Running for Train V/s Train"    	
    	python3 bayes_a.py $2 $2 
    elif [ $4 == $bb ]; then 
        python3 bayes_b.py $2 $3 
    elif [ $4 == $cc ]; then 
        python3 bayes_c.py $2 $3     
    elif [ $4 == $dd ]; then 
        python3 bayes_d.py $2 $3     
    elif [ $4 == $ee ]; then 
        python3 bayes_e.py $2 $3     
    elif [ $4 == $ff ]; then 
        python3 bayes_f.py $2 $3     
    elif [ $4 == $gg ]; then 
        python3 bayes_f.py $2 $3     
    fi 
    
#Question 2(SVM)
elif [ $1 == $b ]; then 
	#Binary
    if [ $4 == $zero ]; then 
    	if [ $5 == $aa ]; then 
        	python3 svm.py $2 $3 $5 
	    elif [ $5 == $bb ]; then 
	        python3 svm.py $2 $3 $5 
	    elif [ $5 == $cc ]; then 
	        python3 libsvm.py $2 $3 $5     
	    fi
    #Multi
    elif [ $4 == $a ]; then 
		if [ $5 == $aa ]; then 
        	python3 svmmulti.py $2 $3 $aa 
	    elif [ $5 == $bb ]; then 
	        python3 libsvmmulti.py $2 $3
	    elif [ $5 == $cc ]; then 
	        python3 svmmulti.py $2 $3 $cc     
	    elif [ $5 == $dd ]; then 
	        python3 validation.py $2 $3     
	    fi 
    fi
    # python3 q2.py $2 $3 $4 
fi


