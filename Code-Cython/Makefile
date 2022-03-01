
default: langevin

langevin: setup.py *.pyx
	python3 setup.py build_ext -i clean && rm -f pyexamples.c 

clean:
	rm *.so
