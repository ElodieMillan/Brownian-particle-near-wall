CC = gcc -fPIC -lgsl -lgslcblas -lm

default: librandom.a

librandom.a: random.o
	ar rcs $@ $^

examples.o: random.c random.h
	$(CC) -c $<

clean:
	rm *.o *.a