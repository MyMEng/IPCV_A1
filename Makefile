all:
	g++ counter.cpp -o counter.a $(O_LIBS)

run:
	./counter.a
