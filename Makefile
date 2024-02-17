main: main.o CTRNN.o TSearch.o Prey.o Predator.o random.o
	g++ -pthread -o main main.o CTRNN.o TSearch.o Prey.o Predator.o random.o
random.o: random.cpp random.h VectorMatrix.h
	g++ -pthread -c -O3 random.cpp
CTRNN.o: CTRNN.cpp random.h CTRNN.h
	g++ -pthread -c -O3 CTRNN.cpp
TSearch.o: TSearch.cpp TSearch.h
	g++ -pthread -c -O3 TSearch.cpp
Prey.o: Prey.cpp Prey.h Predator.h CTRNN.h random.h VectorMatrix.h
	g++ -pthread -c -O3 Prey.cpp
Predator.o: Predator.cpp Predator.h Prey.h CTRNN.h random.h VectorMatrix.h
	g++ -pthread -c -O3 Predator.cpp
main.o: main.cpp CTRNN.h Prey.h Predator.h TSearch.h
	g++ -pthread -c -O3 main.cpp
clean:
	rm *.o main