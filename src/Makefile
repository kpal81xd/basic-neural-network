CC = gcc

FLAGS = -g -Werror -Wall -pedantic -std=c99 -lm

SRC = $(wildcard *.c)
OUT = $(patsubst %.c,%.o,$(SRC))

EXE = ann

run: $(EXE)
	./$<

val: $(EXE)
	valgrind --tool=memcheck --leak-check=full ./$<

$(EXE): $(OUT)
	$(CC) $^ -o $@ $(FLAGS)

%.o: %.c
	$(CC) $(FLAGS) -c $< -o $@

clean:
	rm -f $(wildcard *.o)
	rm -f $(EXE)

	
