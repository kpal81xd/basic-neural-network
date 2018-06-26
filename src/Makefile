CC = gcc

FLAGS = -Werror -Wall -pedantic -std=c99 -lm

SRC = $(wildcard *.c)
OUT = $(patsubst %.c,%.o,$(SRC))

EXE = ann

all: $(EXE)
	./$<

$(EXE): $(OUT)
	$(CC) $^ -o $@ $(FLAGS)

%.o: %.c
	$(CC) $(FLAGS) -c $< -o $@

clean:
	rm -f $(wildcard *.o)
	rm -f $(EXE)

	
