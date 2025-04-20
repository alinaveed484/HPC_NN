CC = gcc
CFLAGS = -Wall -O2

EXE = nn.exe
SRC = nn.c

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)
	
gprof: $(SRC)
	$(CC) $(CFLAGS) -pg -o $(EXE) $(SRC) -lm
	./$(EXE)
	gprof $(EXE) gmon.out > gprof_report.txt

clean:
	rm -f $(EXE)
	rm -f gmon.out