# Compiler from NVIDIA HPC SDK

# first run: source /opt/scripts/nvidia.sh

CC = nvc
CFLAGS = -acc -gpu=cc86 -O2 -Minfo=accel -lm

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