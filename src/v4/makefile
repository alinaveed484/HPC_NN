CC = nvcc
# Architecture-specific flags for NVIDIA GPUs
# Adjust the architecture flag according to your GPU
# For example, for NVIDIA Turing architecture, use -arch=sm_75
CFLAGS = -O2 -arch=sm_86 -lcublas

EXE = nn.exe
SRC = nn.cu

all: $(EXE) run

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $(EXE) $(SRC) -lm

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)
