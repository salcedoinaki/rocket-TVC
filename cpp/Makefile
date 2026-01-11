# Rocket TVC Control - C++ Build Instructions
# For Windows with g++ (MinGW) or MSVC

# === Using g++ (MinGW) ===
# Compile and run tests:
# g++ -std=c++17 -O2 -Wall -Wextra -o test_control.exe rocket_control.cpp test_control.cpp -lm
# ./test_control.exe

# === Using MSVC (Visual Studio Developer Command Prompt) ===
# cl /EHsc /O2 /W4 /std:c++17 rocket_control.cpp test_control.cpp /Fe:test_control.exe
# test_control.exe

# === CMake Build (recommended for larger projects) ===
# See CMakeLists.txt

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS = -lm

TARGET = test_control.exe
SRCS = rocket_control.cpp test_control.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp rocket_control.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	del /Q *.o *.exe 2>nul || rm -f *.o *.exe

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
