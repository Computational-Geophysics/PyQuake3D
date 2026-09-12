# Compiler
CXX = g++

# Source file path
SRC = TDstressFS_C.cpp

# Output shared library file name
TARGET = TDstressFS_C.so

# Compilation options
CXXFLAGS = -O3 -fPIC -std=c++11
LDFLAGS = -shared

# Default target
all: $(TARGET)

# Compile and generate .so file
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean up intermediate files
clean:
	rm -f $(TARGET)



# ������
#CXX = g++

# Դ�ļ�·��
#SRC = TDstressFS_C.cpp

# ��� DLL ��
#TARGET = TDstressFS_C.dll

# ����ѡ��
#CXXFLAGS = -O3 -std=c++11 -shared
# �������ű� + ���ɵ���⣨��ѡ��
#LDFLAGS = -Wl,--out-implib,libTDstressFS_C.a

# Ĭ��Ŀ��
#all: $(TARGET)

# �������� DLL �ļ�
#$(TARGET): $(SRC)
#	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# ����
#clean:
#	del /Q *.dll *.a *.o *.exp