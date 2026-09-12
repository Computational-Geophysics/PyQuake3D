rm -rf build/ dist/ *.egg-info src/*.o
find . -name "*.so" -delete

# 2.  Conda  Intel 
unset CFLAGS
unset CXXFLAGS
unset LDFLAGS

# 3.  Clang ， arm64 
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export ARCHFLAGS="-arch arm64"
export CMAKE_OSX_ARCHITECTURES="arm64"

# 4. ，
python -m pip install -e . --no-cache-dir --force-reinstall
