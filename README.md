# PyVideoStab
Python wrapper of OpenCV's videostab

Based on [pyboostcvconverter](https://github.com/Algomorph/pyboostcvconverter)

Compatibility
-----------------
I recommend to use python 3.5

Disclaimer
-----------------
Certain things in the code might be excessive/unneeded, so if you know something is not needed, please make a pull request with an update. Also, conversion errors aren't handled politically correct (i.e. just generates an empty matrix), please let me know if that bothers you or you'd like to fix that.
The code has been tested for memory leaks. If you still find any errors, let me know by positing an issue! 

Compiling & Trying Out Sample Code
----------------------
1. Install CMake and/or CMake-gui (http://www.cmake.org/download/, ```sudo apt-get install cmake cmake-gui``` on Ubuntu/Debian)
2. Automated mode:
- `chmod +x buid.sh`
- `./build.sh`
3. Manual mode:
- Run CMake and/or CMake-gui with the git repository as the source and a build folder of your choice (in-source builds supported.) Choose desired generator, configure, and generate. Remember to set PYTHON_DESIRED_VERSION to 2.X for python 2 and 3.X for python 3.
- Build (run ```make``` on *nix systems with gcc/eclipse CDT generator from within the build folder)
- On *nix systems, ```make install``` run with root privileges will install the compiled library file. Alternatively, you can manually copy it to the pythonXX/dist-packages directory (replace XX with desired python version).
- Run `test_videostab.py`
