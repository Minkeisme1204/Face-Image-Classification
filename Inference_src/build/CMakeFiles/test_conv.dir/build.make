# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src/build

# Include any dependencies generated for this target.
include CMakeFiles/test_conv.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_conv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_conv.dir/flags.make

CMakeFiles/test_conv.dir/test/conv_test.c.o: CMakeFiles/test_conv.dir/flags.make
CMakeFiles/test_conv.dir/test/conv_test.c.o: ../test/conv_test.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/test_conv.dir/test/conv_test.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/test_conv.dir/test/conv_test.c.o   -c /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src/test/conv_test.c

CMakeFiles/test_conv.dir/test/conv_test.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/test_conv.dir/test/conv_test.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src/test/conv_test.c > CMakeFiles/test_conv.dir/test/conv_test.c.i

CMakeFiles/test_conv.dir/test/conv_test.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/test_conv.dir/test/conv_test.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src/test/conv_test.c -o CMakeFiles/test_conv.dir/test/conv_test.c.s

# Object files for target test_conv
test_conv_OBJECTS = \
"CMakeFiles/test_conv.dir/test/conv_test.c.o"

# External object files for target test_conv
test_conv_EXTERNAL_OBJECTS =

test_conv: CMakeFiles/test_conv.dir/test/conv_test.c.o
test_conv: CMakeFiles/test_conv.dir/build.make
test_conv: libconvolution.a
test_conv: libactivation.a
test_conv: libnumpy.a
test_conv: CMakeFiles/test_conv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable test_conv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_conv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_conv.dir/build: test_conv

.PHONY : CMakeFiles/test_conv.dir/build

CMakeFiles/test_conv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_conv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_conv.dir/clean

CMakeFiles/test_conv.dir/depend:
	cd /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src/build /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src/build /home/minkescanor/Desktop/WORKPLACE/Hust/AI/Object_Detection/Inference_src/build/CMakeFiles/test_conv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_conv.dir/depend

