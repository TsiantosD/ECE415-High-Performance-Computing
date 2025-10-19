# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

# Arguments
SIZE ?= 4096
IMAGE_NAME ?= timescapes

#Add any executable you want to be created here.
EXECUTABLES	= 01_sobel_orig 02_sobel_loop_interchange \
03_sobel_pow_reduction 04_sobel_index_reduction 05_sobel_ternary_op \
06_sobel_loop_fusion 07_sobel_loop_unroll_conv 08_sobel_f7_r4_5_6 \
09_sobel_f7_r4_5 10_sobel_f7_r4_6 11_sobel_f7_r6 12_sobel_f8_inline_conv \
13_sobel_subexpr_main_loop 14_sobel_reduce_variables 15_sobel_f13_openmp_outer

#This is the compiler to use
CC = icx

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -Wall -O0 -qopenmp -g -DSIZE=$(SIZE) \
					  -DINPUT_FILE=\"input/$(SIZE)-$(IMAGE_NAME).grey\" \
					  -DOUTPUT_FILE=\"output/$(IMAGE_NAME).grey\" \
					  -DGOLDEN_FILE=\"golden/$(IMAGE_NAME).grey\"

#These are the flags passed to the linker. Nothing in our case
LDFLAGS = -lm


# make all will create all executables
all: $(EXECUTABLES)

# This is the rule to create any executable from the corresponding .c 
# file with the same name.
%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# make clean will remove all executables, jpg files and the 
# output of previous executions.
clean:
	rm -f $(EXECUTABLES) *.jpg output_*.grey

# make image will create the IMAGE_NAME.jpg from the IMAGE_NAME.grey. Remember to pass the SIZE and IMAGE_NAME arguments.
image: output/$(IMAGE_NAME).grey
	convert -depth 8 -size $(SIZE)x$(SIZE) GRAY:output/$(IMAGE_NAME).grey output/$(IMAGE_NAME).jpg

