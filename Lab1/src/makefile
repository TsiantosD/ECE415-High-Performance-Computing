# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

# Arguments
SIZE ?= 4096
IMAGE_NAME ?= timescapes

#Add any executable you want to be created here.
EXECUTABLES	= 1_sobel_orig 2_sobel_loop_interchange \
3_sobel_loop_unrolling 4_sobel_loop_fusion 5_sobel_function_inlining \
 6_sobel_common_subexpression_elimination 7_sobel_strength_reduction \
 7_sobel_strength_reduction_sqrt_lut 8_sobel_openmp

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

