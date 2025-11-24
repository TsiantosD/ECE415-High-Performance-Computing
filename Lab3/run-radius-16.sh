# For question 5b, run with radius 16 and image sizes up to 16k
./run.sh --step step5 --target 16 --size 1024 --repeat 10
./run.sh --step step5 --target 16 --size 2048 --repeat 10
./run.sh --step step5 --target 16 --size 4096 --repeat 10
./run.sh --step step5 --target 16 --size 8192 --repeat 10
./run.sh --step step5 --target 16 --size 16384 --repeat 10

# For question 6, same run but with doubles
./run.sh --step step5 --target 16 --size 1024 --repeat 10 --use-doubles
./run.sh --step step5 --target 16 --size 2048 --repeat 10 --use-doubles
./run.sh --step step5 --target 16 --size 4096 --repeat 10 --use-doubles
./run.sh --step step5 --target 16 --size 8192 --repeat 10 --use-doubles
./run.sh --step step5 --target 16 --size 16384 --repeat 10 --use-doubles
