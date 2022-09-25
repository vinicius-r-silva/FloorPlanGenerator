valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=./logs/valgrind-out.txt ./build/FloorPlanGenerator/FloorPlanGenerator
cat ./logs/valgrind-out.txt