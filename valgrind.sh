valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         --log-file=./logs/valgrind-out.txt \
         ./build/main

cat ./logs/valgrind-out.txt
