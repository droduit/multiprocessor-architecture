- 1
-- Part 1 (pi.c)
--- How the problem was solved in a parallel fashion ?
To parallelize this problem, I first defined a variable chunk = samples / num_threads which indicates how much samples will be processed by each thread. Then, I just loop num_threads times (1x for each threads). In the loop, I call a new function inside_circle, whose I give the number of samples to process. The code of this function thus remains serial each thread will execute this code that simply generates random x and y coordinates, and increase the "temporary circle area" if the point is located inside it. I use the reduction on the += operator with the variable circleArea and define that the variables num_threads and chunk are shared between all the threads. 

--- Perfomances


-- Part 2 (integral.c)
--- How the problem was solved in a parallel fashion ?
The way I parallelized this problem is the same as for the first part. I execute a loop that calls the function getVal for each threads, passing it the number of samples to process. The code of this function remains serial and simply generate randoms x coordinates into the given interval [a,b], then sum their corresponding y coordinates. I use the reduction again on the += operator to "merge" the final result of every threads. Finally, the result is given by the formula proposed on  https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/monte-carlo-methods-in-practice/monte-carlo-integration.

--- Performances


-2

