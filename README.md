# A GPU-accelerated N-Queens solver

This implements the linear time algorithm presented in the following paper:  
[https://arxiv.org/pdf/1805.07329.pdf].

Since the placement of the queens are no longer interdependent, I was able to parallelize the algorithm using the CUDA API (just for fun). The result is an extremely fast solver, capable of solving for N = 268,435,456 in about 173 milliseconds. There is some overhead due to the data transfer between the host computer and the GPU, so the script takes a total of 1.7 seconds to run:

![Screen Capture](screencap.JPG) 

These numbers were achieved with a GTX 1060 GPU, which has 1,280 CUDA cores and 6GB of memory. The seemingly arbitrary N value mentioned above is the maximum the GPU can work with before it runs out of memory.  

The code also has a built-in verification script to ensure that the solution is valid (i.e. no two queens can attack each other).  

Right now, the only task pending is a way to push the solution array from the GPU back to my local machine in chunks. The built-in cudaMemcpy function seems to copy the solution array as one very long, linear byte sequence. As N becomes very large, the solution reaches several hundreds of megabytes to a few gigabytes, before we start hitting integer limits and cannot go any further.  

 
