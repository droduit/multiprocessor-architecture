# Introduction to Multiprocessor Architecture (CS307)

Multiprocessors are a core component in all types of computing infrastructure, from phones to datacenters. This repo gather the assignment made for the course "Introduction to Multiprocessor Architecture" taught at EPFL. This  course introduce the essential technologies required to combine multiple processing elements into a single computer.


## Content of the assignments

### Assignment 1 (OpenMP) - [Monte-Carlo integration](https://en.wikipedia.org/wiki/Monte_Carlo_integration)
Implementation of a simple program for Monte-Carlo integration, parallelized with OpenMP framework. 

### Assignment 2 (OpenMP) - Heat transfer 
Simulation of a heat transfer operating on a variable sized square-shaped two-dimensional array, and usage of software optimization techniques to optimize the process.

### Assignment 3 (OpenMP) - [Thread-safe linked list](https://en.wikipedia.org/wiki/Non-blocking_linked_list) implementation using locks
Implementation of a thread-sage singly linked list using locks. Multiple threads can operate on the linked list in parallel without affecting functionality an correctness of the lists' operations.

The power of linked lists lies in the fact that they are allocated dynamically without any limitations at compile time. They grow and shrink in memory footprint as nodes are added ans deleted at runtime. It is also easy to add and delete nodes at any position by rearranging the link field among nodes. However, this flexibility comes at the expense of the extra storage needed for this link field.

### Assignment 4 (CUDA) - Heat transfer optimized on GPU
The problem is the same as that proposed in assignment 2. Optimized simulation of a heat transfer using the power of the GPU.

_____________________________

## CS307 Course content

- Forms of parallelism
- Parallel programming models
- Cache coherence
- Memory consistency
- Synchronization
- Interconnection networks
- Software efficiency & optimization
- GPU architecture & programming


