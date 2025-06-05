Part 1: CUDA Warm-Up 1: SAXPY (5 pts)
Question 1: The program takes about 188-210ms to complete on my machine. 
Question 2: The entire program takes 188-210ms to execute but the kernel only takes about 5ms to execute. This means that we spend most of the time transferring data between host and device rather than performing the saxpy operation. No, observed bandwith of about 5.3 GB/s does not roughly match the Interconnect Bandwidth of the 16-lane PCIe 3.0 (32 GB/s)

Part 2: CUDA Warm-Up 2: Parallel Prefix-Sum (10 pts)
See code

Part 3: A Simple Circle Renderer (85 pts)
Answer these questions here:
Include both partners names and SUNet id's at the top of your write-up.
Replicate the score table generated for your solution and specify which machine you ran your code on.
Describe how you decomposed the problem and how you assigned work to CUDA thread blocks and threads (and maybe even warps).
Describe where synchronization occurs in your solution.
What, if any, steps did you take to reduce communication requirements (e.g., synchronization or main memory bandwidth requirements)?
Briefly describe how you arrived at your final solution. What other approaches did you try along the way. What was wrong with them?
