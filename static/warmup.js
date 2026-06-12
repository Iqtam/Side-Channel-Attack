/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;

function readNlines(n) {
  /*
   * Implement this function to read n cache lines.
   * 1. Allocate a buffer of size n * LINESIZE.
   * 2. Read each cache line (read the buffer in steps of LINESIZE) 10 times.
   * 3. Collect total time taken in an array using `performance.now()`.
   * 4. Return the median of the time taken in milliseconds.
   */

  // Allocate a buffer of size n * LINESIZE
  const buffer = new Uint8Array(n * LINESIZE);

  // Initialize buffer with some values
  for (let i = 0; i < buffer.length; i++) {
    buffer[i] = 1;
  }

  // Array to store timing measurements
  const timings = [];

  // Perform 10 iterations of reading the buffer
  for (let iter = 0; iter < 10; iter++) {
    let sum = 0; // Dummy variable to ensure reads are not optimized away

    const startTime = performance.now();

    // Read each cache line (steps of LINESIZE)
    for (let i = 0; i < buffer.length; i += LINESIZE) {
      sum += buffer[i];
    }

    const endTime = performance.now();

    // Store the time taken for this iteration
    timings.push(endTime - startTime);

    // Use sum to prevent compiler optimization
    if (sum === 0) {
      console.log("This should never happen");
    }
  }

  // Calculate the median time
  timings.sort((a, b) => a - b);
  const median = timings[Math.floor(timings.length / 2)];

  return median;
}

self.addEventListener("message", function (e) {
  if (e.data === "start") {
    const results = {};

    /* Call the readNlines function for n = 1, 10, ... 10,000,000 and store the result */
    try {
      // Start with n=1 and keep multiplying by 10 until we reach 10,000,000
      for (let n = 1; n <= 10000000; n *= 10) {
        console.log(`Measuring for n = ${n}`);
        results[n] = readNlines(n);

        // If something goes wrong with measurement, break the loop
        if (isNaN(results[n]) || results[n] === undefined) {
          console.error(`Error measuring for n = ${n}`);
          break;
        }
      }
    } catch (error) {
      console.error("Error in worker:", error);
    }

    self.postMessage(results);
  }
});
