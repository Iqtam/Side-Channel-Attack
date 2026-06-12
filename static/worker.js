/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;
/* Find the L3 size by running `getconf -a | grep CACHE` */
// For the user's system, LLC size is 8MB (8388608 bytes)
const LLCSIZE = 8 * 1024 * 1024; // 8MB in bytes
/* Collect traces for 10 seconds; you can vary this */
const TIME = 10000;
/* Collect traces every 10ms; you can vary this */
const P = 10;

function sweep(P) {
  /*
   * Implement this function to run a sweep of the cache.
   * 1. Allocate a buffer of size LLCSIZE.
   * 2. Read each cache line (read the buffer in steps of LINESIZE).
   * 3. Count the number of times each cache line is read in a time period of P milliseconds.
   * 4. Store the count in an array of size K, where K = TIME / P.
   * 5. Return the array of counts.
   */

  // Allocate a buffer of size LLCSIZE
  const buffer = new Uint8Array(LLCSIZE);

  // Initialize buffer with some values
  for (let i = 0; i < buffer.length; i++) {
    buffer[i] = 1;
  }

  // Calculate the number of measurements we'll take (K = TIME / P)
  const K = Math.floor(TIME / P);

  // Array to store the counts
  const counts = new Array(K).fill(0);

  // Run the sweep for a total of TIME milliseconds
  for (let k = 0; k < K; k++) {
    let count = 0;
    let sum = 0; // Dummy variable to ensure reads aren't optimized away

    const startTime = performance.now();

    // Keep reading the buffer until P milliseconds have passed
    while (performance.now() - startTime < P) {
      // Read through the buffer in steps of LINESIZE to access different cache lines
      for (let i = 0; i < buffer.length; i += LINESIZE) {
        sum+=buffer[i];
        //count++;

        // Check time after each complete buffer sweep to avoid excessive time checks
        // if (
        //   i + LINESIZE >= buffer.length &&
        //   performance.now() - startTime >= P
        // ) {
        //   break;
        // }
      }
        count++;
    }

    // Store the count of sweeps performed in this time window
    counts[k] = count;

    // Use sum to prevent compiler optimization
    if (sum === 0) {
      console.log("This should never happen");
    }
  }

  return counts;
}

self.addEventListener("message", function (e) {
  /* Call the sweep function and return the result */
  if (e.data === "start") {
    try {
      console.log("Starting sweep with P =", P, "ms");
      const results = sweep(P);
      console.log("Sweep completed:", results.length, "measurements collected");
      self.postMessage(results);
    } catch (error) {
      console.error("Error in worker:", error);
      self.postMessage({ error: error.message });
    }
  }
});
