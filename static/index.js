function app() {
  return {
    /* This is the main app object containing all the application state and methods. */
    // The following properties are used to store the state of the application

    // results of cache latency measurements
    latencyResults: null,
    // local collection of trace data
    traceData: [],
    // Local collection of heapmap images
    heatmaps: [],
    // Latest website prediction
    latestPrediction: null,

    // Current status message
    status: "",
    // Is any worker running?
    isCollecting: false,
    // Is the status message an error?
    statusIsError: false,
    // Show trace data in the UI?
    showingTraces: false,

    // Collect latency data using warmup.js worker
    async collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyResults = null;
      this.statusIsError = false;
      this.showingTraces = false;

      try {
        // Create a worker
        let worker = new Worker("warmup.js");

        // Start the measurement and wait for result
        const results = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        // Update results
        this.latencyResults = results;
        this.status = "Latency data collection complete!";

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting latency data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Collect trace data using worker.js and send to backend
    async collectTraceData() {
      /*
       * Implement this function to collect trace data.
       * 1. Create a worker to run the sweep function.
       * 2. Collect the trace data from the worker.
       * 3. Send the trace data to the backend for temporary storage and heatmap generation.
       * 4. Fetch the heatmap from the backend and add it to the local collection.
       * 5. Handle errors and update the status.
       */
      this.isCollecting = true;
      this.status = "Collecting trace data...";
      this.statusIsError = false;
      this.showingTraces = true;

      try {
        // Create a worker
        let worker = new Worker("worker.js");

        // Start the measurement and wait for result
        const traceData = await new Promise((resolve, reject) => {
          worker.onmessage = (e) => {
            if (e.data.error) {
              reject(new Error(e.data.error));
            } else {
              resolve(e.data);
            }
          };
          worker.onerror = (error) => reject(error);
          worker.postMessage("start");
        });

        // Send trace data to backend for processing
        const response = await fetch("/collect_trace", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ trace: traceData }),
        });

        if (!response.ok) {
          throw new Error(
            `Server responded with ${response.status}: ${response.statusText}`
          );
        }

        const result = await response.json();

        // Debug: log the response to see what we're getting from the server
        console.log("Server response:", result);

        // Add the new trace and heatmap to our collections
        this.traceData.push(traceData);

        // Include both heatmap and stats in the heatmap object
        const heatmapWithStats = {
          ...result.heatmap,
          stats: result.stats,
          prediction: result.prediction || null
        };

        // Debug: log the combined object to verify it has the stats
        console.log("Heatmap with stats:", heatmapWithStats);

        this.heatmaps.push(heatmapWithStats);

        // Handle prediction results
        if (result.prediction) {
          this.latestPrediction = result.prediction;
          console.log("Website prediction:", result.prediction);
        }

        this.status = "Trace data collection complete!";

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting trace data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Download the trace data as JSON (array of arrays format for ML)
    async downloadTraces() {
      /*
       * Implement this function to download the trace data.
       * 1. Fetch the latest data from the backend API.
       * 2. Create a download file with the trace data in JSON format.
       * 3. Handle errors and update the status.
       */
      try {
        this.status = "Downloading traces...";
        this.statusIsError = false;

        // Fetch the latest data from backend
        const response = await fetch("/api/traces");
        if (!response.ok) {
          throw new Error(
            `Server responded with ${response.status}: ${response.statusText}`
          );
        }

        const tracesData = await response.json();

        // Create a download file
        const dataStr = JSON.stringify(tracesData);
        const dataUri =
          "data:application/json;charset=utf-8," + encodeURIComponent(dataStr);

        const exportName = `website_fingerprinting_traces_${new Date()
          .toISOString()
          .slice(0, 10)}.json`;

        // Create download link and trigger download
        const downloadLink = document.createElement("a");
        downloadLink.setAttribute("href", dataUri);
        downloadLink.setAttribute("download", exportName);
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);

        this.status = "Traces downloaded successfully!";
      } catch (error) {
        console.error("Error downloading traces:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      }
    },

    // Clear all results from the server
    async clearResults() {
      /*
       * Implement this function to clear all results from the server.
       * 1. Send a request to the backend API to clear all results.
       * 2. Clear local copies of trace data and heatmaps.
       * 3. Handle errors and update the status.
       */
      try {
        this.status = "Clearing results...";
        this.statusIsError = false;

        // Send request to backend
        const response = await fetch("/api/clear_results", {
          method: "POST",
        });

        if (!response.ok) {
          throw new Error(
            `Server responded with ${response.status}: ${response.statusText}`
          );
        }

        // Clear local data
        this.traceData = [];
        this.heatmaps = [];
        this.latestPrediction = null;

        this.status = "All results cleared successfully!";
      } catch (error) {
        console.error("Error clearing results:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      }
    },

    // Fetch existing results from the server
    async fetchResults() {
      try {
        // Fetch heatmaps
        const heatmapResponse = await fetch("/api/heatmaps");
        if (heatmapResponse.ok) {
          const heatmaps = await heatmapResponse.json();
          this.heatmaps = heatmaps.map((heatmap) => ({
            ...heatmap,
            stats: heatmap.stats || {},
          }));
          if (heatmaps.length > 0) {
            this.showingTraces = true;
          }
        }

        // Fetch model info to show available classes
        const modelResponse = await fetch("/api/model_info");
        if (modelResponse.ok) {
          const modelInfo = await modelResponse.json();
          console.log("Model info:", modelInfo);
        }
      } catch (error) {
        console.error("Error fetching results:", error);
      }
    },
  };
}
