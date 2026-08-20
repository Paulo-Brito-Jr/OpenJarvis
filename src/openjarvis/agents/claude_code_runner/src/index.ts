/**
 * Fail-closed entrypoint for the legacy Claude Code runner.
 *
 * No external SDK is imported here: module loading can itself have side
 * effects. The runner remains disabled until it receives an authenticated
 * OpenJarvis capability decision and runs inside a verified isolated sandbox.
 */

const OUTPUT_START = "---OPENJARVIS_OUTPUT_START---";
const OUTPUT_END = "---OPENJARVIS_OUTPUT_END---";

interface RunnerResponse {
  content: string;
  tool_results: [];
  metadata: {
    error: true;
    security_disabled?: true;
    reason?: string;
  };
}

function emitResult(response: RunnerResponse): void {
  console.log(OUTPUT_START);
  console.log(JSON.stringify(response));
  console.log(OUTPUT_END);
}

function emitError(
  message: string,
  metadata: RunnerResponse["metadata"],
): void {
  emitResult({
    content: message,
    tool_results: [],
    metadata,
  });
}

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (chunk: string) => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", (error: Error) => reject(error));
  });
}

async function main(): Promise<void> {
  try {
    JSON.parse(await readStdin());
  } catch (error) {
    emitError(`Failed to parse input: ${error}`, { error: true });
    process.exit(1);
  }

  emitError(
    "Claude Code runner disabled: verified capability and sandbox context required.",
    {
      error: true,
      security_disabled: true,
      reason: "unverified_external_sandbox",
    },
  );
  process.exit(2);
}

void main();
