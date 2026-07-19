import { chmodSync } from "node:fs";

// TypeScript preserves the shebang but not the executable mode required by
// npm's POSIX bin shim. chmod is harmless on Windows.
chmodSync(new URL("../dist/index.js", import.meta.url), 0o755);
