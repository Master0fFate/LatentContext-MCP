import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const projectRoot = fileURLToPath(new URL("..", import.meta.url));
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const packageManifest = JSON.parse(readFileSync(join(projectRoot, "package.json"), "utf8")) as {
    name: string;
    version: string;
    bin: Record<string, string>;
};
const tempDirs: string[] = [];

after(() => {
    for (const dir of tempDirs.splice(0)) {
        rmSync(dir, { recursive: true, force: true });
    }
});

function makeTempDir(prefix: string): string {
    const dir = mkdtempSync(join(tmpdir(), prefix));
    tempDirs.push(dir);
    return dir;
}

function runNpm(args: string[], cwd: string, encoding?: "utf8"): string | Buffer {
    return execFileSync(npmCommand, args, {
        cwd,
        encoding,
        stdio: encoding ? ["ignore", "pipe", "pipe"] : "pipe",
        shell: process.platform === "win32",
        // npm publish --dry-run propagates this flag to prepublishOnly. The
        // nested npm pack must produce a real tarball for this smoke test.
        env: { ...process.env, npm_config_dry_run: "false" },
    });
}

function toolText(result: { content: Array<{ type: string; text?: string }> }): string {
    const content = result.content.find((item) => item.type === "text");
    assert.ok(content?.text, "tool response must contain text content");
    return content.text;
}

function hasInstalledDependency(packageDir: string, projectDir: string, dependency: string): boolean {
    for (let directory = packageDir; ; directory = dirname(directory)) {
        if (existsSync(join(directory, "node_modules", dependency, "package.json"))) return true;
        if (directory === projectDir || dirname(directory) === directory) return false;
    }
}

function resolveFromInstalledPackage(packageDir: string, specifier: string): string {
    // Resolve only: importing transformers initializes its runtime and may later
    // fetch a model. createRequire is explicitly anchored at the packed package.
    const source = [
        'import { createRequire } from "node:module";',
        'import { join } from "node:path";',
        'const require = createRequire(join(process.cwd(), "package.json"));',
        `process.stdout.write(require.resolve(${JSON.stringify(specifier)}));`,
    ].join("\n");
    return (execFileSync(process.execPath, ["--input-type=module", "--eval", source], {
        cwd: packageDir,
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
        // Resolution does not make network requests; these guards make any
        // accidental future import in this probe fail offline instead.
        env: { ...process.env, HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" },
    }) as string).trim();
}

test("packed build resolves the default embedding runtime offline and completes an isolated MCP memory workflow", async () => {
    // Build first so `npm pack` archives the current source, not a stale dist
    // directory. The production install below intentionally has no access to
    // this checkout's node_modules.
    runNpm(["run", "build"], projectRoot);

    const packDir = makeTempDir("latentcontext-mcp-pack-");
    const packed = JSON.parse(runNpm(
        ["pack", "--json", "--pack-destination", packDir],
        projectRoot,
        "utf8"
    ) as string) as Array<{ filename: string }>;
    const tarballName = packed[0]?.filename;
    assert.ok(tarballName, "npm pack must produce a tarball");
    const tarball = join(packDir, tarballName);
    assert.ok(existsSync(tarball), "the packed tarball must exist");

    const installProject = makeTempDir("latentcontext-mcp-install-");
    writeFileSync(join(installProject, "package.json"), JSON.stringify({
        name: "latentcontext-mcp-packed-smoke",
        private: true,
        version: "1.0.0",
    }));
    // Install only production dependencies from the actual tarball. This
    // catches both omitted publish files and runtime dependencies incorrectly
    // left in devDependencies.
    // `--offline` makes npm fail rather than reaching the registry. The test
    // consumes the dependency cache populated by the project's locked install,
    // while the packed artifact itself remains the only package under test.
    runNpm(["install", "--offline", "--omit=dev", "--no-audit", "--no-fund", tarball], installProject);

    const installedPackageDir = join(installProject, "node_modules", packageManifest.name);
    const installedManifest = JSON.parse(readFileSync(join(installedPackageDir, "package.json"), "utf8")) as {
        version: string;
        bin: Record<string, string>;
        dependencies?: Record<string, string>;
    };
    assert.equal(installedManifest.version, packageManifest.version, "npm must install the packed artifact");
    assert.deepEqual(installedManifest.bin, packageManifest.bin, "the packed artifact must retain its bin declaration");

    const [binName] = Object.keys(installedManifest.bin);
    assert.ok(binName, "the packed artifact must declare a binary");
    const bin = join(
        installProject,
        "node_modules",
        ".bin",
        `${binName}${process.platform === "win32" ? ".cmd" : ""}`
    );
    assert.ok(existsSync(bin), "the installed package must expose its declared bin");
    if (process.platform !== "win32") {
        assert.notEqual(statSync(bin).mode & 0o111, 0, "the installed bin must be executable on POSIX");
    }

    // The default embedding provider lazily imports this runtime. Assert its
    // production declaration explicitly instead of deriving the assertion from
    // the manifest: a move to devDependencies or its removal must fail here.
    const transformersDependency = "@huggingface/transformers";
    assert.ok(
        Object.hasOwn(installedManifest.dependencies ?? {}, transformersDependency),
        `${transformersDependency} must be declared in production dependencies`
    );

    // Check every declared production dependency at the installed package
    // boundary. This catches a dependency absent from the tarball install even
    // when this workflow does not activate its lazy path. The server workflow
    // below verifies the dependencies used at startup and during memory
    // operations can actually be loaded.
    for (const dependency of Object.keys(installedManifest.dependencies ?? {})) {
        assert.ok(
            hasInstalledDependency(installedPackageDir, installProject, dependency),
            `installed package must contain production dependency ${dependency}`
        );
    }

    // This is deliberately a resolution probe, not an import or embed call:
    // no embedding model is initialized or downloaded. Its subprocess is
    // rooted at the isolated packed package, not this checkout.
    const transformersRuntime = resolveFromInstalledPackage(installedPackageDir, transformersDependency);
    assert.match(transformersRuntime, /@huggingface[\\/]transformers[\\/]/,
        "the isolated package must resolve the default embedding runtime");

    const dataDir = makeTempDir("latentcontext-mcp-package-data-");
    writeFileSync(join(dataDir, "latentcontext.config.json"), JSON.stringify({
        embedding: { provider: "none" },
    }));

    // Run the npm-generated bin shim rather than dist/index.js directly. A
    // missing bin target or missing production runtime dependency must make
    // this connection fail.
    const transport = new StdioClientTransport({
        command: bin,
        cwd: installProject,
        env: { LATENTCONTEXT_DATA_DIR: dataDir },
        stderr: "pipe",
    });
    const protocolErrors: Error[] = [];
    let stderr = "";
    transport.onerror = (error) => protocolErrors.push(error);
    transport.stderr?.on("data", (chunk: Buffer) => {
        stderr += chunk.toString();
    });

    const client = new Client({ name: "latentcontext-package-smoke", version: "1.0.0" });
    try {
        await client.connect(transport, { timeout: 10_000 });
        assert.deepEqual(client.getServerVersion(), {
            name: packageManifest.name,
            version: packageManifest.version,
        });

        const tools = await client.listTools(undefined, { timeout: 10_000 });
        assert.ok(tools.tools.some((tool) => tool.name === "session_start"));
        assert.ok(tools.tools.some((tool) => tool.name === "memory_store"));
        assert.ok(tools.tools.some((tool) => tool.name === "memory_retrieve"));

        const started = toolText(await client.callTool(
            { name: "session_start", arguments: {} }, undefined, { timeout: 10_000 }
        ));
        assert.match(started, /^New session started: /m);

        const memory = "The packaged server smoke test stores this detailed release workflow note in a temporary isolated data directory without embeddings enabled.";
        assert.match(toolText(await client.callTool(
            { name: "memory_store", arguments: { content: memory, memory_type: "event" } },
            undefined,
            { timeout: 10_000 }
        )), /^Stored as event \(Tier 0\)$/m);

        assert.match(toolText(await client.callTool(
            { name: "memory_retrieve", arguments: { query: "packaged release workflow", token_budget: 1_000 } },
            undefined,
            { timeout: 10_000 }
        )), new RegExp(memory));

        await client.callTool({ name: "session_start", arguments: {} }, undefined, { timeout: 10_000 });
        const afterReset = toolText(await client.callTool(
            { name: "memory_retrieve", arguments: { query: "packaged release workflow", token_budget: 1_000 } },
            undefined,
            { timeout: 10_000 }
        ));
        assert.doesNotMatch(afterReset, new RegExp(memory), "a new session must not retrieve prior session data");

        assert.deepEqual(protocolErrors, [], "the package must preserve JSON-RPC stdout framing");
        assert.equal(stderr, "", "the package must not write diagnostics to stderr");
    } finally {
        await client.close().catch(() => undefined);
    }
});
