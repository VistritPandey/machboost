import Foundation
import Observation

@MainActor
@Observable
final class DaemonManager {
    enum State: Equatable {
        case stopped
        case starting
        case running
        case stopping
        case failed(String)
    }

    private(set) var state: State = .stopped
    private(set) var recentLogs: [String] = []
    private(set) var ownsProcess = false

    private var process: Process?
    private var outputPipe: Pipe?
    private let sourceRootOverride: URL?

    init(sourceRootOverride: URL? = nil) {
        self.sourceRootOverride = sourceRootOverride
    }

    func start(
        configuration: ServerConfiguration,
        apiToken: String?
    ) async throws {
        guard state != .starting, state != .running else { return }
        state = .starting
        let api = MachBoostAPI(endpoint: configuration.endpoint, apiToken: apiToken)
        if (try? await api.health()) == true {
            ownsProcess = false
            state = .running
            return
        }

        let launch = try runtimeLaunch()
        let teamDatabase = try teamDatabaseURL()
        let process = Process()
        process.executableURL = launch.executable
        process.arguments = launch.prefixArguments + [
            "-m", "machboost.cli", "serve",
            "--host", configuration.bindHost,
            "--port", String(configuration.port),
            "--replicas", String(configuration.replicas),
            "--max-queue", String(configuration.maxQueue),
            "--queue-timeout", String(configuration.queueTimeout),
            "--team",
            "--team-db", teamDatabase.path,
        ]
        if configuration.lanEnabled {
            process.arguments?.append("--require-auth")
        }
        process.currentDirectoryURL = launch.workingDirectory
        process.environment = Self.launchEnvironment(
            base: ProcessInfo.processInfo.environment,
            apiToken: apiToken
        )

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in self?.appendLog(text) }
        }
        process.terminationHandler = { [weak self] process in
            Task { @MainActor in
                guard let self, self.process === process else { return }
                self.outputPipe?.fileHandleForReading.readabilityHandler = nil
                self.process = nil
                self.outputPipe = nil
                self.ownsProcess = false
                if self.state != .stopping {
                    self.state = process.terminationStatus == 0
                        ? .stopped
                        : .failed("Daemon exited with status \(process.terminationStatus).")
                }
            }
        }

        do {
            try process.run()
        } catch {
            state = .failed(error.localizedDescription)
            throw error
        }
        self.process = process
        self.outputPipe = pipe
        ownsProcess = true

        do {
            try await waitUntilReady(api: api, process: process)
            state = .running
        } catch {
            process.terminate()
            state = .failed(error.localizedDescription)
            throw error
        }
    }

    private func teamDatabaseURL() throws -> URL {
        let root = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        .appendingPathComponent("MachBoost", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        return root.appendingPathComponent("team.sqlite3")
    }

    func restart(
        currentEndpoint: URL,
        currentAPIToken: String?,
        configuration: ServerConfiguration,
        apiToken: String?
    ) async throws {
        if state == .running, !ownsProcess {
            throw DaemonError.externallyManaged
        }
        await shutdown(endpoint: currentEndpoint, apiToken: currentAPIToken)
        try await start(configuration: configuration, apiToken: apiToken)
    }

    func shutdown(endpoint: URL, apiToken: String?) async {
        guard state != .stopped else { return }
        state = .stopping
        if ownsProcess {
            let api = MachBoostAPI(endpoint: endpoint, apiToken: apiToken)
            try? await api.shutdown()
            if let process {
                for _ in 0..<30 where process.isRunning {
                    try? await Task.sleep(for: .milliseconds(100))
                }
                if process.isRunning {
                    process.terminate()
                }
            }
        }
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        process = nil
        outputPipe = nil
        ownsProcess = false
        state = .stopped
    }

    func terminateImmediately() {
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        if ownsProcess, process?.isRunning == true {
            process?.terminate()
        }
        process = nil
        outputPipe = nil
        ownsProcess = false
        state = .stopped
    }

    private func waitUntilReady(api: MachBoostAPI, process: Process) async throws {
        for _ in 0..<300 {
            if !process.isRunning {
                throw DaemonError.exited(process.terminationStatus)
            }
            if (try? await api.health()) == true {
                return
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        throw DaemonError.timedOut
    }

    private func appendLog(_ text: String) {
        recentLogs.append(contentsOf: text.split(whereSeparator: { $0.isNewline }).map(String.init))
        if recentLogs.count > 300 {
            recentLogs.removeFirst(recentLogs.count - 300)
        }
    }

    static func launchEnvironment(
        base: [String: String],
        apiToken: String?
    ) -> [String: String] {
        var environment = base
        environment["PYTHONUNBUFFERED"] = "1"
        // The app bundle is code-signed and immutable. Python bytecode written into
        // the embedded runtime would invalidate its sealed-resource signature.
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        if let apiToken, !apiToken.isEmpty {
            environment["MACHBOOST_API_TOKEN"] = apiToken
        } else {
            environment.removeValue(forKey: "MACHBOOST_API_TOKEN")
        }
        return environment
    }

    private func runtimeLaunch() throws -> RuntimeLaunch {
        #if DEBUG
        if let sourceRootOverride {
            return RuntimeLaunch(
                executable: URL(fileURLWithPath: "/usr/bin/env"),
                prefixArguments: ["python3"],
                workingDirectory: sourceRootOverride
            )
        }
        #endif

        if let resources = Bundle.main.resourceURL {
            let embedded = resources
                .appendingPathComponent("runtime", isDirectory: true)
                .appendingPathComponent("python", isDirectory: true)
                .appendingPathComponent("bin", isDirectory: true)
                .appendingPathComponent("python3", isDirectory: false)
            if FileManager.default.isExecutableFile(atPath: embedded.path) {
                return RuntimeLaunch(
                    executable: embedded,
                    prefixArguments: [],
                    workingDirectory: resources
                )
            }
        }

        #if DEBUG
        let sourceRoot = developerSourceRoot()
        return RuntimeLaunch(
            executable: URL(fileURLWithPath: "/usr/bin/env"),
            prefixArguments: ["python3"],
            workingDirectory: sourceRoot
        )
        #else
        throw DaemonError.runtimeMissing
        #endif
    }

    private func developerSourceRoot() -> URL {
        if let configured = ProcessInfo.processInfo.environment["MACHBOOST_SOURCE_ROOT"] {
            return URL(fileURLWithPath: configured, isDirectory: true)
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}

private struct RuntimeLaunch {
    let executable: URL
    let prefixArguments: [String]
    let workingDirectory: URL
}

enum DaemonError: LocalizedError {
    case runtimeMissing
    case timedOut
    case exited(Int32)
    case externallyManaged

    var errorDescription: String? {
        switch self {
        case .runtimeMissing:
            "The bundled MachBoost runtime is missing or damaged."
        case .timedOut:
            "The MachBoost daemon did not become ready within 30 seconds."
        case let .exited(status):
            "The MachBoost daemon exited with status \(status)."
        case .externallyManaged:
            "This server was started outside the app. Stop it before changing app server settings."
        }
    }
}
