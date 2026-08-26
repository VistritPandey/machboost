import Darwin
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
    private(set) var authenticationRequired = false

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
        authenticationRequired = false
        let api = MachBoostAPI(endpoint: configuration.endpoint, apiToken: apiToken)
        if let health = try? await api.serverHealth(), health.isReady {
            authenticationRequired = health.requiresAuthentication
            var canAuthenticate = true
            if health.requiresAuthentication {
                canAuthenticate = (try? await api.authenticatedServerVersion()) != nil
            }
            if !canAuthenticate {
                try await reclaimBundledDaemon(on: configuration.port)
                authenticationRequired = false
            } else if let serverVersion = health.version,
                      Self.isOlderVersion(serverVersion, than: Self.applicationVersion()) {
                let appVersion = Self.applicationVersion()
                try await stopOlderDaemon(
                    api: api,
                    serverVersion: serverVersion,
                    appVersion: appVersion
                )
            } else {
                ownsProcess = false
                state = .running
                return
            }
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
                self.authenticationRequired = false
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
        authenticationRequired = configuration.lanEnabled

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
        authenticationRequired = false
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
        authenticationRequired = false
        state = .stopped
    }

    func runCLI(
        _ arguments: [String],
        apiToken: String? = nil
    ) async throws -> String {
        let launch = try runtimeLaunch()
        let environment = Self.launchEnvironment(
            base: ProcessInfo.processInfo.environment,
            apiToken: apiToken
        )
        return try await Task.detached(priority: .userInitiated) {
            let process = Process()
            let output = Pipe()
            process.executableURL = launch.executable
            process.arguments = launch.prefixArguments + ["-m", "machboost.cli"] + arguments
            process.currentDirectoryURL = launch.workingDirectory
            process.environment = environment
            process.standardOutput = output
            process.standardError = output
            try process.run()
            process.waitUntilExit()
            let data = output.fileHandleForReading.readDataToEndOfFile()
            let text = String(data: data, encoding: .utf8) ?? ""
            guard process.terminationStatus == 0 else {
                throw DaemonError.commandFailed(
                    text.trimmingCharacters(in: .whitespacesAndNewlines)
                )
            }
            return text
        }.value
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

    private func stopOlderDaemon(
        api: MachBoostAPI,
        serverVersion: String,
        appVersion: String
    ) async throws {
        appendLog(
            "Replacing MachBoost daemon \(serverVersion) with bundled version \(appVersion)."
        )
        do {
            try await api.shutdown()
        } catch {
            throw DaemonError.incompatibleDaemon(
                running: serverVersion,
                expected: appVersion
            )
        }
        for _ in 0..<50 {
            if (try? await api.health(timeoutInterval: 0.15)) != true {
                return
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        throw DaemonError.incompatibleDaemon(
            running: serverVersion,
            expected: appVersion
        )
    }

    private func appendLog(_ text: String) {
        recentLogs.append(contentsOf: text.split(whereSeparator: { $0.isNewline }).map(String.init))
        if recentLogs.count > 300 {
            recentLogs.removeFirst(recentLogs.count - 300)
        }
    }

    private func reclaimBundledDaemon(on port: Int) async throws {
        let listeners = Self.listenerPIDs(on: port)
        guard !listeners.isEmpty else { return }
        for pid in listeners {
            let command = Self.processCommand(pid: pid)
            guard Self.isBundledDaemonCommand(command, port: port) else {
                throw DaemonError.unrecognizedSecuredDaemon(port: port)
            }
            guard Darwin.kill(pid, SIGTERM) == 0 else {
                throw DaemonError.unrecognizedSecuredDaemon(port: port)
            }
        }
        for _ in 0..<50 {
            if Self.listenerPIDs(on: port).isEmpty {
                return
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        throw DaemonError.unrecognizedSecuredDaemon(port: port)
    }

    static func isBundledDaemonCommand(_ command: String, port: Int) -> Bool {
        command.contains("/Contents/Resources/runtime/python/bin/python")
            && command.contains("-m machboost.cli serve")
            && command.contains("--port \(port)")
    }

    private static func listenerPIDs(on port: Int) -> [Int32] {
        commandOutput(
            executable: "/usr/sbin/lsof",
            arguments: ["-nP", "-tiTCP:\(port)", "-sTCP:LISTEN"]
        )
        .split(whereSeparator: { $0.isWhitespace })
        .compactMap { Int32($0) }
    }

    private static func processCommand(pid: Int32) -> String {
        commandOutput(
            executable: "/bin/ps",
            arguments: ["-p", String(pid), "-o", "command="]
        )
        .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func commandOutput(executable: String, arguments: [String]) -> String {
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return ""
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8) ?? ""
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

    static func isOlderVersion(_ serverVersion: String, than appVersion: String) -> Bool {
        serverVersion.compare(appVersion, options: .numeric) == .orderedAscending
    }

    private static func applicationVersion() -> String {
        Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String ?? "0"
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

private struct RuntimeLaunch: Sendable {
    let executable: URL
    let prefixArguments: [String]
    let workingDirectory: URL
}

enum DaemonError: LocalizedError {
    case runtimeMissing
    case timedOut
    case exited(Int32)
    case externallyManaged
    case unrecognizedSecuredDaemon(port: Int)
    case incompatibleDaemon(running: String, expected: String)
    case commandFailed(String)

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
        case let .unrecognizedSecuredDaemon(port):
            "An authenticated server the app cannot safely replace is using port \(port). Stop that server, then reopen MachBoost."
        case let .incompatibleDaemon(running, expected):
            "MachBoost \(running) is already using the local server port and could not be replaced by \(expected). Quit the older MachBoost process and reopen the app."
        case let .commandFailed(message):
            message.isEmpty ? "The MachBoost command failed." : message
        }
    }
}
