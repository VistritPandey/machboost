#if DEBUG
import Foundation

final class UITestMachBoostAPI: MachBoostAPIProtocol, @unchecked Sendable {
    private let lock = NSLock()
    private var downloadedModels: Set<String> = []
    private var loadedRepositories: Set<String> = [
        "mlx-community/Qwen2.5-3B-Instruct-4bit"
    ]
    private var cancelledRequests: Set<String> = []
    private var chatRequestCount = 0
    private lazy var fixtureWorkspaceRoot: String = {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-ui-workspace-\(ProcessInfo.processInfo.processIdentifier)")
        let source = root.appendingPathComponent("Sources/App.swift")
        try? FileManager.default.createDirectory(
            at: source.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? "let greeting = \"hello\"\n".write(
            to: source,
            atomically: true,
            encoding: .utf8
        )
        return root.path
    }()

    func catalogSnapshot() -> [CatalogModel] {
        lock.withLock { makeCatalog() }
    }

    func catalog() async throws -> [CatalogModel] {
        catalogSnapshot()
    }

    func metrics() async throws -> ServerMetrics {
        ServerMetrics(
            schema: "machboost.metrics.v1",
            operations: .init(
                activeCount: 0,
                totals: .init(
                    started: 1,
                    completed: 1,
                    cancelled: cancelledRequestCount,
                    failed: 0,
                    generatedTokens: 2
                ),
                latencySeconds: .init(p50: 0.24, p95: 0.38),
                generationTokensPerSecond: 42
            ),
            models: try await models(),
            scheduler: .init(
                activeRequests: 0,
                queuedRequests: 0,
                rejectedRequests: 0
            ),
            process: .init(peakResidentMemoryBytes: 512 * 1_024 * 1_024)
        )
    }

    func models() async throws -> [ModelInstance] {
        lock.withLock { loadedRepositories.sorted().map(modelInstance) }
    }

    func workspaces() async throws -> [WorkspaceSummary] {
        [fixtureWorkspace()]
    }

    func reindexWorkspace(id: String) async throws -> WorkspaceSummary {
        fixtureWorkspace()
    }

    func removeWorkspace(id: String) async throws {}

    func preflight(model: String) async throws -> ModelPreflightResponse.Preflight {
        ModelPreflightResponse.Preflight(
            model: model,
            backend: model.contains("vl") ? "mlx-vlm" : "mlx",
            capabilities: model.contains("vl") ? ["chat", "vision"] : ["chat", "completion"],
            runtimeAvailable: true,
            cached: false,
            modelType: "fixture",
            supported: true,
            reason: "compatible"
        )
    }

    func load(
        model: String,
        keepAlive: String,
        warmup: Bool
    ) async throws -> ModelLoadResponse {
        let repository = lock.withLock {
            makeCatalog().first {
                $0.name == model || $0.repository == model
            }?.repository ?? model
        }
        let instance = modelInstance(repository)
        _ = lock.withLock { loadedRepositories.insert(repository) }
        return ModelLoadResponse(
            status: "success",
            model: model,
            loadDurationSeconds: 0.42,
            warmupDurationSeconds: warmup ? 0.08 : 0,
            warmupPerformed: warmup,
            instance: instance
        )
    }

    func stop(model: String?) async throws {
        lock.withLock {
            guard let model else {
                loadedRepositories.removeAll()
                return
            }
            let repository = makeCatalog().first {
                $0.name == model || $0.repository == model
            }?.repository ?? model
            loadedRepositories.remove(repository)
        }
    }

    func cancel(requestID: String) async throws -> Bool {
        _ = lock.withLock { cancelledRequests.insert(requestID) }
        return true
    }

    func streamChat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        let requestNumber = lock.withLock {
            chatRequestCount += 1
            return chatRequestCount
        }
        return AsyncThrowingStream { continuation in
            let task = Task<Void, Never> {
                do {
                    if self.isCodingFixture(request) {
                        try await self.streamCodingFixture(
                            request,
                            continuation: continuation
                        )
                        return
                    }
                    let response = self.fixtureResponse(
                        for: request,
                        requestNumber: requestNumber
                    )
                    let splitIndex = response.index(
                        response.startIndex,
                        offsetBy: max(1, response.count / 2)
                    )
                    let isCancellationFixture = request.messages.last?.content
                        == "Stop this fixture response"
                    try await Task.sleep(
                        for: .seconds(isCancellationFixture ? 15 : 5)
                    )
                    if self.wasCancelled(request.requestID) {
                        continuation.yield(self.cancelledChatEvent(requestID: request.requestID))
                        continuation.finish()
                        return
                    }
                    if
                        request.model == "muse-glimmer:30b",
                        request.messages.last?.content == "Use Muse tools"
                    {
                        continuation.yield(
                            self.museReasoningEvent(requestID: request.requestID)
                        )
                        continuation.yield(
                            self.museToolEvent(requestID: request.requestID)
                        )
                    }
                    continuation.yield(
                        self.chatEvent(
                            requestID: request.requestID,
                            content: String(response[..<splitIndex]),
                            done: false
                        )
                    )
                    try await Task.sleep(for: .milliseconds(200))
                    if self.wasCancelled(request.requestID) {
                        continuation.yield(self.cancelledChatEvent(requestID: request.requestID))
                    } else {
                        continuation.yield(
                            self.chatEvent(
                                requestID: request.requestID,
                                content: String(response[splitIndex...]),
                                done: true
                            )
                        )
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func isCodingFixture(_ request: ChatRequest) -> Bool {
        let explicitlyEnabled = ProcessInfo.processInfo.environment[
            "MACHBOOST_UI_TEST_CODING"
        ] == "1"
        return (explicitlyEnabled && request.tools?.isEmpty == false) || request.messages.contains {
            $0.role == "user" && $0.content == "Exercise coding agent"
        }
    }

    private func streamCodingFixture(
        _ request: ChatRequest,
        continuation: AsyncThrowingStream<ChatEvent, Error>.Continuation
    ) async throws {
        try await Task.sleep(for: .milliseconds(120))
        let completedTools = request.messages.filter { $0.role == "tool" }.count
        if completedTools == 0 {
            continuation.yield(museReasoningEvent(requestID: request.requestID))
            continuation.yield(
                toolEvent(
                    requestID: request.requestID,
                    calls: [
                        .init(
                            id: "list-1",
                            type: "function",
                            function: .init(
                                name: "list_files",
                                arguments: .object(["path": .string("Sources")])
                            )
                        ),
                        .init(
                            id: "read-1",
                            type: "function",
                            function: .init(
                                name: "read_file",
                                arguments: .object(["path": .string("Sources/App.swift")])
                            )
                        ),
                    ]
                )
            )
            continuation.yield(chatEvent(requestID: request.requestID, content: "", done: true))
        } else if completedTools == 2 {
            continuation.yield(
                toolEvent(
                    requestID: request.requestID,
                    calls: [
                        .init(
                            id: "edit-1",
                            type: "function",
                            function: .init(
                                name: "replace_in_file",
                                arguments: .object([
                                    "path": .string("Sources/App.swift"),
                                    "old_text": .string("hello"),
                                    "new_text": .string("hello team"),
                                ])
                            )
                        ),
                    ]
                )
            )
            continuation.yield(chatEvent(requestID: request.requestID, content: "", done: true))
        } else {
            continuation.yield(
                chatEvent(
                    requestID: request.requestID,
                    content: "Reviewed the repository after three tool results.",
                    done: true
                )
            )
        }
        continuation.finish()
    }

    func streamPull(
        model: String,
        requestID: String
    ) -> AsyncThrowingStream<PullEvent, Error> {
        AsyncThrowingStream { continuation in
            continuation.yield(
                PullEvent(
                    requestID: requestID,
                    status: "downloading",
                    file: "weights.safetensors",
                    completed: 4,
                    total: 8,
                    unit: "bytes",
                    done: false,
                    path: nil,
                    error: nil
                )
            )
            _ = lock.withLock { downloadedModels.insert(model) }
            continuation.yield(
                PullEvent(
                    requestID: requestID,
                    status: "success",
                    file: nil,
                    completed: 8,
                    total: 8,
                    unit: "bytes",
                    done: true,
                    path: "/tmp/machboost-ui-fixture",
                    error: nil
                )
            )
            continuation.finish()
        }
    }

    private func model(
        name: String,
        displayName: String,
        repository: String?,
        backend: String = "mlx",
        capabilities: [String],
        cached: Bool,
        recommended: Bool,
        size: Double,
        memory: Double,
        contextLength: Int? = nil,
        sourceRepository: String? = nil
    ) -> CatalogModel {
        CatalogModel(
            name: name,
            displayName: displayName,
            repository: repository,
            backend: backend,
            capabilities: capabilities,
            cached: cached,
            cachedPath: cached ? "/tmp/machboost-ui-fixture/\(name)" : nil,
            recommended: recommended,
            tested: true,
            downloadSizeGB: size,
            diskSizeGB: cached ? size : nil,
            minimumMemoryGB: memory,
            contextLength: contextLength,
            sourceRepository: sourceRepository,
            support: "ready",
            supportReason: "UI automation fixture"
        )
    }

    private func modelInstance(_ repository: String) -> ModelInstance {
        let isVision = repository.localizedCaseInsensitiveContains("vl")
        return ModelInstance(
            model: repository,
            backend: isVision ? "mlx-vlm" : "mlx",
            idleSeconds: 0.2,
            keepAliveSeconds: -1,
            requests: 1,
            capabilities: isVision ? ["chat", "vision"] : ["chat", "completion"],
            scheduler: .init(replicas: 1, activeRequests: 0, queuedRequests: 0)
        )
    }

    private func makeCatalog() -> [CatalogModel] {
        let startsEmpty = ProcessInfo.processInfo.environment[
            "MACHBOOST_UI_TEST_NO_CACHED_MODELS"
        ] == "1"
        return [
            model(
                name: "qwen2.5:3b",
                displayName: "Qwen2.5 3B",
                repository: "mlx-community/Qwen2.5-3B-Instruct-4bit",
                capabilities: ["chat", "completion"],
                cached: !startsEmpty || downloadedModels.contains("qwen2.5:3b"),
                recommended: true,
                size: 1.9,
                memory: 8
            ),
            model(
                name: "qwen2.5-vl:3b",
                displayName: "Qwen2.5 VL 3B",
                repository: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
                backend: "mlx-vlm",
                capabilities: ["chat", "vision"],
                cached: !startsEmpty || downloadedModels.contains("qwen2.5-vl:3b"),
                recommended: true,
                size: 2.4,
                memory: 12
            ),
            model(
                name: "llama3.2:1b",
                displayName: "Llama 3.2 1B",
                repository: "mlx-community/Llama-3.2-1B-Instruct-4bit",
                capabilities: ["chat", "completion"],
                cached: downloadedModels.contains("llama3.2:1b"),
                recommended: false,
                size: 0.8,
                memory: 4
            ),
            model(
                name: "muse-glimmer:30b",
                displayName: "Muse Glimmer 30B",
                repository: "mlx-community/Muse-Glimmer-30B-4bit",
                backend: "mlx-vlm",
                capabilities: ["chat", "completion", "vision", "reasoning", "tools"],
                cached: !startsEmpty || downloadedModels.contains("muse-glimmer:30b"),
                recommended: true,
                size: 6,
                memory: 32,
                contextLength: 131_072,
                sourceRepository: "meta-models/Muse-Glimmer-30B"
            ),
        ]
    }

    private func fixtureResponse(
        for request: ChatRequest,
        requestNumber: Int
    ) -> String {
        if request.messages.last?.content == "Show a long Markdown response" {
            let lines = (1...18).map {
                "- Check \($0): validated the repository path and retained its citation."
            }
            return """
            ## Repository review

            This response exercises a long, selectable Markdown surface while tokens stream.

            \(lines.joined(separator: "\n"))

            ```swift
            let endpoint = URL(string: "http://127.0.0.1:11435/v1")!
            let model = "qwen2.5:3b"
            ```

            STREAM END MARKER
            """
        }
        let imageCount = request.messages.last?.images?.count ?? 0
        let contextCount = request.context.count
        guard imageCount > 0 || contextCount > 0 else {
            return requestNumber == 1
                ? "Fixture response."
                : "Regenerated fixture response."
        }
        return "Fixture context: \(imageCount) image, \(contextCount) file."
    }

    private var cancelledRequestCount: Int {
        lock.withLock { cancelledRequests.count }
    }

    private func chatEvent(
        requestID: String,
        content: String,
        done: Bool
    ) -> ChatEvent {
        ChatEvent(
            requestID: requestID,
            message: .init(role: "assistant", content: content),
            done: done,
            doneReason: done ? "stop" : nil,
            totalDuration: done ? 240_000_000 : nil,
            evalDuration: done ? 100_000_000 : nil,
            evalCount: done ? 2 : nil,
            machboost: done
                ? .init(
                    backend: "mlx",
                    stats: .init(
                        generatedTokens: 2,
                        generationSeconds: 0.1,
                        promptTokens: 8
                    ),
                    timeToFirstTokenSeconds: 0.12
                )
                : nil,
            error: nil
        )
    }

    private func cancelledChatEvent(requestID: String) -> ChatEvent {
        ChatEvent(
            requestID: requestID,
            message: .init(role: "assistant", content: ""),
            done: true,
            doneReason: "cancelled",
            totalDuration: nil,
            evalDuration: nil,
            evalCount: nil,
            machboost: nil,
            error: nil
        )
    }

    private func museReasoningEvent(requestID: String) -> ChatEvent {
        ChatEvent(
            requestID: requestID,
            message: .init(
                role: "assistant",
                content: "",
                thinking: "I should inspect the repository before answering."
            ),
            done: false,
            doneReason: nil,
            totalDuration: nil,
            evalDuration: nil,
            evalCount: nil,
            machboost: nil,
            error: nil
        )
    }

    private func museToolEvent(requestID: String) -> ChatEvent {
        ChatEvent(
            requestID: requestID,
            message: .init(
                role: "assistant",
                content: "",
                toolCalls: [
                    .init(
                        function: .init(
                            name: "search_repository",
                            arguments: .object(["query": .string("request cancellation")])
                        )
                    )
                ]
            ),
            done: false,
            doneReason: nil,
            totalDuration: nil,
            evalDuration: nil,
            evalCount: nil,
            machboost: nil,
            error: nil
        )
    }

    private func toolEvent(requestID: String, calls: [APIToolCall]) -> ChatEvent {
        ChatEvent(
            requestID: requestID,
            message: .init(role: "assistant", content: "", toolCalls: calls),
            done: false,
            doneReason: nil,
            totalDuration: nil,
            evalDuration: nil,
            evalCount: nil,
            machboost: nil,
            error: nil
        )
    }

    private func fixtureWorkspace() -> WorkspaceSummary {
        WorkspaceSummary(
            id: "workspace-ui-fixture",
            name: "MachBoost fixture",
            path: fixtureWorkspaceRoot,
            createdAt: "2026-08-16T00:00:00Z",
            updatedAt: "2026-08-16T00:00:00Z",
            indexedAt: "2026-08-16T00:00:00Z",
            revision: "fixture-revision",
            fileCount: 1,
            chunkCount: 1,
            totalBytes: 23,
            languages: [.init(name: "Swift", files: 1)]
        )
    }

    private func wasCancelled(_ requestID: String) -> Bool {
        lock.withLock { cancelledRequests.contains(requestID) }
    }
}
#endif
