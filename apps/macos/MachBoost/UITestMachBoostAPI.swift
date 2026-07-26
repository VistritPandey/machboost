#if DEBUG
import Foundation

final class UITestMachBoostAPI: MachBoostAPIProtocol, @unchecked Sendable {
    private let lock = NSLock()
    private var downloadedModels: Set<String> = []
    private var cancelledRequests: Set<String> = []
    private var chatRequestCount = 0

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
        [
            ModelInstance(
                model: "mlx-community/Qwen2.5-3B-Instruct-4bit",
                backend: "mlx",
                idleSeconds: 0.2,
                keepAliveSeconds: -1,
                requests: 1,
                capabilities: ["chat", "completion"],
                scheduler: .init(replicas: 1, activeRequests: 0, queuedRequests: 0)
            )
        ]
    }

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

    func stop(model: String?) async throws {}

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
        repository: String,
        backend: String = "mlx",
        capabilities: [String],
        cached: Bool,
        recommended: Bool,
        size: Double,
        memory: Double
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
            support: "ready",
            supportReason: "UI automation fixture"
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
        ]
    }

    private func fixtureResponse(
        for request: ChatRequest,
        requestNumber: Int
    ) -> String {
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

    private func wasCancelled(_ requestID: String) -> Bool {
        lock.withLock { cancelledRequests.contains(requestID) }
    }
}
#endif
