import Foundation

public enum MachBoostAPIError: LocalizedError {
    case invalidResponse
    case server(status: Int, message: String)
    case stream(String)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "MachBoost returned an invalid response."
        case let .server(status, message):
            "MachBoost server error \(status): \(message)"
        case let .stream(message):
            message
        }
    }
}

public protocol MachBoostAPIProtocol: AnyObject, Sendable {
    func catalog() async throws -> [CatalogModel]
    func metrics() async throws -> ServerMetrics
    func models() async throws -> [ModelInstance]
    func workspaces() async throws -> [WorkspaceSummary]
    func registerWorkspace(path: String, name: String?) async throws -> WorkspaceSummary
    func reindexWorkspace(id: String) async throws -> WorkspaceSummary
    func removeWorkspace(id: String) async throws
    func teamStatus() async throws -> TeamStatus?
    func teamKeys() async throws -> [TeamKey]
    func createTeamKey(
        name: String,
        scopes: [String],
        allowedModels: [String],
        maxConcurrent: Int,
        requestsPerMinute: Int
    ) async throws -> CreatedTeamKeyResponse
    func revokeTeamKey(id: String) async throws
    func updateTeamSettings(
        traceMode: String,
        retentionDays: Int?,
        maxStorageBytes: Int64
    ) async throws -> TeamSettings
    func traces(limit: Int) async throws -> [TraceSummary]
    func evaluations(limit: Int) async throws -> [TraceEvaluation]
    func evaluate(
        traceIDs: [String],
        name: String,
        model: String?
    ) async throws -> TraceEvaluation
    func preflight(model: String) async throws -> ModelPreflightResponse.Preflight
    func load(
        model: String,
        keepAlive: String,
        warmup: Bool
    ) async throws -> ModelLoadResponse
    func stop(model: String?) async throws
    func cancel(requestID: String) async throws -> Bool
    func streamChat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error>
    func streamPull(model: String, requestID: String) -> AsyncThrowingStream<PullEvent, Error>
}

public extension MachBoostAPIProtocol {
    func stop() async throws {
        try await stop(model: nil)
    }

    func workspaces() async throws -> [WorkspaceSummary] {
        []
    }

    func registerWorkspace(path: String, name: String?) async throws -> WorkspaceSummary {
        throw MachBoostAPIError.server(
            status: 501,
            message: "This client does not support repository workspaces."
        )
    }

    func reindexWorkspace(id: String) async throws -> WorkspaceSummary {
        throw MachBoostAPIError.server(
            status: 501,
            message: "This client does not support repository workspaces."
        )
    }

    func removeWorkspace(id: String) async throws {
        throw MachBoostAPIError.server(
            status: 501,
            message: "This client does not support repository workspaces."
        )
    }

    func teamStatus() async throws -> TeamStatus? { nil }
    func teamKeys() async throws -> [TeamKey] { [] }
    func traces(limit: Int) async throws -> [TraceSummary] { [] }
    func evaluations(limit: Int) async throws -> [TraceEvaluation] { [] }

    func createTeamKey(
        name: String,
        scopes: [String],
        allowedModels: [String],
        maxConcurrent: Int,
        requestsPerMinute: Int
    ) async throws -> CreatedTeamKeyResponse {
        throw MachBoostAPIError.server(status: 501, message: "Team mode is unavailable.")
    }

    func revokeTeamKey(id: String) async throws {
        throw MachBoostAPIError.server(status: 501, message: "Team mode is unavailable.")
    }

    func updateTeamSettings(
        traceMode: String,
        retentionDays: Int?,
        maxStorageBytes: Int64
    ) async throws -> TeamSettings {
        throw MachBoostAPIError.server(status: 501, message: "Team mode is unavailable.")
    }

    func evaluate(
        traceIDs: [String],
        name: String,
        model: String?
    ) async throws -> TraceEvaluation {
        throw MachBoostAPIError.server(status: 501, message: "Team mode is unavailable.")
    }
}

public final class MachBoostAPI: MachBoostAPIProtocol, @unchecked Sendable {
    private let endpoint: URL
    private let apiToken: String?
    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    public init(endpoint: URL, apiToken: String? = nil, session: URLSession? = nil) {
        self.endpoint = endpoint
        self.apiToken = apiToken
        if let session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.default
            configuration.timeoutIntervalForRequest = 3_600
            configuration.timeoutIntervalForResource = 86_400
            configuration.waitsForConnectivity = false
            self.session = URLSession(configuration: configuration)
        }
        self.encoder = JSONEncoder()
        self.decoder = JSONDecoder()
    }

    public func health(timeoutInterval: TimeInterval = 1) async throws -> Bool {
        struct Health: Decodable { let status: String }
        var request = try request(
            path: "/healthz",
            method: "GET",
            authenticated: false
        )
        request.timeoutInterval = timeoutInterval
        let health: Health = try await perform(request)
        return health.status == "ok"
    }

    public func catalog() async throws -> [CatalogModel] {
        let response: CatalogResponse = try await get("/api/catalog")
        return response.models
    }

    public func metrics() async throws -> ServerMetrics {
        try await get("/api/metrics")
    }

    public func models() async throws -> [ModelInstance] {
        let response: ModelsResponse = try await get("/api/ps")
        return response.models
    }

    public func workspaces() async throws -> [WorkspaceSummary] {
        let response: WorkspacesResponse = try await get("/api/workspaces")
        return response.workspaces
    }

    public func registerWorkspace(
        path: String,
        name: String? = nil
    ) async throws -> WorkspaceSummary {
        var payload: [String: Any] = ["path": path, "index": true]
        if let name, !name.isEmpty {
            payload["name"] = name
        }
        let response: WorkspaceIndexResponse = try await post(
            "/api/workspaces",
            jsonObject: payload
        )
        return response.workspace
    }

    public func reindexWorkspace(id: String) async throws -> WorkspaceSummary {
        let response: WorkspaceIndexResponse = try await post(
            "/api/workspaces/index",
            jsonObject: ["workspace_id": id]
        )
        return response.workspace
    }

    public func removeWorkspace(id: String) async throws {
        struct RemoveResponse: Decodable { let removed: Bool }
        let response: RemoveResponse = try await post(
            "/api/workspaces/delete",
            jsonObject: ["workspace_id": id]
        )
        guard response.removed else {
            throw MachBoostAPIError.invalidResponse
        }
    }

    public func teamStatus() async throws -> TeamStatus? {
        try await get("/api/team/status")
    }

    public func teamKeys() async throws -> [TeamKey] {
        let response: TeamKeysResponse = try await get("/api/team/keys")
        return response.keys
    }

    public func createTeamKey(
        name: String,
        scopes: [String],
        allowedModels: [String],
        maxConcurrent: Int,
        requestsPerMinute: Int
    ) async throws -> CreatedTeamKeyResponse {
        try await post(
            "/api/team/keys",
            jsonObject: [
                "name": name,
                "scopes": scopes,
                "allowed_models": allowedModels,
                "max_concurrent": maxConcurrent,
                "requests_per_minute": requestsPerMinute,
            ]
        )
    }

    public func revokeTeamKey(id: String) async throws {
        struct RevokeResponse: Decodable { let revoked: Bool }
        let response: RevokeResponse = try await post(
            "/api/team/keys/revoke",
            jsonObject: ["key_id": id]
        )
        guard response.revoked else { throw MachBoostAPIError.invalidResponse }
    }

    public func updateTeamSettings(
        traceMode: String,
        retentionDays: Int?,
        maxStorageBytes: Int64
    ) async throws -> TeamSettings {
        struct SettingsResponse: Decodable { let settings: TeamSettings }
        let response: SettingsResponse = try await post(
            "/api/team/settings",
            jsonObject: [
                "trace_mode": traceMode,
                "retention_days": retentionDays.map { $0 as Any } ?? NSNull(),
                "max_storage_bytes": maxStorageBytes,
            ]
        )
        return response.settings
    }

    public func traces(limit: Int = 100) async throws -> [TraceSummary] {
        let response: TracesResponse = try await get("/api/traces?limit=\(limit)")
        return response.traces
    }

    public func evaluations(limit: Int = 50) async throws -> [TraceEvaluation] {
        let response: EvaluationsResponse = try await get("/api/evaluations?limit=\(limit)")
        return response.evaluations
    }

    public func evaluate(
        traceIDs: [String],
        name: String,
        model: String? = nil
    ) async throws -> TraceEvaluation {
        var payload: [String: Any] = ["trace_ids": traceIDs, "name": name]
        if let model, !model.isEmpty { payload["model"] = model }
        let response: EvaluationResponse = try await post(
            "/api/evaluations",
            jsonObject: payload
        )
        return response.evaluation
    }

    public func preflight(model: String) async throws -> ModelPreflightResponse.Preflight {
        let payload: [String: Any] = [
            "model": model,
            "backend": "auto",
            "preflight": true,
            "allow_network": true,
        ]
        let response: ModelPreflightResponse = try await post(
            "/api/show",
            jsonObject: payload
        )
        return response.preflight
    }

    public func load(
        model: String,
        keepAlive: String = "forever",
        warmup: Bool = true
    ) async throws -> ModelLoadResponse {
        try await post(
            "/api/load",
            jsonObject: [
                "model": model,
                "keep_alive": keepAlive,
                "warmup": warmup,
                "options": ["backend": "auto"],
            ]
        )
    }

    public func stop(model: String? = nil) async throws {
        let payload: [String: Any] = model.map { ["model": $0] } ?? [:]
        let _: EmptyResponse = try await post("/api/stop", jsonObject: payload)
    }

    public func shutdown() async throws {
        let _: EmptyResponse = try await post("/api/shutdown", jsonObject: [:])
    }

    public func cancel(requestID: String) async throws -> Bool {
        struct CancelResponse: Decodable { let cancelled: Bool }
        do {
            let response: CancelResponse = try await post(
                "/api/cancel",
                jsonObject: ["request_id": requestID]
            )
            return response.cancelled
        } catch MachBoostAPIError.server(status: 404, message: _) {
            return false
        }
    }

    public func streamChat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        stream(path: "/api/chat", body: request, event: ChatEvent.self)
    }

    public func streamPull(
        model: String,
        requestID: String
    ) -> AsyncThrowingStream<PullEvent, Error> {
        struct PullRequest: Encodable, Sendable {
            let model: String
            let requestID: String
            let stream = true

            enum CodingKeys: String, CodingKey {
                case model
                case requestID = "request_id"
                case stream
            }
        }
        return stream(
            path: "/api/pull",
            body: PullRequest(model: model, requestID: requestID),
            event: PullEvent.self
        )
    }

    private func stream<
        RequestBody: Encodable & Sendable,
        Event: Decodable & Sendable
    >(
        path: String,
        body: RequestBody,
        event: Event.Type
    ) -> AsyncThrowingStream<Event, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = try self.request(path: path, method: "POST")
                    request.setValue("application/x-ndjson", forHTTPHeaderField: "Accept")
                    request.httpBody = try self.encoder.encode(body)
                    let (bytes, response) = try await self.session.bytes(for: request)
                    try await self.validate(response: response, bytes: bytes)
                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                            continue
                        }
                        let data = Data(line.utf8)
                        let decoded = try self.decoder.decode(event, from: data)
                        continuation.yield(decoded)
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

    private func get<Response: Decodable>(
        _ path: String,
        authenticated: Bool = true
    ) async throws -> Response {
        let request = try request(
            path: path,
            method: "GET",
            authenticated: authenticated
        )
        return try await perform(request)
    }

    private func post<Response: Decodable>(
        _ path: String,
        jsonObject: [String: Any]
    ) async throws -> Response {
        var request = try request(path: path, method: "POST")
        request.httpBody = try JSONSerialization.data(withJSONObject: jsonObject)
        return try await perform(request)
    }

    private func perform<Response: Decodable>(_ request: URLRequest) async throws -> Response {
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(Response.self, from: data)
    }

    private func request(
        path: String,
        method: String,
        authenticated: Bool = true
    ) throws -> URLRequest {
        guard let url = URL(string: path, relativeTo: endpoint) else {
            throw MachBoostAPIError.invalidResponse
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if authenticated, let apiToken, !apiToken.isEmpty {
            request.setValue("Bearer \(apiToken)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let response = response as? HTTPURLResponse else {
            throw MachBoostAPIError.invalidResponse
        }
        guard (200..<300).contains(response.statusCode) else {
            throw MachBoostAPIError.server(
                status: response.statusCode,
                message: serverMessage(from: data)
            )
        }
    }

    private func validate(
        response: URLResponse,
        bytes: URLSession.AsyncBytes
    ) async throws {
        guard let response = response as? HTTPURLResponse else {
            throw MachBoostAPIError.invalidResponse
        }
        guard (200..<300).contains(response.statusCode) else {
            var data = Data()
            for try await byte in bytes {
                data.append(byte)
            }
            throw MachBoostAPIError.server(
                status: response.statusCode,
                message: serverMessage(from: data)
            )
        }
    }

    private func serverMessage(from data: Data) -> String {
        if
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let message = object["error"] as? String
        {
            return message
        }
        return String(data: data, encoding: .utf8) ?? "Unknown error"
    }
}

private struct EmptyResponse: Decodable {}
