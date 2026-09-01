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
    func teamConnect() async throws -> TeamConnectResponse
    func reportTeamPresence(
        deviceID: String,
        deviceName: String,
        appVersion: String,
        workspaceName: String?,
        workspaceFingerprint: String?,
        model: String?
    ) async throws -> TeamClient
    func teamClients() async throws -> [TeamClient]
    func teamModelRequests(status: String?) async throws -> [TeamModelRequest]
    func requestTeamModel(
        model: String,
        deviceID: String,
        note: String?
    ) async throws -> TeamModelRequest
    func resolveTeamModelRequest(
        id: String,
        status: String,
        note: String?
    ) async throws -> TeamModelRequest
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
    func memories(workspaceID: String?) async throws -> [MemorySummary]
    func cacheMetrics() async throws -> CacheMetrics
    func deleteMemory(id: String) async throws
    func providers() async throws -> [ProviderSummary]
    func extensions() async throws -> ExtensionsResponse
    func configureMCPServer(
        id: String?,
        name: String,
        transport: String,
        url: String?,
        command: String?,
        args: [String],
        environment: [String: String],
        headers: [String: String],
        enabled: Bool
    ) async throws -> MCPServerSummary
    func deleteMCPServer(id: String) async throws
    func testMCPServer(id: String) async throws -> [MCPToolSummary]
    func searchMCPTools(query: String, limit: Int) async throws -> [MCPToolSummary]
    func callMCPTool(
        serverID: String,
        name: String,
        arguments: JSONValue
    ) async throws -> MCPToolResult
    func configureSkill(
        id: String?,
        name: String,
        instructions: String,
        enabled: Bool
    ) async throws -> SkillSummary
    func deleteSkill(id: String) async throws
    func configureProvider(
        id: String?,
        name: String,
        baseURL: String,
        models: [String],
        apiKey: String?,
        monthlyBudgetUSD: Double?
    ) async throws -> ProviderSummary
    func setProviderSecret(id: String, apiKey: String) async throws
    func deleteProvider(id: String) async throws
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
    func teamConnect() async throws -> TeamConnectResponse {
        throw MachBoostAPIError.server(status: 501, message: "Team connections are unavailable.")
    }
    func reportTeamPresence(
        deviceID: String,
        deviceName: String,
        appVersion: String,
        workspaceName: String?,
        workspaceFingerprint: String?,
        model: String?
    ) async throws -> TeamClient {
        throw MachBoostAPIError.server(status: 501, message: "Team connections are unavailable.")
    }
    func teamClients() async throws -> [TeamClient] { [] }
    func teamModelRequests(status: String?) async throws -> [TeamModelRequest] { [] }
    func requestTeamModel(
        model: String,
        deviceID: String,
        note: String?
    ) async throws -> TeamModelRequest {
        throw MachBoostAPIError.server(status: 501, message: "Team connections are unavailable.")
    }
    func resolveTeamModelRequest(
        id: String,
        status: String,
        note: String?
    ) async throws -> TeamModelRequest {
        throw MachBoostAPIError.server(status: 501, message: "Team connections are unavailable.")
    }
    func traces(limit: Int) async throws -> [TraceSummary] { [] }
    func evaluations(limit: Int) async throws -> [TraceEvaluation] { [] }
    func memories(workspaceID: String?) async throws -> [MemorySummary] { [] }
    func cacheMetrics() async throws -> CacheMetrics {
        CacheMetrics(schema: "machboost.cache-metrics.v1", totals: [:], namespaces: [:])
    }
    func deleteMemory(id: String) async throws {}
    func providers() async throws -> [ProviderSummary] { [] }
    func extensions() async throws -> ExtensionsResponse {
        ExtensionsResponse(
            schema: "machboost.extensions.v1",
            mcpServers: [],
            skills: [],
            gatewayTools: []
        )
    }
    func configureMCPServer(
        id: String?,
        name: String,
        transport: String,
        url: String?,
        command: String?,
        args: [String],
        environment: [String: String],
        headers: [String: String],
        enabled: Bool
    ) async throws -> MCPServerSummary {
        throw MachBoostAPIError.server(status: 501, message: "MCP connectors are unavailable.")
    }
    func deleteMCPServer(id: String) async throws {}
    func testMCPServer(id: String) async throws -> [MCPToolSummary] { [] }
    func searchMCPTools(query: String, limit: Int) async throws -> [MCPToolSummary] { [] }
    func callMCPTool(
        serverID: String,
        name: String,
        arguments: JSONValue
    ) async throws -> MCPToolResult {
        throw MachBoostAPIError.server(status: 501, message: "MCP connectors are unavailable.")
    }
    func configureSkill(
        id: String?,
        name: String,
        instructions: String,
        enabled: Bool
    ) async throws -> SkillSummary {
        throw MachBoostAPIError.server(status: 501, message: "Reusable instructions are unavailable.")
    }
    func deleteSkill(id: String) async throws {}
    func configureProvider(
        id: String?,
        name: String,
        baseURL: String,
        models: [String],
        apiKey: String?,
        monthlyBudgetUSD: Double?
    ) async throws -> ProviderSummary {
        throw MachBoostAPIError.server(status: 501, message: "Provider routing is unavailable.")
    }
    func setProviderSecret(id: String, apiKey: String) async throws {}
    func deleteProvider(id: String) async throws {}

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

public struct ServerHealth: Decodable, Sendable, Equatable {
    public let status: String
    public let version: String?
    public let authentication: String?

    public var isReady: Bool { status == "ok" }
    public var requiresAuthentication: Bool { authentication == "required" }
}

public final class MachBoostAPI: MachBoostAPIProtocol, @unchecked Sendable {
    private let endpoint: URL
    private let apiToken: String?
    private let deviceID: String?
    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    public init(
        endpoint: URL,
        apiToken: String? = nil,
        deviceID: String? = nil,
        session: URLSession? = nil
    ) {
        self.endpoint = endpoint
        self.apiToken = apiToken
        self.deviceID = deviceID
        if let session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.default
            configuration.timeoutIntervalForRequest = 3_600
            configuration.timeoutIntervalForResource = 86_400
            configuration.waitsForConnectivity = false
            self.session = URLSession(configuration: configuration)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        self.encoder = encoder
        self.decoder = JSONDecoder()
    }

    public func health(timeoutInterval: TimeInterval = 1) async throws -> Bool {
        try await serverHealth(timeoutInterval: timeoutInterval).isReady
    }

    public func serverHealth(timeoutInterval: TimeInterval = 1) async throws -> ServerHealth {
        var request = try request(
            path: "/healthz",
            method: "GET",
            authenticated: false
        )
        request.timeoutInterval = timeoutInterval
        return try await perform(request)
    }

    public func serverVersion(timeoutInterval: TimeInterval = 1) async throws -> String {
        let health = try await serverHealth(timeoutInterval: timeoutInterval)
        guard health.isReady, let version = health.version else {
            throw MachBoostAPIError.invalidResponse
        }
        return version
    }

    public func authenticatedServerVersion(timeoutInterval: TimeInterval = 1) async throws -> String {
        struct Version: Decodable { let version: String }
        var request = try request(path: "/api/version", method: "GET")
        request.timeoutInterval = timeoutInterval
        let response: Version = try await perform(request)
        return response.version
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

    public func teamConnect() async throws -> TeamConnectResponse {
        try await get("/api/team/connect")
    }

    public func reportTeamPresence(
        deviceID: String,
        deviceName: String,
        appVersion: String,
        workspaceName: String? = nil,
        workspaceFingerprint: String? = nil,
        model: String? = nil
    ) async throws -> TeamClient {
        var payload: [String: Any] = [
            "device_id": deviceID,
            "device_name": deviceName,
            "app_version": appVersion,
            "mode": "connect",
        ]
        if let workspaceName, !workspaceName.isEmpty {
            payload["workspace_name"] = workspaceName
        }
        if let workspaceFingerprint, !workspaceFingerprint.isEmpty {
            payload["workspace_fingerprint"] = workspaceFingerprint
        }
        if let model, !model.isEmpty { payload["model"] = model }
        let response: TeamPresenceResponse = try await post(
            "/api/team/presence",
            jsonObject: payload
        )
        return response.client
    }

    public func teamClients() async throws -> [TeamClient] {
        let response: TeamClientsResponse = try await get("/api/team/clients")
        return response.clients
    }

    public func teamModelRequests(status: String? = nil) async throws -> [TeamModelRequest] {
        let suffix = status.map {
            "?status=" + $0.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed)!
        } ?? ""
        let response: TeamModelRequestsResponse = try await get(
            "/api/team/model-requests\(suffix)"
        )
        return response.requests
    }

    public func requestTeamModel(
        model: String,
        deviceID: String,
        note: String? = nil
    ) async throws -> TeamModelRequest {
        var payload: [String: Any] = ["model": model, "device_id": deviceID]
        if let note, !note.isEmpty { payload["note"] = note }
        let response: TeamModelRequestResponse = try await post(
            "/api/team/model-requests",
            jsonObject: payload
        )
        return response.request
    }

    public func resolveTeamModelRequest(
        id: String,
        status: String,
        note: String? = nil
    ) async throws -> TeamModelRequest {
        var payload: [String: Any] = ["request_id": id, "status": status]
        if let note, !note.isEmpty { payload["note"] = note }
        let response: TeamModelRequestResponse = try await post(
            "/api/team/model-requests/resolve",
            jsonObject: payload
        )
        return response.request
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

    public func memories(workspaceID: String? = nil) async throws -> [MemorySummary] {
        let suffix = workspaceID.map {
            "?workspace_id=" + $0.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed)!
        } ?? ""
        let response: MemoriesResponse = try await get("/api/memory\(suffix)")
        return response.memories
    }

    public func cacheMetrics() async throws -> CacheMetrics {
        try await get("/api/cache/metrics")
    }

    public func deleteMemory(id: String) async throws {
        struct DeleteResponse: Decodable { let removed: Int }
        let response: DeleteResponse = try await post(
            "/api/memory/delete",
            jsonObject: ["memory_ids": [id]]
        )
        guard response.removed == 1 else { throw MachBoostAPIError.invalidResponse }
    }

    public func providers() async throws -> [ProviderSummary] {
        let response: ProvidersResponse = try await get("/api/providers")
        return response.providers
    }

    public func extensions() async throws -> ExtensionsResponse {
        try await get("/api/extensions")
    }

    public func configureMCPServer(
        id: String?,
        name: String,
        transport: String,
        url: String?,
        command: String?,
        args: [String],
        environment: [String: String],
        headers: [String: String],
        enabled: Bool
    ) async throws -> MCPServerSummary {
        struct Response: Decodable { let server: MCPServerSummary }
        var payload: [String: Any] = [
            "name": name,
            "transport": transport,
            "args": args,
            "enabled": enabled,
        ]
        if let id { payload["id"] = id }
        if id == nil || !environment.isEmpty { payload["env"] = environment }
        if id == nil || !headers.isEmpty { payload["headers"] = headers }
        if let url { payload["url"] = url }
        if let command { payload["command"] = command }
        let response: Response = try await post("/api/mcp/servers", jsonObject: payload)
        return response.server
    }

    public func deleteMCPServer(id: String) async throws {
        struct Response: Decodable { let removed: Bool }
        let response: Response = try await post(
            "/api/mcp/servers/delete",
            jsonObject: ["server_id": id]
        )
        guard response.removed else { throw MachBoostAPIError.invalidResponse }
    }

    public func testMCPServer(id: String) async throws -> [MCPToolSummary] {
        struct Response: Decodable { let tools: [MCPToolSummary] }
        let response: Response = try await post(
            "/api/mcp/servers/test",
            jsonObject: ["server_id": id]
        )
        return response.tools
    }

    public func searchMCPTools(query: String, limit: Int = 8) async throws -> [MCPToolSummary] {
        struct Response: Decodable { let tools: [MCPToolSummary] }
        let response: Response = try await post(
            "/api/mcp/search",
            jsonObject: ["query": query, "limit": limit]
        )
        return response.tools
    }

    public func callMCPTool(
        serverID: String,
        name: String,
        arguments: JSONValue
    ) async throws -> MCPToolResult {
        struct Response: Decodable { let result: MCPToolResult }
        let response: Response = try await post(
            "/api/mcp/call",
            jsonObject: [
                "server_id": serverID,
                "name": name,
                "arguments": Self.foundationValue(arguments),
            ]
        )
        return response.result
    }

    public func configureSkill(
        id: String?,
        name: String,
        instructions: String,
        enabled: Bool
    ) async throws -> SkillSummary {
        struct Response: Decodable { let skill: SkillSummary }
        var payload: [String: Any] = [
            "name": name,
            "instructions": instructions,
            "enabled": enabled,
        ]
        if let id { payload["id"] = id }
        let response: Response = try await post("/api/skills", jsonObject: payload)
        return response.skill
    }

    public func deleteSkill(id: String) async throws {
        struct Response: Decodable { let removed: Bool }
        let response: Response = try await post(
            "/api/skills/delete",
            jsonObject: ["skill_id": id]
        )
        guard response.removed else { throw MachBoostAPIError.invalidResponse }
    }

    public func configureProvider(
        id: String? = nil,
        name: String,
        baseURL: String,
        models: [String],
        apiKey: String?,
        monthlyBudgetUSD: Double?
    ) async throws -> ProviderSummary {
        var payload: [String: Any] = [
            "name": name,
            "base_url": baseURL,
            "models": models,
            "enabled": true,
        ]
        if let id, !id.isEmpty { payload["id"] = id }
        if let apiKey, !apiKey.isEmpty { payload["api_key"] = apiKey }
        if let monthlyBudgetUSD { payload["monthly_budget_usd"] = monthlyBudgetUSD }
        let response: ProviderResponse = try await post(
            "/api/providers",
            jsonObject: payload
        )
        return response.provider
    }

    public func deleteProvider(id: String) async throws {
        struct DeleteResponse: Decodable { let removed: Bool }
        let response: DeleteResponse = try await post(
            "/api/providers/delete",
            jsonObject: ["provider_id": id]
        )
        guard response.removed else { throw MachBoostAPIError.invalidResponse }
    }

    public func setProviderSecret(id: String, apiKey: String) async throws {
        struct SecretResponse: Decodable {
            let providerID: String
            let hasSecret: Bool

            enum CodingKeys: String, CodingKey {
                case providerID = "provider_id"
                case hasSecret = "has_secret"
            }
        }
        let response: SecretResponse = try await post(
            "/api/providers/secret",
            jsonObject: ["provider_id": id, "api_key": apiKey]
        )
        guard response.providerID == id, response.hasSecret else {
            throw MachBoostAPIError.invalidResponse
        }
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
        stream(
            path: "/api/chat",
            body: request,
            event: ChatEvent.self,
            isTerminal: { $0.done || $0.error != nil }
        )
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
            event: PullEvent.self,
            isTerminal: { $0.done || $0.error != nil }
        )
    }

    private func stream<
        RequestBody: Encodable & Sendable,
        Event: Decodable & Sendable
    >(
        path: String,
        body: RequestBody,
        event: Event.Type,
        isTerminal: @escaping @Sendable (Event) -> Bool
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
                        if isTerminal(decoded) {
                            continuation.finish()
                            return
                        }
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
        if let deviceID, !deviceID.isEmpty {
            request.setValue(deviceID, forHTTPHeaderField: "X-MachBoost-Device-ID")
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

    private static func foundationValue(_ value: JSONValue) -> Any {
        switch value {
        case let .object(object):
            return object.mapValues(foundationValue)
        case let .array(array):
            return array.map(foundationValue)
        case let .string(string):
            return string
        case let .number(number):
            return number
        case let .boolean(boolean):
            return boolean
        case .null:
            return NSNull()
        }
    }
}

private struct EmptyResponse: Decodable {}
