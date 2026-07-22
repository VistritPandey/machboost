import Foundation

enum MachBoostAPIError: LocalizedError {
    case invalidResponse
    case server(status: Int, message: String)
    case stream(String)

    var errorDescription: String? {
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

final class MachBoostAPI: @unchecked Sendable {
    private let endpoint: URL
    private let apiToken: String?
    private let session: URLSession
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(endpoint: URL, apiToken: String? = nil, session: URLSession? = nil) {
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

    func health(timeoutInterval: TimeInterval = 1) async throws -> Bool {
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

    func catalog() async throws -> [CatalogModel] {
        let response: CatalogResponse = try await get("/api/catalog")
        return response.models
    }

    func metrics() async throws -> ServerMetrics {
        try await get("/api/metrics")
    }

    func models() async throws -> [ModelInstance] {
        let response: ModelsResponse = try await get("/api/ps")
        return response.models
    }

    func preflight(model: String) async throws -> ModelPreflightResponse.Preflight {
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

    func stop(model: String? = nil) async throws {
        let payload: [String: Any] = model.map { ["model": $0] } ?? [:]
        let _: EmptyResponse = try await post("/api/stop", jsonObject: payload)
    }

    func shutdown() async throws {
        let _: EmptyResponse = try await post("/api/shutdown", jsonObject: [:])
    }

    func cancel(requestID: String) async throws -> Bool {
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

    func streamChat(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        stream(path: "/api/chat", body: request, event: ChatEvent.self)
    }

    func streamPull(model: String, requestID: String) -> AsyncThrowingStream<PullEvent, Error> {
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
