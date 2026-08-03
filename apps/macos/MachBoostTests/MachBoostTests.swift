import Foundation
import SwiftData
import XCTest
@testable import MachBoost

final class MachBoostTests: XCTestCase {
    func testCatalogSchemaDecodesDesktopFields() throws {
        let data = Data(
            """
            {
              "schema":"machboost.catalog.v1",
              "models":[{
                "name":"llama3.2:3b",
                "display_name":"Llama 3.2 3B",
                "repository":"mlx-community/Llama-3.2-3B-Instruct-4bit",
                "backend":"mlx",
                "capabilities":["chat","completion"],
                "cached":true,
                "cached_path":"/tmp/model",
                "recommended":true,
                "tested":true,
                "download_size_gb":2.0,
                "disk_size_gb":1.8,
                "minimum_memory_gb":8.0,
                "support":"ready",
                "support_reason":"compatible mlx architecture (llama)"
              }]
            }
            """.utf8
        )

        let response = try JSONDecoder().decode(CatalogResponse.self, from: data)

        XCTAssertEqual(response.schema, "machboost.catalog.v1")
        XCTAssertEqual(response.models.first?.name, "llama3.2:3b")
        XCTAssertEqual(response.models.first?.downloadSizeGB, 2.0)
        XCTAssertEqual(response.models.first?.diskSizeGB, 1.8)
        XCTAssertEqual(
            response.models.first?.supportReason,
            "compatible mlx architecture (llama)"
        )
        XCTAssertFalse(response.models.first?.supportsVision ?? true)
    }

    func testChatRequestUsesBackwardCompatibleWireKeys() throws {
        let request = ChatRequest(
            requestID: "chat-123",
            model: "llama3.2:3b",
            messages: [.init(role: "user", content: "Hello", images: nil)],
            context: ["/tmp/context.txt"],
            options: .init(maxTokens: 64, temperature: 0.2, affinityKey: "thread-1"),
            workspaceID: "workspace-123"
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        let options = try XCTUnwrap(object["options"] as? [String: Any])

        XCTAssertEqual(object["request_id"] as? String, "chat-123")
        XCTAssertEqual(object["keep_alive"] as? String, "forever")
        XCTAssertEqual(object["workspace_id"] as? String, "workspace-123")
        XCTAssertEqual(options["num_predict"] as? Int, 64)
        XCTAssertEqual(options["affinity_key"] as? String, "thread-1")
    }

    func testWorkspaceSchemaDecodesRepositoryMetadata() throws {
        let data = Data(
            """
            {
              "id":"0123456789abcdef",
              "name":"MachBoost",
              "path":"/tmp/machboost",
              "created_at":"2026-07-29T12:00:00Z",
              "updated_at":"2026-07-29T12:01:00Z",
              "indexed_at":"2026-07-29T12:01:00Z",
              "revision":"abc123",
              "file_count":181,
              "chunk_count":944,
              "total_bytes":1048576,
              "languages":[{"name":"Python","files":32}]
            }
            """.utf8
        )

        let workspace = try JSONDecoder().decode(WorkspaceSummary.self, from: data)

        XCTAssertEqual(workspace.name, "MachBoost")
        XCTAssertEqual(workspace.revision, "abc123")
        XCTAssertEqual(workspace.fileCount, 181)
        XCTAssertEqual(workspace.languages.first?.name, "Python")
    }

    @MainActor
    func testUpdaterStaysDisabledWithoutPublicKey() {
        let updates = UpdateController(
            startingUpdater: true,
            publicKey: "   "
        )

        XCTAssertFalse(updates.isAvailable)
        XCTAssertFalse(updates.automaticallyChecksForUpdates)
    }

    @MainActor
    func testConversationMessagesPersistInOrder() throws {
        let configuration = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(
            for: Conversation.self,
            ChatMessage.self,
            ChatAttachment.self,
            configurations: configuration
        )
        let context = container.mainContext
        let conversation = Conversation()
        context.insert(conversation)
        conversation.messages.append(
            ChatMessage(
                role: .assistant,
                content: "Second",
                createdAt: Date(timeIntervalSince1970: 2),
                conversation: conversation
            )
        )
        conversation.messages.append(
            ChatMessage(
                role: .user,
                content: "First",
                createdAt: Date(timeIntervalSince1970: 1),
                conversation: conversation
            )
        )
        try context.save()

        XCTAssertEqual(conversation.orderedMessages.map(\.content), ["First", "Second"])
    }

    @MainActor
    func testConversationPersistsSelectedWorkspace() throws {
        let configuration = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(
            for: Conversation.self,
            ChatMessage.self,
            ChatAttachment.self,
            configurations: configuration
        )
        let conversation = Conversation(workspaceID: "0123456789abcdef")
        container.mainContext.insert(conversation)
        try container.mainContext.save()

        XCTAssertEqual(conversation.workspaceID, "0123456789abcdef")
    }

    @MainActor
    func testAttachmentImporterCopiesUTF8Context() throws {
        let configuration = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(
            for: Conversation.self,
            ChatMessage.self,
            ChatAttachment.self,
            configurations: configuration
        )
        let conversation = Conversation()
        container.mainContext.insert(conversation)
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-test-\(UUID().uuidString).txt")
        try Data("local context".utf8).write(to: temporary)
        defer { try? FileManager.default.removeItem(at: temporary) }

        let attachments = try AttachmentStore.importURLs(
            [temporary],
            conversation: conversation
        )

        XCTAssertEqual(attachments.count, 1)
        XCTAssertEqual(attachments[0].kind, .text)
        XCTAssertTrue(FileManager.default.fileExists(atPath: attachments[0].importedPath))
        AttachmentStore.remove(attachments[0])
    }

    @MainActor
    func testAttachmentCopiesAreConversationScopedAndDeduplicated() throws {
        let configuration = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(
            for: Conversation.self,
            ChatMessage.self,
            ChatAttachment.self,
            configurations: configuration
        )
        let firstConversation = Conversation()
        let secondConversation = Conversation()
        container.mainContext.insert(firstConversation)
        container.mainContext.insert(secondConversation)
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-shared-\(UUID().uuidString).txt")
        try Data("shared context".utf8).write(to: temporary)
        defer { try? FileManager.default.removeItem(at: temporary) }

        let first = try XCTUnwrap(
            AttachmentStore.importURLs([temporary], conversation: firstConversation).first
        )
        firstConversation.attachments.append(first)
        let duplicate = try AttachmentStore.importURLs(
            [temporary],
            conversation: firstConversation
        )
        let second = try XCTUnwrap(
            AttachmentStore.importURLs([temporary], conversation: secondConversation).first
        )

        XCTAssertTrue(duplicate.isEmpty)
        XCTAssertNotEqual(first.importedPath, second.importedPath)
        AttachmentStore.remove(first)
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.importedPath))
        AttachmentStore.remove(second)
    }

    func testServerConfigurationUsesLoopbackUntilLANIsEnabled() {
        var configuration = ServerConfiguration()

        XCTAssertEqual(configuration.bindHost, "127.0.0.1")
        configuration.lanEnabled = true
        XCTAssertEqual(configuration.bindHost, "0.0.0.0")
        XCTAssertEqual(configuration.endpoint.host, "127.0.0.1")
        XCTAssertEqual(configuration.advertisedEndpoint.host, ProcessInfo.processInfo.hostName)
    }

    func testAuthenticatedCatalogRequestUsesBearerToken() async throws {
        let session = mockSession { request in
            XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer secret-token")
            return self.response(
                for: request,
                body: #"{"schema":"machboost.catalog.v1","models":[]}"#
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            apiToken: "secret-token",
            session: session
        )

        let models = try await api.catalog()

        XCTAssertTrue(models.isEmpty)
    }

    func testTeamStatusDecodesPrivacyAndRetentionPolicy() throws {
        let data = Data(
            """
            {
              "schema":"machboost.team-status.v1",
              "keys":3,
              "traces":42,
              "evaluations":2,
              "settings":{
                "trace_mode":"redacted",
                "retention_days":30,
                "max_storage_bytes":536870912
              }
            }
            """.utf8
        )

        let status = try JSONDecoder().decode(TeamStatus.self, from: data)

        XCTAssertEqual(status.keys, 3)
        XCTAssertEqual(status.settings.traceMode, "redacted")
        XCTAssertEqual(status.settings.retentionDays, 30)
        XCTAssertEqual(status.settings.maxStorageBytes, 536_870_912)
    }

    func testCreateTeamKeySendsLimitsAndReturnsOneTimeToken() async throws {
        let session = mockSession { request in
            let data = try XCTUnwrap(request.httpBody)
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: data) as? [String: Any]
            )
            XCTAssertEqual(object["name"] as? String, "Alice")
            XCTAssertEqual(object["max_concurrent"] as? Int, 3)
            XCTAssertEqual(object["requests_per_minute"] as? Int, 120)
            return self.response(
                for: request,
                status: 201,
                body: """
                {
                  "schema":"machboost.team-key.v1",
                  "token":"mbk_once",
                  "key":{
                    "id":"key_1","name":"Alice","kind":"key",
                    "scopes":["inference"],"allowed_models":["llama3.2:3b"],
                    "max_concurrent":3,"requests_per_minute":120,
                    "created_at":"2026-08-03T12:00:00Z"
                  }
                }
                """
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            session: session
        )

        let created = try await api.createTeamKey(
            name: "Alice",
            scopes: ["inference"],
            allowedModels: ["llama3.2:3b"],
            maxConcurrent: 3,
            requestsPerMinute: 120
        )

        XCTAssertEqual(created.token, "mbk_once")
        XCTAssertEqual(created.key.allowedModels, ["llama3.2:3b"])
    }

    func testHealthProbeUsesItsOwnShortTimeout() async throws {
        let session = mockSession { request in
            XCTAssertEqual(request.timeoutInterval, 0.25, accuracy: 0.001)
            return self.response(
                for: request,
                body: #"{"status":"ok"}"#
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            session: session
        )

        let healthy = try await api.health(timeoutInterval: 0.25)

        XCTAssertTrue(healthy)
    }

    func testCancellationSendsClientRequestID() async throws {
        let session = mockSession { request in
            let data = try XCTUnwrap(request.httpBody)
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: data) as? [String: Any]
            )
            XCTAssertEqual(object["request_id"] as? String, "chat-request-42")
            return self.response(
                for: request,
                status: 202,
                body: #"{"cancelled":true}"#
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            session: session
        )

        let cancelled = try await api.cancel(requestID: "chat-request-42")

        XCTAssertTrue(cancelled)
    }

    func testNDJSONChatStreamPreservesRequestIDAndCompletion() async throws {
        let session = mockSession { request in
            self.response(
                for: request,
                contentType: "application/x-ndjson",
                body: """
                {"request_id":"chat-stream-7","message":{"role":"assistant","content":"Hi"},"done":false}
                {"request_id":"chat-stream-7","message":{"role":"assistant","content":" there"},"done":true,"done_reason":"stop"}

                """
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            session: session
        )
        let request = ChatRequest(
            requestID: "chat-stream-7",
            model: "llama3.2:3b",
            messages: [.init(role: "user", content: "Hello", images: nil)],
            context: [],
            options: .init(maxTokens: 32, temperature: 0, affinityKey: nil)
        )
        var events: [ChatEvent] = []

        for try await event in api.streamChat(request) {
            events.append(event)
        }

        XCTAssertEqual(events.count, 2)
        XCTAssertEqual(events.map(\.requestID), ["chat-stream-7", "chat-stream-7"])
        XCTAssertEqual(events.compactMap(\.message?.content).joined(), "Hi there")
        XCTAssertTrue(events.last?.done ?? false)
    }

    @MainActor
    func testConversationMarkdownExportIsOrderedAndSanitized() throws {
        let conversation = Conversation(title: "Release: notes/July", model: "qwen2.5:3b")
        conversation.messages = [
            ChatMessage(
                role: .assistant,
                content: "Ready.",
                createdAt: Date(timeIntervalSince1970: 2),
                conversation: conversation
            ),
            ChatMessage(
                role: .user,
                content: "Ship it?",
                createdAt: Date(timeIntervalSince1970: 1),
                conversation: conversation
            ),
        ]

        let markdown = ConversationExporter.markdown(conversation)

        XCTAssertEqual(ConversationExporter.fileName(for: conversation), "Release- notes-July.md")
        XCTAssertLessThan(
            try XCTUnwrap(markdown.range(of: "## User")?.lowerBound),
            try XCTUnwrap(markdown.range(of: "## Assistant")?.lowerBound)
        )
        XCTAssertTrue(markdown.contains("Model: `qwen2.5:3b`"))
    }

    @MainActor
    func testDaemonStartsAndShutsDownFromSourceRuntime() async throws {
        let manager = DaemonManager()
        var configuration = ServerConfiguration()
        configuration.port = 19_435
        do {
            try await manager.start(configuration: configuration, apiToken: nil)
            XCTAssertEqual(manager.state, .running)
            XCTAssertTrue(manager.ownsProcess)
            await manager.shutdown(endpoint: configuration.endpoint, apiToken: nil)
            XCTAssertEqual(manager.state, .stopped)
            XCTAssertFalse(manager.ownsProcess)
        } catch {
            await manager.shutdown(endpoint: configuration.endpoint, apiToken: nil)
            throw error
        }
    }

    private func mockSession(
        handler: @escaping (URLRequest) throws -> (HTTPURLResponse, Data)
    ) -> URLSession {
        MockURLProtocol.handler = handler
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    private func response(
        for request: URLRequest,
        status: Int = 200,
        contentType: String = "application/json",
        body: String
    ) -> (HTTPURLResponse, Data) {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": contentType]
        )!
        return (response, Data(body.utf8))
    }
}

private final class MockURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler: ((URLRequest) throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        do {
            let handler = try XCTUnwrap(Self.handler)
            let (response, data) = try handler(requestWithMaterializedBody())
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}

    private func requestWithMaterializedBody() -> URLRequest {
        guard request.httpBody == nil, let stream = request.httpBodyStream else {
            return request
        }

        stream.open()
        defer { stream.close() }

        var body = Data()
        var buffer = [UInt8](repeating: 0, count: 4_096)
        while true {
            let count = stream.read(&buffer, maxLength: buffer.count)
            guard count > 0 else { break }
            body.append(contentsOf: buffer.prefix(count))
        }

        var materialized = request
        materialized.httpBodyStream = nil
        materialized.httpBody = body
        return materialized
    }
}
