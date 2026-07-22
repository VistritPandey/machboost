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
                "minimum_memory_gb":8.0,
                "support":"ready"
              }]
            }
            """.utf8
        )

        let response = try JSONDecoder().decode(CatalogResponse.self, from: data)

        XCTAssertEqual(response.schema, "machboost.catalog.v1")
        XCTAssertEqual(response.models.first?.name, "llama3.2:3b")
        XCTAssertEqual(response.models.first?.downloadSizeGB, 2.0)
        XCTAssertFalse(response.models.first?.supportsVision ?? true)
    }

    func testChatRequestUsesBackwardCompatibleWireKeys() throws {
        let request = ChatRequest(
            requestID: "chat-123",
            model: "llama3.2:3b",
            messages: [.init(role: "user", content: "Hello", images: nil)],
            context: ["/tmp/context.txt"],
            options: .init(maxTokens: 64, temperature: 0.2, affinityKey: "thread-1")
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        let options = try XCTUnwrap(object["options"] as? [String: Any])

        XCTAssertEqual(object["request_id"] as? String, "chat-123")
        XCTAssertEqual(object["keep_alive"] as? String, "forever")
        XCTAssertEqual(options["num_predict"] as? Int, 64)
        XCTAssertEqual(options["affinity_key"] as? String, "thread-1")
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
    }
}
