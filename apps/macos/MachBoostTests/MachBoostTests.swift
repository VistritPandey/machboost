import Foundation
import MachBoostDaemonClient
import SwiftData
import XCTest
@testable import MachBoost

final class MachBoostTests: XCTestCase {
    func testAssistantTimelinePreservesInterleavedReasoningContentAndTools() throws {
        let call = APIToolCall(function: .init(name: "read_file", arguments: .object([:])))
        let expected = [
            AssistantTimelineEntry(kind: .reasoning, text: "Inspecting."),
            AssistantTimelineEntry(kind: .content, text: "I will read it."),
            AssistantTimelineEntry(
                kind: .tools,
                activities: [CodingToolActivity(call: call, state: .succeeded)]
            ),
            AssistantTimelineEntry(kind: .reasoning, text: "Checking result."),
            AssistantTimelineEntry(kind: .content, text: "Done."),
        ]

        let decoded = try JSONDecoder().decode(
            [AssistantTimelineEntry].self,
            from: JSONEncoder().encode(expected)
        )

        XCTAssertEqual(decoded.map(\.kind), [.reasoning, .content, .tools, .reasoning, .content])
        XCTAssertEqual(decoded[2].activities.first?.call.function.name, "read_file")
    }

    func testHostRoutingPrefersResidentModelUntilQueuePressureRequiresSpillover() {
        let availableResident = serverMetrics(active: 1, queued: 2, p50: 0.2)
        let saturatedResident = serverMetrics(active: 1, queued: 20, p50: 0.2)
        let idleColdHost = serverMetrics(active: 0, queued: 0, p50: 0.1)

        XCTAssertLessThan(
            HostRoutingPolicy.score(metrics: availableResident, modelLoaded: true),
            HostRoutingPolicy.score(metrics: idleColdHost, modelLoaded: false)
        )
        XCTAssertGreaterThan(
            HostRoutingPolicy.score(metrics: saturatedResident, modelLoaded: true),
            HostRoutingPolicy.score(metrics: idleColdHost, modelLoaded: false)
        )
        XCTAssertGreaterThan(
            HostRoutingPolicy.score(
                metrics: availableResident,
                modelLoaded: true,
                reservedRequests: 2
            ),
            HostRoutingPolicy.score(metrics: idleColdHost, modelLoaded: false)
        )
    }

    func testTurnMetricsAggregateEveryToolRound() {
        var metrics = GenerationTurnMetrics()
        metrics.absorb(
            ChatEvent(
                requestID: "round-1",
                message: nil,
                done: true,
                doneReason: "tool_calls",
                totalDuration: 2_000_000_000,
                evalDuration: 1_000_000_000,
                evalCount: 20,
                machboost: .init(
                    backend: "mlx-vlm",
                    stats: nil,
                    timeToFirstTokenSeconds: 0.25
                ),
                error: nil
            )
        )
        metrics.absorb(
            ChatEvent(
                requestID: "round-2",
                message: nil,
                done: true,
                doneReason: "stop",
                totalDuration: 1_000_000_000,
                evalDuration: 500_000_000,
                evalCount: 10,
                machboost: nil,
                error: nil
            )
        )

        let message = ChatMessage(role: .assistant, content: "Done")
        message.wasCancelled = true
        metrics.apply(to: message)

        XCTAssertEqual(metrics.rounds, 2)
        XCTAssertEqual(message.generatedTokens, 30)
        XCTAssertEqual(message.tokensPerSecond ?? 0, 20, accuracy: 0.001)
        XCTAssertEqual(message.durationSeconds ?? 0, 3, accuracy: 0.001)
        XCTAssertEqual(message.timeToFirstTokenSeconds, 0.25)
        XCTAssertTrue(message.wasCancelled)
    }

    func testCodingPermissionModesApplyDistinctWritePolicies() {
        let smallEdit = APIToolCall(
            function: .init(
                name: "replace_in_file",
                arguments: .object([
                    "path": .string("Sources/App.swift"),
                    "old_text": .string("let old = true"),
                    "new_text": .string("let old = false"),
                ])
            )
        )
        let create = APIToolCall(
            function: .init(
                name: "create_file",
                arguments: .object([
                    "path": .string("Sources/New.swift"),
                    "content": .string("let value = 1"),
                ])
            )
        )

        XCTAssertEqual(
            CodingWorkspace.permissionDecision(for: smallEdit, mode: .automatic),
            .allow
        )
        XCTAssertEqual(
            CodingWorkspace.permissionDecision(for: create, mode: .automatic),
            .ask
        )
        XCTAssertEqual(
            CodingWorkspace.permissionDecision(for: smallEdit, mode: .manual),
            .ask
        )
        XCTAssertEqual(
            CodingWorkspace.permissionDecision(for: create, mode: .acceptEdits),
            .allow
        )
        XCTAssertEqual(CodingWorkspace.tools(for: .plan).count, 3)
        guard case .deny = CodingWorkspace.permissionDecision(for: create, mode: .plan) else {
            return XCTFail("Plan mode must deny writes")
        }
    }

    func testCodingWorkspaceRejectsMalformedLegacyToolNames() {
        let valid = APIToolCall(
            function: .init(name: "search_repository", arguments: .object([:]))
        )
        let malformed = APIToolCall(
            function: .init(name: "list_files<|message|", arguments: .object([:]))
        )

        XCTAssertTrue(CodingWorkspace.supports(valid))
        XCTAssertFalse(CodingWorkspace.supports(malformed))
    }

    func testAssistantProtocolSanitizerHidesRecipientAndToolMarkup() {
        let visible = CodingWorkspace.visibleAssistantText(
            "<|start|>assistant to=user<|message|>to=user\n"
                + "<atem:function_calls>\n"
                + "<atem:invoke name=\"read_file\">\n"
                + "<atem:parameter name=\"path\">src/App.swift</atem:parameter>\n"
                + "</atem:invoke>\n"
                + "</atem:function_calls>\n"
                + "<tool_call name=\"search_files\">\n"
                + "{\"arguments\":{\"query\":\"hello\"}}\n"
                + "</tool_call>\n"
                + "Here is the result.<|eot|>"
        )

        XCTAssertEqual(visible, "Here is the result.")
    }

    func testWorkspaceChangesLoadsTrackedAndUntrackedDiffs() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-changes-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        func git(_ arguments: [String]) throws {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
            process.arguments = ["-C", root.path] + arguments
            process.standardOutput = Pipe()
            process.standardError = Pipe()
            try process.run()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus, 0, arguments.joined(separator: " "))
        }

        try git(["init", "-b", "main"])
        try git(["config", "user.email", "tests@machboost.local"])
        try git(["config", "user.name", "MachBoost Tests"])
        let tracked = root.appendingPathComponent("tracked.txt")
        try Data("before\n".utf8).write(to: tracked)
        try git(["add", "tracked.txt"])
        try git(["commit", "-m", "fixture"])
        try Data("after\nsecond\n".utf8).write(to: tracked)
        try Data("new\n".utf8).write(to: root.appendingPathComponent("new.txt"))

        let snapshot = WorkspaceChanges.load(workspaceRoot: root.path)

        XCTAssertNil(snapshot.error)
        XCTAssertEqual(snapshot.branch, "main")
        XCTAssertEqual(Set(snapshot.changes.map(\.path)), ["tracked.txt", "new.txt"])
        XCTAssertTrue(snapshot.changes.first { $0.path == "tracked.txt" }?.patch.contains("+after") == true)
        XCTAssertEqual(snapshot.changes.first { $0.path == "new.txt" }?.status, "Untracked")
    }

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

    func testMuseCatalogAndRequestExposeNativeCapabilities() throws {
        let data = Data(
            """
            {
              "schema":"machboost.catalog.v1",
              "models":[{
                "name":"muse-glimmer:30b",
                "display_name":"Muse Glimmer 30B",
                "repository":"mlx-community/Muse-Glimmer-30B-4bit",
                "source_repository":"meta-models/Muse-Glimmer-30B",
                "backend":"mlx-vlm",
                "capabilities":["chat","completion","vision","reasoning","tools"],
                "cached":true,
                "cached_path":"/tmp/model",
                "recommended":true,
                "tested":true,
                "download_size_gb":21.0,
                "disk_size_gb":6.0,
                "minimum_memory_gb":32.0,
                "context_length":131072,
                "support":"ready",
                "support_reason":"compatible"
              }]
            }
            """.utf8
        )

        let model = try XCTUnwrap(
            JSONDecoder().decode(CatalogResponse.self, from: data).models.first
        )

        XCTAssertTrue(model.supportsVision)
        XCTAssertTrue(model.supportsReasoning)
        XCTAssertTrue(model.supportsTools)
        XCTAssertEqual(model.contextLength, 131_072)
        XCTAssertEqual(model.sourceRepository, "meta-models/Muse-Glimmer-30B")
    }

    func testMuseChatRequestEncodesReasoningAndNativeTools() throws {
        let tool = MachBoostDaemonClient.APIToolDefinition(
            function: .init(
                name: "search_repository",
                description: "Search the active repository.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object(["type": .string("string")])
                    ]),
                    "required": .array([.string("query")]),
                ])
            )
        )
        let request = ChatRequest(
            requestID: "muse-chat-1",
            model: "muse-glimmer:30b",
            messages: [.init(role: "user", content: "Find cancellation.")],
            context: [],
            options: .init(maxTokens: 256, temperature: 1, affinityKey: "repo-1"),
            workspaceID: "repo-1",
            reasoningStrength: "high",
            tools: [tool]
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        let tools = try XCTUnwrap(object["tools"] as? [[String: Any]])
        let firstTool = try XCTUnwrap(tools.first)
        let function = try XCTUnwrap(firstTool["function"] as? [String: Any])

        XCTAssertEqual(object["think"] as? String, "high")
        XCTAssertEqual(function["name"] as? String, "search_repository")
        XCTAssertEqual(firstTool["type"] as? String, "function")
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

    func testConversationCompactionKeepsRecentCompletedTurns() {
        let conversation = Conversation(model: "qwen2.5:3b")
        let messages = (0..<12).map { index in
            ChatMessage(
                role: index.isMultiple(of: 2) ? .user : .assistant,
                content: "message-\(index)",
                createdAt: Date(timeIntervalSince1970: Double(index)),
                conversation: conversation
            )
        }
        conversation.messages = messages

        let candidates = ConversationCompaction.candidates(
            messages: conversation.orderedMessages,
            keepRecent: 8
        )

        XCTAssertEqual(candidates.map(\.content), ["message-0", "message-1", "message-2", "message-3"])
        XCTAssertEqual(candidates.last?.createdAt, Date(timeIntervalSince1970: 3))
    }

    func testConversationCompactionSkipsEmptyAndCancelledMessages() {
        let completed = ChatMessage(role: .user, content: "keep")
        let empty = ChatMessage(role: .assistant, content: "")
        let cancelled = ChatMessage(role: .assistant, content: "cancelled")
        cancelled.wasCancelled = true

        let candidates = ConversationCompaction.candidates(
            messages: [completed, empty, cancelled],
            keepRecent: 0
        )

        XCTAssertEqual(candidates.map(\.content), ["keep"])
        XCTAssertGreaterThan(
            ConversationCompaction.estimatedTokens(
                summary: "prior summary",
                messages: [completed]
            ),
            0
        )
    }

    func testConversationCompactionEstimatesConservatively() {
        let conversation = Conversation()
        let message = ChatMessage(
            role: .user,
            content: String(repeating: "a", count: 276),
            conversation: conversation
        )

        XCTAssertEqual(
            ConversationCompaction.estimatedTokens(
                summary: String(repeating: "b", count: 24),
                messages: [message]
            ),
            108
        )
    }

    func testConversationCompactionRepairsInvalidStoredSettings() {
        XCTAssertEqual(ConversationCompaction.clampedMaxTokens(0), 32)
        XCTAssertEqual(ConversationCompaction.clampedMaxTokens(8_192), 4_096)
        XCTAssertEqual(ConversationCompaction.clampedThreshold(0), 70)
        XCTAssertEqual(ConversationCompaction.clampedThreshold(100), 95)
        XCTAssertEqual(ConversationCompaction.clampedThreshold(90), 90)
    }

    func testConversationCompactionTriggersAtConfiguredContextThreshold() {
        XCTAssertFalse(
            ConversationCompaction.shouldCompact(
                estimatedTokens: 809,
                contextLength: 1_000,
                reservedOutputTokens: 100,
                thresholdPercent: 90
            )
        )
        XCTAssertTrue(
            ConversationCompaction.shouldCompact(
                estimatedTokens: 810,
                contextLength: 1_000,
                reservedOutputTokens: 100,
                thresholdPercent: 90
            )
        )
    }

    @MainActor
    func testCommunityUpdaterRejectsPlaceholderKeyAndDefaultsToReleaseChecks() {
        let releasesURL = URL(string: "https://example.com/releases/latest")!
        let defaultsName = "MachBoostTests.community-updates.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: defaultsName)!
        defer { defaults.removePersistentDomain(forName: defaultsName) }
        var openedURL: URL?
        let updates = UpdateController(
            startingUpdater: false,
            publicKey: "$(SPARKLE_PUBLIC_ED_KEY)",
            releasesURL: releasesURL,
            openRelease: { openedURL = $0 },
            defaults: defaults
        )

        XCTAssertTrue(updates.isAvailable)
        XCTAssertTrue(updates.supportsAutomaticUpdates)
        XCTAssertTrue(updates.automaticallyChecksForUpdates)
        XCTAssertEqual(updates.actionTitle, "Check Now")
        XCTAssertEqual(
            updates.deliveryDescription,
            "Checks GitHub Releases; community installation is manual"
        )

        updates.automaticallyChecksForUpdates = false
        XCTAssertFalse(updates.automaticallyChecksForUpdates)

        updates.downloadUpdate()
        XCTAssertNil(openedURL)
    }

    @MainActor
    func testCommunityUpdaterSurfacesNewGitHubRelease() async {
        let defaultsName = "MachBoostTests.community-release.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: defaultsName)!
        defer { defaults.removePersistentDomain(forName: defaultsName) }
        let updates = UpdateController(
            startingUpdater: false,
            publicKey: "$(SPARKLE_PUBLIC_ED_KEY)",
            defaults: defaults,
            currentVersion: "0.11.0",
            fetchLatestRelease: { _ in "v0.12.0" }
        )

        await updates.checkCommunityRelease()

        XCTAssertTrue(updates.communityCheckCompleted)
        XCTAssertFalse(updates.communityCheckFailed)
        XCTAssertEqual(updates.latestCommunityVersion, "v0.12.0")
        XCTAssertTrue(updates.updateAvailable)
        XCTAssertTrue(updates.canDownloadUpdate)
        XCTAssertEqual(updates.downloadTitle, "Download v0.12.0")
        XCTAssertNotNil(updates.lastCheckedAt)
        XCTAssertEqual(
            updates.deliveryDescription,
            "v0.12.0 is available on GitHub; installation is manual"
        )
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
    func testConversationPersistsMuseReasoningAndToolCalls() throws {
        let configuration = ModelConfiguration(isStoredInMemoryOnly: true)
        let container = try ModelContainer(
            for: Conversation.self,
            ChatMessage.self,
            ChatAttachment.self,
            configurations: configuration
        )
        let conversation = Conversation(model: "muse-glimmer:30b")
        var activity = CodingToolActivity(
            call: APIToolCall(
                id: "search-1",
                type: "function",
                function: .init(
                    name: "search_code",
                    arguments: .object(["query": .string("repository")])
                )
            )
        )
        activity.state = .succeeded
        activity.output = #"{"matches":[]}"#
        let activityJSON = String(
            decoding: try JSONEncoder().encode([activity]),
            as: UTF8.self
        )
        let message = ChatMessage(
            role: .assistant,
            content: "Checking now.",
            reasoningContent: "I should search the repository.",
            toolCallsJSON: "[{\"function\":{\"name\":\"search_repository\"}}]",
            toolActivityJSON: activityJSON,
            conversation: conversation
        )
        conversation.messages.append(message)
        container.mainContext.insert(conversation)
        try container.mainContext.save()

        let stored = try XCTUnwrap(conversation.orderedMessages.first)
        XCTAssertEqual(stored.reasoningContent, "I should search the repository.")
        XCTAssertTrue(stored.toolCallsJSON?.contains("search_repository") ?? false)
        let storedActivities = try JSONDecoder().decode(
            [CodingToolActivity].self,
            from: Data(try XCTUnwrap(stored.toolActivityJSON).utf8)
        )
        XCTAssertEqual(storedActivities.first?.state, .succeeded)
        XCTAssertEqual(storedActivities.first?.call.function.name, "search_code")
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
        XCTAssertNotEqual(configuration.advertisedEndpoint.host, "127.0.0.1")
        XCTAssertFalse(configuration.advertisedEndpoint.host?.isEmpty ?? true)
    }

    func testLoadRequestWarmsAndKeepsModelResident() async throws {
        let session = mockSession { request in
            XCTAssertEqual(request.url?.path, "/api/load")
            let data = try XCTUnwrap(request.httpBody)
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: data) as? [String: Any]
            )
            XCTAssertEqual(object["model"] as? String, "qwen2.5:3b")
            XCTAssertEqual(object["keep_alive"] as? String, "1h")
            XCTAssertEqual(object["warmup"] as? Bool, true)
            let options = try XCTUnwrap(object["options"] as? [String: Any])
            XCTAssertEqual(options["backend"] as? String, "auto")
            return self.response(
                for: request,
                body: """
                {
                  "status":"success",
                  "model":"qwen2.5:3b",
                  "load_duration_seconds":1.25,
                  "warmup_duration_seconds":0.18,
                  "warmup_performed":true,
                  "instance":{
                    "model":"mlx-community/Qwen2.5-3B-Instruct-4bit",
                    "backend":"mlx",
                    "idle_seconds":0.0,
                    "keep_alive_seconds":3600.0,
                    "requests":0,
                    "capabilities":["chat","completion"],
                    "scheduler":{"replicas":1,"active_requests":0,"queued_requests":0}
                  }
                }
                """
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            session: session
        )

        let loaded = try await api.load(
            model: "qwen2.5:3b",
            keepAlive: "1h",
            warmup: true
        )

        XCTAssertTrue(loaded.warmupPerformed)
        XCTAssertEqual(loaded.instance.keepAliveSeconds, 3_600)
        XCTAssertEqual(loaded.instance.backend, "mlx")
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
              "online_clients":5,
              "pending_model_requests":1,
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
        XCTAssertEqual(status.onlineClients, 5)
        XCTAssertEqual(status.pendingModelRequests, 1)
        XCTAssertEqual(status.settings.traceMode, "redacted")
        XCTAssertEqual(status.settings.retentionDays, 30)
        XCTAssertEqual(status.settings.maxStorageBytes, 536_870_912)
    }

    func testTeamConnectUsesBearerAndStableDeviceIdentity() async throws {
        let session = mockSession { request in
            XCTAssertEqual(request.url?.path, "/api/team/connect")
            XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer mbk_team")
            XCTAssertEqual(request.value(forHTTPHeaderField: "X-MachBoost-Device-ID"), "device-42")
            return self.response(
                for: request,
                body: """
                {
                  "schema":"machboost.team-connect.v1",
                  "host":{"name":"Inference Mac","version":"0.13.0"},
                  "principal":{
                    "id":"key_1","name":"Alice","kind":"key",
                    "scopes":["inference","models:read"],"allowed_models":[],
                    "max_concurrent":2,"requests_per_minute":60
                  },
                  "models":[],"loaded_models":[],
                  "capabilities":["chat","tool_calls"]
                }
                """
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://192.168.1.20:11435")!,
            apiToken: "mbk_team",
            deviceID: "device-42",
            session: session
        )

        let connection = try await api.teamConnect()

        XCTAssertEqual(connection.host.name, "Inference Mac")
        XCTAssertEqual(connection.principal.name, "Alice")
        XCTAssertTrue(connection.capabilities.contains("tool_calls"))
    }

    func testTeamPresenceNeverSendsRepositoryPath() async throws {
        let session = mockSession { request in
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: try XCTUnwrap(request.httpBody))
                    as? [String: Any]
            )
            XCTAssertEqual(object["workspace_name"] as? String, "checkout-service")
            XCTAssertEqual(object["workspace_fingerprint"] as? String, "abc123")
            XCTAssertNil(object["workspace_path"])
            return self.response(
                for: request,
                body: """
                {
                  "schema":"machboost.team-presence.v1",
                  "client":{
                    "device_id":"device-42",
                    "principal":{"id":"key_1","name":"Alice"},
                    "device_name":"Alice's Mac","app_version":"0.13.0",
                    "mode":"connect","workspace_name":"checkout-service",
                    "workspace_fingerprint":"abc123","model":"coder",
                    "first_seen_at":"2026-08-15T12:00:00Z",
                    "last_seen_at":"2026-08-15T12:00:01Z",
                    "last_request_at":null,"request_count":0,"online":true
                  }
                }
                """
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://192.168.1.20:11435")!,
            apiToken: "mbk_team",
            deviceID: "device-42",
            session: session
        )

        let client = try await api.reportTeamPresence(
            deviceID: "device-42",
            deviceName: "Alice's Mac",
            appVersion: "0.13.0",
            workspaceName: "checkout-service",
            workspaceFingerprint: "abc123",
            model: "coder"
        )

        XCTAssertTrue(client.online)
        XCTAssertEqual(client.workspaceName, "checkout-service")
    }

    func testToolResultMessageEncodesCallID() throws {
        let message = MachBoostDaemonClient.APIChatMessage(
            role: "tool",
            content: #"{"files":["src/main.swift"]}"#,
            toolName: "list_files",
            toolCallID: "call-1"
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(message))
                as? [String: Any]
        )

        XCTAssertEqual(object["tool_name"] as? String, "list_files")
        XCTAssertEqual(object["tool_call_id"] as? String, "call-1")
    }

    func testCodingWorkspaceToolsAreBoundedToSelectedRepository() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-tools-\(UUID().uuidString)", isDirectory: true)
        let source = root.appendingPathComponent("Sources/App.swift")
        try FileManager.default.createDirectory(
            at: source.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try "let greeting = \"hello\"\n".write(to: source, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: root) }

        let search = APIToolCall(
            id: "search-1",
            type: "function",
            function: .init(
                name: "search_code",
                arguments: .object(["query": .string("greeting")])
            )
        )
        let searchResult = try CodingWorkspace.execute(search, workspaceRoot: root.path)
        XCTAssertTrue(searchResult.content.contains("Sources/App.swift"))

        let replace = APIToolCall(
            id: "replace-1",
            type: "function",
            function: .init(
                name: "replace_in_file",
                arguments: .object([
                    "path": .string("Sources/App.swift"),
                    "old_text": .string("hello"),
                    "new_text": .string("hello team"),
                ])
            )
        )
        XCTAssertTrue(CodingWorkspace.isMutating(replace))
        let replaceResult = try CodingWorkspace.execute(replace, workspaceRoot: root.path)
        XCTAssertTrue(try String(contentsOf: source, encoding: .utf8).contains("hello team"))
        XCTAssertEqual(replaceResult.changedPath, "Sources/App.swift")
        XCTAssertTrue(replaceResult.changePatch?.contains("-hello") ?? false)
        XCTAssertTrue(replaceResult.changePatch?.contains("+hello team") ?? false)
        XCTAssertEqual(
            CodingWorkspace.fileURL(
                relativePath: "Sources/App.swift",
                workspaceRoot: root.path
            ),
            source
        )

        let escape = APIToolCall(
            function: .init(
                name: "read_file",
                arguments: .object(["path": .string("../private.txt")])
            )
        )
        XCTAssertThrowsError(try CodingWorkspace.execute(escape, workspaceRoot: root.path))
    }

    func testCodingToolActivitiesRoundTripMultipleCallsAndFormatResults() throws {
        let calls = [
            APIToolCall(
                id: "list-1",
                type: "function",
                function: .init(
                    name: "list_files",
                    arguments: .object(["path": .string("Sources")])
                )
            ),
            APIToolCall(
                id: "read-1",
                type: "function",
                function: .init(
                    name: "read_file",
                    arguments: .object(["path": .string("Sources/App.swift")])
                )
            ),
        ]
        var activities = calls.map { CodingToolActivity(call: $0) }
        activities[0].state = .succeeded
        activities[0].output = #"{"files":["Sources/App.swift"],"truncated":false}"#
        activities[1].state = .running

        let decoded = try JSONDecoder().decode(
            [CodingToolActivity].self,
            from: JSONEncoder().encode(activities)
        )

        XCTAssertEqual(decoded.map(\.call.function.name), ["list_files", "read_file"])
        XCTAssertEqual(decoded.map(\.state), [.succeeded, .running])
        XCTAssertEqual(
            CodingWorkspace.displayResult(try XCTUnwrap(decoded[0].output)),
            "Sources/App.swift"
        )
        XCTAssertEqual(
            CodingWorkspace.activitySummary(of: decoded[1].call),
            "Read Sources/App.swift"
        )
    }

    func testCodingWorkspaceRejectsSymlinkEscapesAndAmbiguousWrites() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-tools-\(UUID().uuidString)", isDirectory: true)
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-outside-\(UUID().uuidString).txt")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try "secret".write(to: outside, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("outside.txt"),
            withDestinationURL: outside
        )
        let duplicate = root.appendingPathComponent("duplicate.txt")
        try "same same".write(to: duplicate, atomically: true, encoding: .utf8)
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: outside)
        }

        let readLink = APIToolCall(
            function: .init(
                name: "read_file",
                arguments: .object(["path": .string("outside.txt")])
            )
        )
        XCTAssertThrowsError(try CodingWorkspace.execute(readLink, workspaceRoot: root.path))

        let ambiguous = APIToolCall(
            function: .init(
                name: "replace_in_file",
                arguments: .object([
                    "path": .string("duplicate.txt"),
                    "old_text": .string("same"),
                    "new_text": .string("new"),
                ])
            )
        )
        XCTAssertThrowsError(try CodingWorkspace.execute(ambiguous, workspaceRoot: root.path))
    }

    func testMemoryCacheAndProviderSchemasDecode() throws {
        let memory = try JSONDecoder().decode(
            MemoriesResponse.self,
            from: Data(
                #"{"schema":"machboost.memories.v1","memories":[{"id":"mem_1","workspace_id":"repo","scope":"team","kind":"fix","title":"Retry checkout","content":"Reuse the key","confidence":0.9,"pinned":false,"stale":false}]}"#.utf8
            )
        )
        let metrics = try JSONDecoder().decode(
            CacheMetrics.self,
            from: Data(
                #"{"schema":"machboost.cache-metrics.v1","totals":{"avoided_prompt_tokens":2400},"namespaces":{"abc":{"exact_cache_hits":1}}}"#.utf8
            )
        )
        let providers = try JSONDecoder().decode(
            ProvidersResponse.self,
            from: Data(
                #"{"schema":"machboost.providers.v1","providers":[{"id":"provider_1","name":"Fallback","base_url":"https://api.example.com","models":["coder"],"enabled":true,"has_secret":true,"monthly_budget_usd":20,"spent_this_month_usd":1.5,"remaining_budget_usd":18.5}]}"#.utf8
            )
        )

        XCTAssertEqual(memory.memories.first?.title, "Retry checkout")
        XCTAssertEqual(metrics.totals["avoided_prompt_tokens"], 2_400)
        XCTAssertEqual(providers.providers.first?.remainingBudgetUSD, 18.5)
    }

    func testProviderSecretRestorationDoesNotOverwriteProviderConfiguration() async throws {
        let session = mockSession { request in
            XCTAssertEqual(request.url?.path, "/api/providers/secret")
            let data = try XCTUnwrap(request.httpBody)
            let object = try XCTUnwrap(
                JSONSerialization.jsonObject(with: data) as? [String: String]
            )
            XCTAssertEqual(object, [
                "provider_id": "provider_1",
                "api_key": "local-key",
            ])
            return self.response(
                for: request,
                status: 200,
                body: #"{"provider_id":"provider_1","has_secret":true}"#
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            session: session
        )

        try await api.setProviderSecret(id: "provider_1", apiKey: "local-key")
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

    func testServerVersionComesFromHealthProbe() async throws {
        let session = mockSession { request in
            XCTAssertEqual(request.url?.path, "/healthz")
            XCTAssertEqual(request.timeoutInterval, 0.4, accuracy: 0.001)
            return self.response(
                for: request,
                body: #"{"status":"ok","version":"0.13.1"}"#
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            session: session
        )

        let version = try await api.serverVersion(timeoutInterval: 0.4)

        XCTAssertEqual(version, "0.13.1")
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

    func testNDJSONChatStreamPreservesMuseReasoningAndToolCalls() async throws {
        let session = mockSession { request in
            self.response(
                for: request,
                contentType: "application/x-ndjson",
                body: """
                {"request_id":"muse-stream-1","message":{"role":"assistant","content":"","thinking":"I should search."},"done":false}
                {"request_id":"muse-stream-1","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"search_repository","arguments":{"query":"cancellation"}}}]},"done":false}
                {"request_id":"muse-stream-1","message":{"role":"assistant","content":"Found it."},"done":true,"done_reason":"stop"}

                """
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            session: session
        )
        let request = ChatRequest(
            requestID: "muse-stream-1",
            model: "muse-glimmer:30b",
            messages: [.init(role: "user", content: "Find cancellation.")],
            context: [],
            options: .init(maxTokens: 64, temperature: 1, affinityKey: nil),
            reasoningStrength: "medium"
        )
        var events: [ChatEvent] = []

        for try await event in api.streamChat(request) {
            events.append(event)
        }

        XCTAssertEqual(events[0].message?.thinking, "I should search.")
        XCTAssertEqual(events[1].message?.toolCalls?.first?.function.name, "search_repository")
        XCTAssertEqual(
            events[1].message?.toolCalls?.first?.function.arguments,
            .object(["query": .string("cancellation")])
        )
        XCTAssertEqual(events[2].message?.content, "Found it.")
    }

    @MainActor
    func testConversationMarkdownExportIsOrderedAndSanitized() throws {
        let conversation = Conversation(title: "Release: notes/July", model: "qwen2.5:3b")
        conversation.messages = [
            ChatMessage(
                role: .assistant,
                content: "Ready.",
                reasoningContent: "The release checks passed.",
                toolCallsJSON: "[{\"function\":{\"name\":\"run_checks\"}}]",
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
        XCTAssertTrue(markdown.contains("<summary>Reasoning</summary>"))
        XCTAssertTrue(markdown.contains("The release checks passed."))
        XCTAssertTrue(markdown.contains("### Tool calls"))
        XCTAssertTrue(markdown.contains("run_checks"))
        XCTAssertTrue(markdown.contains("Model: `qwen2.5:3b`"))
    }

    @MainActor
    func testDaemonEnvironmentKeepsBundledRuntimeImmutable() {
        let environment = DaemonManager.launchEnvironment(
            base: [
                "MACHBOOST_API_TOKEN": "stale-token",
                "PATH": "/usr/bin",
            ],
            apiToken: nil
        )

        XCTAssertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        XCTAssertEqual(environment["PYTHONUNBUFFERED"], "1")
        XCTAssertEqual(environment["PATH"], "/usr/bin")
        XCTAssertNil(environment["MACHBOOST_API_TOKEN"])

        let secured = DaemonManager.launchEnvironment(base: [:], apiToken: "fresh-token")
        XCTAssertEqual(secured["MACHBOOST_API_TOKEN"], "fresh-token")
    }

    @MainActor
    func testDaemonOnlyReplacesOlderVersions() {
        XCTAssertTrue(DaemonManager.isOlderVersion("0.12.1", than: "0.13.1"))
        XCTAssertTrue(DaemonManager.isOlderVersion("0.9.0", than: "0.10.0"))
        XCTAssertFalse(DaemonManager.isOlderVersion("0.13.1", than: "0.13.1"))
        XCTAssertFalse(DaemonManager.isOlderVersion("0.14.0", than: "0.13.1"))
    }

    @MainActor
    func testDaemonStartsAndShutsDownFromIsolatedSourceRuntime() async throws {
        let temporaryRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-source-\(UUID().uuidString)", isDirectory: true)
        let packageRoot = temporaryRoot.appendingPathComponent("machboost", isDirectory: true)
        try FileManager.default.createDirectory(
            at: packageRoot,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: temporaryRoot) }
        try "".write(
            to: packageRoot.appendingPathComponent("__init__.py"),
            atomically: true,
            encoding: .utf8
        )
        try """
        import argparse
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def send_json(self, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send_json({"status": "ok"})

            def do_POST(self):
                self.send_json({"status": "stopping"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("command")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, required=True)
        args, _ = parser.parse_known_args()
        HTTPServer((args.host, args.port), Handler).serve_forever()
        """.write(
            to: packageRoot.appendingPathComponent("cli.py"),
            atomically: true,
            encoding: .utf8
        )
        let manager = DaemonManager(sourceRootOverride: temporaryRoot)
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

    private func serverMetrics(active: Int, queued: Int, p50: Double) -> ServerMetrics {
        ServerMetrics(
            schema: "machboost.metrics.v1",
            operations: .init(
                activeCount: active,
                totals: .init(
                    started: 0,
                    completed: 0,
                    cancelled: 0,
                    failed: 0,
                    generatedTokens: 0
                ),
                latencySeconds: .init(p50: p50, p95: p50),
                generationTokensPerSecond: 0
            ),
            models: [],
            scheduler: .init(
                activeRequests: active,
                queuedRequests: queued,
                rejectedRequests: 0
            ),
            process: .init(peakResidentMemoryBytes: 0)
        )
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
