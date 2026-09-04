import Foundation
import MachBoostDaemonClient
import SwiftData
import XCTest
@testable import MachBoost

final class MachBoostTests: XCTestCase {
    @MainActor
    func testInferencePresentationSeparatesRemoteFailureFromLocalDestination() {
        let presentation = AppState.inferencePresentation(
            mode: .team,
            serverIsRunning: true,
            onlineHostNames: [],
            selectedOnlineName: nil
        )

        XCTAssertEqual(presentation.destination, "This Mac")
        XCTAssertEqual(presentation.status, "Local fallback \u{00b7} remote unavailable")
    }

    @MainActor
    func testInferencePresentationCountsOnlyOnlineHosts() {
        let presentation = AppState.inferencePresentation(
            mode: .team,
            serverIsRunning: true,
            onlineHostNames: ["Studio", "Build Mac"],
            selectedOnlineName: "Studio"
        )

        XCTAssertEqual(presentation.destination, "Host pool (2)")
        XCTAssertEqual(presentation.status, "Host pool (2)")
    }

    func testCommunityCredentialStorePersistsUpdatesAndDeletesSecrets() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = CommunityCredentialStore(root: root)

        try store.save(value: "first-token", account: "lan-api-token")
        try store.save(value: "provider-secret", account: "provider-test")
        try store.save(value: "updated-token", account: "lan-api-token")

        XCTAssertEqual(store.value(account: "lan-api-token"), "updated-token")
        XCTAssertEqual(store.value(account: "provider-test"), "provider-secret")
        let attributes = try FileManager.default.attributesOfItem(
            atPath: store.credentialsURL.path
        )
        XCTAssertEqual(attributes[.posixPermissions] as? NSNumber, NSNumber(value: 0o600))

        try store.delete(account: "lan-api-token")
        XCTAssertNil(store.value(account: "lan-api-token"))
        XCTAssertEqual(store.value(account: "provider-test"), "provider-secret")
    }

    func testAppsGatewayOmitsCredentialsForOpenLocalDaemon() throws {
        let token = try AppsGatewayCredentials.localToken(
            authenticationRequired: false,
            runtimeToken: "stale-runtime-token",
            keychainToken: "stale-keychain-token"
        )

        XCTAssertNil(token)
    }

    func testAppsGatewayUsesRuntimeTokenForSecuredLocalDaemon() throws {
        let token = try AppsGatewayCredentials.localToken(
            authenticationRequired: true,
            runtimeToken: "runtime-token",
            keychainToken: "keychain-token"
        )

        XCTAssertEqual(token, "runtime-token")
    }

    func testAppsGatewayFallsBackToKeychainForSecuredLocalDaemon() throws {
        let token = try AppsGatewayCredentials.localToken(
            authenticationRequired: true,
            runtimeToken: nil,
            keychainToken: "keychain-token"
        )

        XCTAssertEqual(token, "keychain-token")
    }

    func testAppsGatewayRejectsSecuredLocalDaemonWithoutCredentials() {
        XCTAssertThrowsError(
            try AppsGatewayCredentials.localToken(
                authenticationRequired: true,
                runtimeToken: nil,
                keychainToken: nil
            )
        )
    }

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
                reservedRequests: 8
            ),
            HostRoutingPolicy.score(metrics: idleColdHost, modelLoaded: false)
        )

        XCTAssertLessThan(
            HostRoutingPolicy.score(
                metrics: availableResident,
                modelLoaded: true,
                roundTripSeconds: 0.01,
                replicas: 2,
                activeRequests: 1,
                queuedRequests: 1
            ),
            HostRoutingPolicy.score(
                metrics: availableResident,
                modelLoaded: true,
                roundTripSeconds: 0.2,
                replicas: 1,
                activeRequests: 2,
                queuedRequests: 2
            )
        )
    }

    func testHostRoutingUsesLocalAsEmergencyFallbackWhenRemotesAreOffline() {
        XCTAssertFalse(
            HostRoutingPolicy.shouldIncludeLocal(
                includeLocalInPool: false,
                prefersLocal: false,
                hasOnlineRemote: true
            )
        )
        XCTAssertTrue(
            HostRoutingPolicy.shouldIncludeLocal(
                includeLocalInPool: false,
                prefersLocal: false,
                hasOnlineRemote: false
            )
        )
        XCTAssertTrue(
            HostRoutingPolicy.shouldIncludeLocal(
                includeLocalInPool: false,
                prefersLocal: true,
                hasOnlineRemote: true
            )
        )
        XCTAssertTrue(
            HostRoutingPolicy.shouldIncludeLocal(
                includeLocalInPool: true,
                prefersLocal: false,
                hasOnlineRemote: true
            )
        )
    }

    func testHostRoutingFailsOverOnlyBeforeVisibleOutput() {
        XCTAssertTrue(
            HostRoutingPolicy.canFailOver(
                error: MachBoostAPIError.server(status: 503, message: "busy"),
                emittedOutput: false
            )
        )
        XCTAssertTrue(
            HostRoutingPolicy.canFailOver(
                error: URLError(.timedOut),
                emittedOutput: false
            )
        )
        XCTAssertFalse(
            HostRoutingPolicy.canFailOver(
                error: MachBoostAPIError.server(status: 503, message: "busy"),
                emittedOutput: true
            )
        )
        XCTAssertFalse(
            HostRoutingPolicy.canFailOver(
                error: MachBoostAPIError.server(status: 400, message: "bad request"),
                emittedOutput: false
            )
        )
    }

    func testHostRoutingPreservesAuthoritativeFullContentCorrection() {
        let event = ChatEvent(
            requestID: "chat-prefix-repair",
            message: .init(role: "assistant", content: ""),
            done: false,
            doneReason: nil,
            totalDuration: nil,
            evalDuration: nil,
            evalCount: nil,
            machboost: .init(
                backend: nil,
                stats: nil,
                timeToFirstTokenSeconds: nil,
                fullContent: "You're welcome!"
            ),
            error: nil
        )

        XCTAssertTrue(AppState.hasVisibleOutput(event))
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

    func testTurnMetricsUsesTheFirstVisibleRoundForTTFT() {
        var metrics = GenerationTurnMetrics()
        metrics.absorb(
            ChatEvent(
                requestID: "hidden",
                message: nil,
                done: true,
                doneReason: "stop",
                totalDuration: 12_000_000_000,
                evalDuration: 250_000_000,
                evalCount: 5,
                machboost: .init(
                    backend: "mlx-vlm",
                    stats: nil,
                    timeToFirstTokenSeconds: 11.8
                ),
                error: nil
            ),
            producedVisibleOutput: false
        )
        metrics.absorb(
            ChatEvent(
                requestID: "visible",
                message: nil,
                done: true,
                doneReason: "stop",
                totalDuration: 1_000_000_000,
                evalDuration: 500_000_000,
                evalCount: 10,
                machboost: .init(
                    backend: "mlx-vlm",
                    stats: nil,
                    timeToFirstTokenSeconds: 0.62
                ),
                error: nil
            ),
            producedVisibleOutput: true
        )
        let message = ChatMessage(role: .assistant, content: "Done")

        metrics.apply(to: message)

        XCTAssertEqual(message.timeToFirstTokenSeconds, 0.62)
        XCTAssertEqual(message.generatedTokens, 15)
    }

    func testTurnMetricsPersistPaidProviderMetadata() throws {
        let event = try JSONDecoder().decode(
            ChatEvent.self,
            from: Data(
                #"{"request_id":"route-1","message":{"role":"assistant","content":""},"done":true,"total_duration":800000000,"eval_duration":800000000,"eval_count":20,"machboost":{"backend":"external","route":{"source":"external","provider_id":"paid-1","latency_seconds":0.8,"cost_usd":0.00125,"buffered_upstream":true}}}"#.utf8
            )
        )
        var metrics = GenerationTurnMetrics()
        metrics.absorb(event)
        let message = ChatMessage(role: .assistant, content: "Done")

        metrics.apply(to: message)

        XCTAssertEqual(message.inferenceSource, "external")
        XCTAssertEqual(message.providerID, "paid-1")
        XCTAssertEqual(message.providerLatencySeconds, 0.8)
        XCTAssertEqual(message.providerCostUSD, 0.00125)
        XCTAssertEqual(message.tokensPerSecond, 25)
    }

    func testTurnMetricsPersistHostAndPromptPrefillBreakdown() throws {
        let event = try JSONDecoder().decode(
            ChatEvent.self,
            from: Data(
                #"{"request_id":"prefill-1","message":{"role":"assistant","content":"Done"},"done":true,"total_duration":18200000000,"load_duration":0,"prompt_eval_duration":17700000000,"prompt_eval_count":2204,"eval_duration":500000000,"eval_count":20,"machboost":{"backend":"mlx-vlm","time_to_first_token_seconds":18.0,"scheduler":{"queue_wait_seconds":0.0},"stats":{"cached_prompt_tokens":2188,"prompt_cache_prefix_tokens":2188}}}"#.utf8
            )
        )
        var metrics = GenerationTurnMetrics()
        metrics.absorb(event)
        metrics.recordRoute(
            InferenceRouteRecord(
                hostID: "studio-id",
                hostName: "Mac Studio",
                expectedDelay: 0.1
            )
        )
        let message = ChatMessage(role: .assistant, content: "Done")

        metrics.apply(to: message)

        XCTAssertEqual(message.inferenceHostID, "studio-id")
        XCTAssertEqual(message.inferenceHostName, "Mac Studio")
        XCTAssertEqual(message.timeToFirstTokenSeconds, 18)
        XCTAssertEqual(message.modelLoadSeconds, nil)
        XCTAssertEqual(message.queueWaitSeconds, 0)
        XCTAssertEqual(message.promptEvalSeconds, 17.7)
        XCTAssertEqual(message.promptTokens, 2_204)
        XCTAssertEqual(message.cachedPromptTokens, 2_188)
        XCTAssertEqual(message.tokensPerSecond, 40)
    }

    @MainActor
    func testConversationPersistsPreferredInferenceDevice() {
        let conversation = Conversation(
            title: "Shared coding task",
            model: "mlx-community/Muse-Glimmer-30B-4bit"
        )

        conversation.preferredInferenceHostID = "studio-id"

        XCTAssertEqual(conversation.preferredInferenceHostID, "studio-id")
    }

    @MainActor
    func testHostDiscoveryRecognizesOnlyMatchingDeviceIdentityAsSelf() {
        XCTAssertTrue(
            MachBoostHostDiscovery.isSelf(deviceID: "device-a", localDeviceID: "device-a")
        )
        XCTAssertFalse(
            MachBoostHostDiscovery.isSelf(deviceID: "device-b", localDeviceID: "device-a")
        )
        XCTAssertFalse(MachBoostHostDiscovery.isSelf(deviceID: nil, localDeviceID: "device-a"))
    }

    @MainActor
    func testTeamConnectionRejectsLocalAliasesButKeepsRemoteHosts() throws {
        let names: Set<String> = ["workstation.local"]
        let addresses: Set<String> = ["192.168.1.25"]

        XCTAssertTrue(
            AppState.isLocalTeamEndpoint(
                try XCTUnwrap(URL(string: "http://workstation.local:11435")),
                localNames: names,
                localAddresses: addresses
            )
        )
        XCTAssertTrue(
            AppState.isLocalTeamEndpoint(
                try XCTUnwrap(URL(string: "http://192.168.1.25:11435")),
                localNames: names,
                localAddresses: addresses
            )
        )
        XCTAssertTrue(
            AppState.isLocalTeamEndpoint(
                try XCTUnwrap(URL(string: "http://127.0.0.1:11435")),
                localNames: names,
                localAddresses: addresses
            )
        )
        XCTAssertFalse(
            AppState.isLocalTeamEndpoint(
                try XCTUnwrap(URL(string: "http://studio.local:11435")),
                localNames: names,
                localAddresses: addresses
            )
        )
    }

    @MainActor
    func testTeamClientsHideThisMacAndCollapseStaleDeviceIDs() throws {
        let payload = """
        [
          {"device_id":"local-id","principal":{"id":"p1","name":"Local"},"device_name":"This Mac","app_version":"1","mode":"connect","first_seen_at":"2026-01-01T00:00:00Z","last_seen_at":"2026-01-01T00:00:03Z","request_count":3,"online":true},
          {"device_id":"old-id","principal":{"id":"p2","name":"Old"},"device_name":"Build Mac","app_version":"1","mode":"connect","first_seen_at":"2026-01-01T00:00:00Z","last_seen_at":"2026-01-01T00:00:01Z","request_count":8,"online":false},
          {"device_id":"new-id","principal":{"id":"p3","name":"New"},"device_name":"build mac","app_version":"2","mode":"connect","first_seen_at":"2026-01-01T00:00:02Z","last_seen_at":"2026-01-01T00:00:04Z","request_count":2,"online":true}
        ]
        """
        let clients: [TeamClient] = try JSONDecoder().decode(
            Array<TeamClient>.self,
            from: Data(payload.utf8)
        )

        let visible = AppState.deduplicatedTeamClients(
            clients,
            localDeviceID: "local-id",
            localDeviceName: "This Mac"
        )

        XCTAssertEqual(visible.count, 1)
        XCTAssertEqual(visible.first?.deviceID, "new-id")
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
        XCTAssertEqual(CodingWorkspace.tools(for: .plan).count, 4)
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

    func testWorkspaceChangesSessionExcludesUnrelatedWorkingTreeFiles() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-session-changes-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", root.path, "init", "-b", "main"]
        process.standardOutput = Pipe()
        process.standardError = Pipe()
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
        try Data("unrelated\n".utf8).write(to: root.appendingPathComponent("other.txt"))

        var activity = CodingToolActivity(
            call: APIToolCall(function: .init(name: "create_file", arguments: .object([:])))
        )
        activity.state = .succeeded
        activity.changedPath = "Sources/New.swift"
        activity.changePatch = "--- /dev/null\n+++ b/Sources/New.swift\n@@ -0,0 +1 @@\n+let ready = true"

        let snapshot = WorkspaceChanges.session(
            workspaceRoot: root.path,
            activities: [activity]
        )

        XCTAssertEqual(snapshot.changes.map(\.path), ["Sources/New.swift"])
        XCTAssertEqual(snapshot.changes.first?.status, "Created")
        XCTAssertFalse(snapshot.changes.contains { $0.path == "other.txt" })
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
            workspaceID: "workspace-123",
            workspaceTopK: 4,
            workspaceMaxChars: 12_000,
            machboost: .init(memory: "off")
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        let options = try XCTUnwrap(object["options"] as? [String: Any])

        XCTAssertEqual(object["request_id"] as? String, "chat-123")
        XCTAssertEqual(object["keep_alive"] as? String, "forever")
        XCTAssertEqual(object["workspace_id"] as? String, "workspace-123")
        XCTAssertEqual(object["workspace_top_k"] as? Int, 4)
        XCTAssertEqual(object["workspace_max_chars"] as? Int, 12_000)
        XCTAssertEqual(options["num_predict"] as? Int, 64)
        XCTAssertEqual(options["affinity_key"] as? String, "thread-1")
        let extensions = try XCTUnwrap(object["machboost"] as? [String: Any])
        XCTAssertEqual(extensions["memory"] as? String, "off")
    }

    func testChatRequestEncodesPaidProviderModelRoute() throws {
        let request = ChatRequest(
            requestID: "chat-route-1",
            model: "mlx-community/Muse-Glimmer-30B-4bit",
            messages: [.init(role: "user", content: "Hello")],
            context: [],
            options: .init(maxTokens: 64, temperature: 0.2, affinityKey: nil),
            machboost: .init(
                route: .init(
                    mode: "local_first",
                    providerID: "production",
                    model: "gpt-5-mini"
                )
            )
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        )
        let extensionObject = try XCTUnwrap(object["machboost"] as? [String: Any])
        let route = try XCTUnwrap(extensionObject["route"] as? [String: Any])

        XCTAssertEqual(route["mode"] as? String, "local_first")
        XCTAssertEqual(route["provider_id"] as? String, "production")
        XCTAssertEqual(route["model"] as? String, "gpt-5-mini")
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

    func testPullProgressDecodesAggregateShardMetrics() throws {
        let event = try JSONDecoder().decode(
            PullEvent.self,
            from: Data(
                #"{"request_id":"pull-1","status":"downloading","file":"2 shards downloading","completed":12000000000,"total":16000000000,"unit":"bytes","files_completed":10,"files_total":12,"active_files":["model-2.safetensors","model-3.safetensors"],"speed_bytes_per_second":125000000,"eta_seconds":32,"done":false}"#.utf8
            )
        )

        XCTAssertEqual(event.filesCompleted, 10)
        XCTAssertEqual(event.filesTotal, 12)
        XCTAssertEqual(event.activeFiles?.count, 2)
        XCTAssertEqual(event.speedBytesPerSecond, 125_000_000)
        XCTAssertEqual(event.etaSeconds, 32)
    }

    func testDeleteModelPurgesManagedWeights() async throws {
        let session = mockSession { request in
            XCTAssertEqual(request.url?.path, "/api/delete")
            let body = try XCTUnwrap(request.httpBody)
            let payload = try XCTUnwrap(
                JSONSerialization.jsonObject(with: body) as? [String: Any]
            )
            XCTAssertEqual(payload["model"] as? String, "mlx-community/example")
            XCTAssertEqual(payload["purge"] as? Bool, true)
            return self.response(
                for: request,
                body: #"{"status":"success","removed":true,"bytes_removed":8000000000,"unloaded":1,"repositories":["mlx-community/example"]}"#
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            session: session
        )

        let response = try await api.deleteModel(model: "mlx-community/example")

        XCTAssertTrue(response.removed)
        XCTAssertEqual(response.bytesRemoved, 8_000_000_000)
        XCTAssertEqual(response.unloaded, 1)
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

    func testCodingWorkspaceToolsAreBoundedToSelectedRepository() async throws {
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
        let searchResult = try await CodingWorkspace.execute(search, workspaceRoot: root.path)
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
        let replaceResult = try await CodingWorkspace.execute(replace, workspaceRoot: root.path)
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
        do {
            _ = try await CodingWorkspace.execute(escape, workspaceRoot: root.path)
            XCTFail("Expected the repository boundary to reject the path")
        } catch {}
    }

    func testCodingWorkspaceDefaultsFileReadsToOneHundredTwentyLines() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-read-limit-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("Large.swift")
        let lines = (1 ... 180).map { "let value\($0) = \($0)" }
        try lines.joined(separator: "\n").write(to: source, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: root) }

        let call = APIToolCall(
            function: .init(
                name: "read_file",
                arguments: .object(["path": .string("Large.swift")])
            )
        )
        let result = try await CodingWorkspace.execute(call, workspaceRoot: root.path)
        let data = try XCTUnwrap(result.content.data(using: .utf8))
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )

        XCTAssertEqual(object["start_line"] as? Int, 1)
        XCTAssertEqual(object["end_line"] as? Int, 120)
        XCTAssertEqual(object["truncated"] as? Bool, true)
        XCTAssertFalse((object["content"] as? String ?? "").contains("value121"))
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

    func testCodingWorkspaceDeletesFilesAndRunsBoundedCommands() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("machboost-command-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let doomed = root.appendingPathComponent("delete-me.txt")
        try "remove me\n".write(to: doomed, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: root) }

        let command = APIToolCall(
            function: .init(
                name: "run_command",
                arguments: .object([
                    "command": .string("pwd"),
                    "timeout_seconds": .number(5),
                ])
            )
        )
        let commandResult = try await CodingWorkspace.execute(command, workspaceRoot: root.path)
        XCTAssertTrue(commandResult.content.contains(root.path))
        XCTAssertEqual(
            CodingWorkspace.permissionDecision(for: command, mode: .manual),
            .ask
        )
        XCTAssertEqual(
            CodingWorkspace.permissionDecision(for: command, mode: .bypass),
            .allow
        )

        let delete = APIToolCall(
            function: .init(
                name: "delete_file",
                arguments: .object(["path": .string("delete-me.txt")])
            )
        )
        let deleteResult = try await CodingWorkspace.execute(delete, workspaceRoot: root.path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: doomed.path))
        XCTAssertEqual(deleteResult.changedPath, "delete-me.txt")
        XCTAssertTrue(deleteResult.changePatch?.contains("+++ /dev/null") ?? false)
    }

    func testCodingWorkspaceRejectsSymlinkEscapesAndAmbiguousWrites() async throws {
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
        do {
            _ = try await CodingWorkspace.execute(readLink, workspaceRoot: root.path)
            XCTFail("Expected the symlink escape to fail")
        } catch {}

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
        do {
            _ = try await CodingWorkspace.execute(ambiguous, workspaceRoot: root.path)
            XCTFail("Expected the ambiguous replacement to fail")
        } catch {}
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

    func testServerHealthReportsAuthenticatedLANMode() async throws {
        let session = mockSession { request in
            XCTAssertEqual(request.url?.path, "/healthz")
            XCTAssertNil(request.value(forHTTPHeaderField: "Authorization"))
            return self.response(
                for: request,
                body: #"{"status":"ok","version":"0.15.0","authentication":"required"}"#
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            apiToken: "saved-token",
            session: session
        )

        let health = try await api.serverHealth(timeoutInterval: 0.4)

        XCTAssertTrue(health.isReady)
        XCTAssertTrue(health.requiresAuthentication)
        XCTAssertEqual(health.version, "0.15.0")
    }

    func testAuthenticatedVersionProbeUsesSavedBearerToken() async throws {
        let session = mockSession { request in
            XCTAssertEqual(request.url?.path, "/api/version")
            XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer saved-token")
            return self.response(for: request, body: #"{"version":"0.15.0"}"#)
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            apiToken: "saved-token",
            session: session
        )

        let version = try await api.authenticatedServerVersion(timeoutInterval: 0.4)

        XCTAssertEqual(version, "0.15.0")
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
                this must never be decoded after the terminal event

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
                {"request_id":"muse-stream-1","message":{"role":"assistant","content":"Found it."},"machboost":{"full_content":"Found it."},"done":true,"done_reason":"stop"}

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
        XCTAssertEqual(events[2].machboost?.fullContent, "Found it.")
    }

    func testChatRequestsEncodeToolSchemasDeterministically() async throws {
        let lock = NSLock()
        var bodies: [Data] = []
        let session = mockSession { request in
            if let body = request.httpBody {
                lock.lock()
                bodies.append(body)
                lock.unlock()
            }
            return self.response(
                for: request,
                contentType: "application/x-ndjson",
                body: #"{"request_id":"stable","message":{"role":"assistant","content":""},"done":true}"#
                    + "\n"
            )
        }
        let api = MachBoostAPI(
            endpoint: URL(string: "http://127.0.0.1:11435")!,
            session: session
        )
        var firstProperties: [String: JSONValue] = [:]
        firstProperties["path"] = .object([
            "type": .string("string"),
            "description": .string("Repository path"),
        ])
        firstProperties["limit"] = .object([
            "type": .string("integer"),
            "description": .string("Result limit"),
        ])
        var secondProperties: [String: JSONValue] = [:]
        secondProperties["limit"] = .object([
            "description": .string("Result limit"),
            "type": .string("integer"),
        ])
        secondProperties["path"] = .object([
            "description": .string("Repository path"),
            "type": .string("string"),
        ])

        for properties in [firstProperties, secondProperties] {
            let request = ChatRequest(
                requestID: "stable",
                model: "muse-glimmer:30b",
                messages: [.init(role: "user", content: "Inspect it")],
                context: [],
                options: .init(maxTokens: 8, temperature: 0, affinityKey: "conversation"),
                tools: [
                    .init(
                        function: .init(
                            name: "list_files",
                            parameters: .object([
                                "type": .string("object"),
                                "properties": .object(properties),
                            ])
                        )
                    ),
                ]
            )
            for try await _ in api.streamChat(request) {}
        }

        XCTAssertEqual(bodies.count, 2)
        XCTAssertEqual(bodies[0], bodies[1])
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
    func testDaemonRecoveryRecognizesOnlyBundledMachBoostServer() {
        let bundled = "/Applications/MachBoost.app/Contents/Resources/runtime/python/bin/python3 -m machboost.cli serve --host 0.0.0.0 --port 11435"

        XCTAssertTrue(DaemonManager.isBundledDaemonCommand(bundled, port: 11_435))
        XCTAssertFalse(DaemonManager.isBundledDaemonCommand(bundled, port: 11_436))
        XCTAssertFalse(
            DaemonManager.isBundledDaemonCommand(
                "/usr/bin/python3 -m machboost.cli serve --port 11435",
                port: 11_435
            )
        )
        XCTAssertFalse(
            DaemonManager.isBundledDaemonCommand(
                "/Applications/MachBoost.app/Contents/Resources/runtime/python/bin/python3 -m http.server 11435",
                port: 11_435
            )
        )
    }

    @MainActor
    func testDaemonRuntimeMatchRejectsDebugPythonOnTheAppPort() {
        let bundledRuntime = "/Applications/MachBoost.app/Contents/Resources/runtime/python/bin/python3"
        let bundled = "\(bundledRuntime) -m machboost.cli serve --port 11435"
        let debug = "/usr/bin/python3 -m machboost.cli serve --port 11435"

        XCTAssertTrue(
            DaemonManager.daemonCommandMatchesRuntime(
                bundled,
                executablePath: bundledRuntime,
                port: 11_435
            )
        )
        XCTAssertFalse(
            DaemonManager.daemonCommandMatchesRuntime(
                debug,
                executablePath: bundledRuntime,
                port: 11_435
            )
        )
        XCTAssertFalse(
            DaemonManager.daemonCommandMatchesRuntime(
                bundled,
                executablePath: bundledRuntime,
                port: 11_436
            )
        )
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

    func testExtensionsSchemaDecodesRedactedConnectorsAndSkills() throws {
        let data = Data(
            #"{"schema":"machboost.extensions.v1","mcp_servers":[{"id":"mcp_1","name":"Tracker","transport":"http","url":"https://example.test/mcp","command":null,"args":[],"enabled":true,"tool_count":4,"last_status":"ready","last_error":null,"env_keys":[],"header_names":["Authorization"],"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}],"skills":[{"id":"skill_1","name":"Concise","instructions":"Use short answers.","enabled":true,"created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T00:00:00Z"}],"gateway_tools":[]}"#.utf8
        )

        let decoded = try JSONDecoder().decode(ExtensionsResponse.self, from: data)

        XCTAssertEqual(decoded.mcpServers.first?.name, "Tracker")
        XCTAssertEqual(decoded.mcpServers.first?.headerNames, ["Authorization"])
        XCTAssertEqual(decoded.skills.first?.instructions, "Use short answers.")
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
