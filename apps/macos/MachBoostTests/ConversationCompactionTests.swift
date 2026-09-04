import Foundation
import SwiftData
import XCTest
@testable import MachBoost

@MainActor
final class ConversationCompactionTests: XCTestCase {
    func testSmallUsageNeverRoundsDownToZero() {
        let label = ConversationCompaction.usageLabel(tokens: 50, contextLength: 262_144)
        XCTAssertTrue(label.contains("50"))
        XCTAssertTrue(label.contains("<0.1%"))
        XCTAssertTrue(ConversationCompaction.usageLabel(tokens: 0, contextLength: 0).contains("0.0%"))
    }

    func testUsageIncludesInstructionsAttachmentsAndUTF8Bytes() {
        let message = ChatMessage(role: .user, content: String(repeating: "\u{4f60}", count: 100))
        let estimate = ConversationCompaction.estimatedTokens(
            summary: nil, messages: [message], additionalBytes: 3_000
        )
        XCTAssertEqual(estimate, 1_108)
        XCTAssertTrue(ConversationCompaction.shouldCompact(
            estimatedTokens: estimate, contextLength: 1_536,
            reservedOutputTokens: 512, thresholdPercent: 90
        ))
    }

    func testBlankOutputLimitUsesRemainingContext() {
        XCTAssertEqual(
            ConversationCompaction.resolvedOutputTokens(
                configuredLimit: 0,
                estimatedInputTokens: 1_250,
                contextLength: 8_192
            ),
            6_942
        )
    }

    func testExplicitOutputLimitIsAContextSafeCeiling() {
        XCTAssertEqual(
            ConversationCompaction.resolvedOutputTokens(
                configuredLimit: 512,
                estimatedInputTokens: 1_250,
                contextLength: 8_192
            ),
            512
        )
        XCTAssertEqual(
            ConversationCompaction.resolvedOutputTokens(
                configuredLimit: 4_096,
                estimatedInputTokens: 8_000,
                contextLength: 8_192
            ),
            192
        )
    }

    func testManualSummaryAcceptsSingleCompletedExchange() {
        let messages = exchange("hi", "hello")
        XCTAssertEqual(ConversationCompaction.candidates(messages: messages, keepRecent: 0).count, 2)
        XCTAssertTrue(ConversationCompaction.candidates(messages: messages, keepRecent: 2).isEmpty)
    }

    func testAutomaticSummaryKeepsLastCompleteExchange() {
        let messages = exchange("old question", "old answer") + exchange("new question", "new answer")
        let candidates = ConversationCompaction.candidates(messages: messages, keepRecent: 2)
        XCTAssertEqual(candidates.map(\.content), ["old question", "old answer"])
        XCTAssertEqual(ConversationCompaction.candidates(messages: messages, keepRecent: 1).count, 2)
    }

    func testSummaryHasSeparateBudgetAndCacheAffinity() throws {
        for (model, requiresReasoning, strength) in [
            ("gemma", false, "off"), ("muse-glimmer", true, "low")
        ] {
            let request = ConversationCompaction.request(
                requestID: "summary-test", model: model, transcript: "USER: keep project A",
                priorSummary: "Earlier decision B", requiresReasoning: requiresReasoning,
                extensions: .init(memory: "off", skills: "off")
            )
            XCTAssertEqual(request.options.maxTokens, 2_048)
            XCTAssertEqual(request.options.affinityKey, "summary-test")
            XCTAssertEqual(request.reasoningStrength, strength)
            XCTAssertTrue(request.context.isEmpty)
            XCTAssertNil(request.tools)
            XCTAssertTrue(request.messages.last!.content.contains("Earlier decision B"))
            XCTAssertTrue(request.messages.last!.content.contains("keep project A"))
            XCTAssertEqual(request.machboost?.memory, "off")
            let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
            if requiresReasoning {
                XCTAssertEqual(encoded["think"] as? String, "low")
            } else {
                XCTAssertEqual(encoded["think"] as? Bool, false)
            }
        }
    }

    func testSummaryReconcilesFinalContentInsteadOfLosingLeadingText() throws {
        var stream = ConversationSummaryStream()
        try stream.absorb(event(content: "ep the API", done: false))
        try stream.absorb(ChatEvent(
            requestID: "summary-test", message: .init(role: "assistant", content: ""),
            done: true, doneReason: "stop",
            totalDuration: nil, evalDuration: nil, evalCount: nil,
            machboost: .init(backend: "mlx", stats: nil, timeToFirstTokenSeconds: nil,
                            fullContent: "Keep the API"), error: nil
        ))
        XCTAssertEqual(try stream.result(), "Keep the API")
    }

    func testSummaryCollectsChunksForBackendsWithoutFullContent() throws {
        var stream = ConversationSummaryStream()
        try stream.absorb(event(content: "Keep ", done: false))
        try stream.absorb(event(content: "the API.", done: true))
        XCTAssertEqual(try stream.result(), "Keep the API.")
    }

    func testSummaryRejectsHiddenOnlyOutput() throws {
        var stream = ConversationSummaryStream()
        try stream.absorb(ChatEvent(
            requestID: "summary-test", message: .init(role: "assistant", content: "", thinking: "thinking"),
            done: true, doneReason: "stop", totalDuration: nil, evalDuration: nil,
            evalCount: nil, machboost: nil, error: nil
        ))
        XCTAssertThrowsError(try stream.result()) { error in
            XCTAssertTrue(error.localizedDescription.contains("no summary text"))
        }
    }

    func testSummaryRejectsIncompleteAndTruncatedStreams() throws {
        var incomplete = ConversationSummaryStream()
        try incomplete.absorb(event(content: "Partial", done: false))
        XCTAssertThrowsError(try incomplete.result())
        for reason in ["length", "max_tokens"] {
            var truncated = ConversationSummaryStream()
            try truncated.absorb(event(content: "Partial", done: true, reason: reason))
            XCTAssertThrowsError(try truncated.result())
        }
    }

    func testSummaryCancellationDoesNotProduceAResult() throws {
        var stream = ConversationSummaryStream()
        try stream.absorb(event(content: "Partial", done: true, reason: "cancelled"))
        XCTAssertThrowsError(try stream.result()) { XCTAssertTrue($0 is CancellationError) }
    }

    func testSummarySurfacesBackendErrors() {
        var stream = ConversationSummaryStream()
        XCTAssertThrowsError(try stream.absorb(ChatEvent(
            requestID: "summary-test", message: nil, done: true, doneReason: nil,
            totalDuration: nil, evalDuration: nil, evalCount: nil, machboost: nil,
            error: "Model unavailable"
        )))
    }

    func testSummaryPreservesToolResultsAndChangedPaths() throws {
        let message = ChatMessage(role: .assistant, content: "Updated the timeout.")
        var activity = CodingToolActivity(call: .init(function: .init(
            name: "edit_file", arguments: .object([:])
        )), state: .succeeded)
        activity.output = "timeout changed from 10 to 30"
        activity.changedPath = "src/client.py"
        message.toolActivityJSON = String(decoding: try JSONEncoder().encode([activity]), as: UTF8.self)
        let transcript = ConversationCompaction.transcript([message])
        XCTAssertTrue(transcript.contains("src/client.py"))
        XCTAssertTrue(transcript.contains("timeout changed from 10 to 30"))
        XCTAssertTrue(transcript.contains("succeeded"))
    }

    func testSummaryRedactsCredentialShapesBeforeAndAfterInference() throws {
        let input = "Authorization: Bearer secret-token API_KEY=sk_abcdefgh1234 team=mbk_abcdefgh1234"
        let redacted = ConversationCompaction.redactSecrets(input)
        XCTAssertFalse(redacted.contains("secret-token"))
        XCTAssertFalse(redacted.contains("sk_abcdefgh1234"))
        XCTAssertFalse(redacted.contains("mbk_abcdefgh1234"))

        var stream = ConversationSummaryStream()
        try stream.absorb(event(content: "Keep API key: hf_abcdefghijkl", done: true))
        XCTAssertEqual(try stream.result(), "Keep API key: [REDACTED]")
    }

    func testSavedSummaryRetainsHistoryButExcludesOldTurnsFromContext() throws {
        let schema = Schema([Conversation.self, ChatMessage.self, ChatAttachment.self])
        let container = try ModelContainer(for: schema, configurations: .init(schema: schema, isStoredInMemoryOnly: true))
        let conversation = Conversation()
        container.mainContext.insert(conversation)
        let messages = exchange("Remember code 73", "Remembered") + exchange("What was the code?", "73")
        for (index, message) in messages.enumerated() {
            message.createdAt = Date(timeIntervalSince1970: Double(index))
            message.conversation = conversation
        }
        conversation.messages = messages
        conversation.contextSummary = "The code is 73."
        conversation.summarizedThrough = messages[1].createdAt
        conversation.summaryUpdatedAt = .now
        try container.mainContext.save()

        let reloaded = try ModelContext(container).fetch(FetchDescriptor<Conversation>()).first!
        XCTAssertEqual(reloaded.messages.count, 4)
        XCTAssertEqual(reloaded.contextSummary, "The code is 73.")
        XCTAssertEqual(ConversationCompaction.activeMessages(in: reloaded).map(\.content), ["What was the code?", "73"])
    }

    func testEditingSummarizedTurnInvalidatesStaleSummary() {
        let conversation = Conversation()
        conversation.contextSummary = "Old value"
        conversation.summarizedThrough = Date(timeIntervalSince1970: 10)
        conversation.summaryUpdatedAt = .now
        ConversationCompaction.invalidateSummary(in: conversation, editingFrom: Date(timeIntervalSince1970: 11))
        XCTAssertNotNil(conversation.contextSummary)
        ConversationCompaction.invalidateSummary(in: conversation, editingFrom: Date(timeIntervalSince1970: 10))
        XCTAssertNil(conversation.contextSummary)
        XCTAssertNil(conversation.summarizedThrough)
    }

    private func exchange(_ user: String, _ assistant: String) -> [ChatMessage] {
        [ChatMessage(role: .user, content: user), ChatMessage(role: .assistant, content: assistant)]
    }

    private func event(content: String, done: Bool, reason: String = "stop") -> ChatEvent {
        ChatEvent(requestID: "summary-test", message: .init(role: "assistant", content: content),
                  done: done, doneReason: done ? reason : nil, totalDuration: nil,
                  evalDuration: nil, evalCount: nil, machboost: nil, error: nil)
    }
}
