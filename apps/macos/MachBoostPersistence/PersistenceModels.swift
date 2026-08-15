import Foundation
import SwiftData

public enum MessageRole: String, Codable, Sendable {
    case system
    case user
    case assistant
}

public enum AttachmentKind: String, Codable, Sendable {
    case text
    case image
}

@Model
public final class Conversation {
    @Attribute(.unique) public var id: UUID
    public var title: String
    public var model: String
    public var workspaceID: String?
    public var contextSummary: String?
    public var summarizedThrough: Date?
    public var summaryUpdatedAt: Date?
    public var createdAt: Date
    public var updatedAt: Date
    @Relationship(deleteRule: .cascade, inverse: \ChatMessage.conversation)
    public var messages: [ChatMessage]
    @Relationship(deleteRule: .cascade, inverse: \ChatAttachment.conversation)
    public var attachments: [ChatAttachment]

    public init(
        id: UUID = UUID(),
        title: String = "New chat",
        model: String = "llama3.2:3b",
        workspaceID: String? = nil,
        createdAt: Date = .now
    ) {
        self.id = id
        self.title = title
        self.model = model
        self.workspaceID = workspaceID
        self.contextSummary = nil
        self.summarizedThrough = nil
        self.summaryUpdatedAt = nil
        self.createdAt = createdAt
        self.updatedAt = createdAt
        self.messages = []
        self.attachments = []
    }

    public var orderedMessages: [ChatMessage] {
        messages.sorted { $0.createdAt < $1.createdAt }
    }

    public var orderedAttachments: [ChatAttachment] {
        attachments.sorted { $0.createdAt < $1.createdAt }
    }
}

@Model
public final class ChatMessage {
    @Attribute(.unique) public var id: UUID
    public var roleValue: String
    public var content: String
    public var reasoningContent: String?
    public var toolCallsJSON: String?
    public var createdAt: Date
    public var durationSeconds: Double?
    public var timeToFirstTokenSeconds: Double?
    public var generatedTokens: Int?
    public var tokensPerSecond: Double?
    public var wasCancelled: Bool
    public var conversation: Conversation?

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        reasoningContent: String? = nil,
        toolCallsJSON: String? = nil,
        createdAt: Date = .now,
        conversation: Conversation? = nil
    ) {
        self.id = id
        self.roleValue = role.rawValue
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCallsJSON = toolCallsJSON
        self.createdAt = createdAt
        self.wasCancelled = false
        self.conversation = conversation
    }

    public var role: MessageRole {
        get { MessageRole(rawValue: roleValue) ?? .user }
        set { roleValue = newValue.rawValue }
    }
}

@Model
public final class ChatAttachment {
    @Attribute(.unique) public var id: UUID
    public var kindValue: String
    public var displayName: String
    public var importedPath: String
    public var sourcePath: String
    public var byteCount: Int64
    public var createdAt: Date
    public var conversation: Conversation?

    public init(
        id: UUID = UUID(),
        kind: AttachmentKind,
        displayName: String,
        importedPath: String,
        sourcePath: String,
        byteCount: Int64,
        createdAt: Date = .now,
        conversation: Conversation? = nil
    ) {
        self.id = id
        self.kindValue = kind.rawValue
        self.displayName = displayName
        self.importedPath = importedPath
        self.sourcePath = sourcePath
        self.byteCount = byteCount
        self.createdAt = createdAt
        self.conversation = conversation
    }

    public var kind: AttachmentKind {
        get { AttachmentKind(rawValue: kindValue) ?? .text }
        set { kindValue = newValue.rawValue }
    }
}
