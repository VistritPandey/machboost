import Foundation
import SwiftData

enum SidebarDestination: Hashable {
    case conversation(UUID)
    case models
    case server
    case settings
}

enum MessageRole: String, Codable {
    case system
    case user
    case assistant
}

enum AttachmentKind: String, Codable {
    case text
    case image
}

@Model
final class Conversation {
    @Attribute(.unique) var id: UUID
    var title: String
    var model: String
    var createdAt: Date
    var updatedAt: Date
    @Relationship(deleteRule: .cascade, inverse: \ChatMessage.conversation)
    var messages: [ChatMessage]
    @Relationship(deleteRule: .cascade, inverse: \ChatAttachment.conversation)
    var attachments: [ChatAttachment]

    init(
        id: UUID = UUID(),
        title: String = "New chat",
        model: String = "llama3.2:3b",
        createdAt: Date = .now
    ) {
        self.id = id
        self.title = title
        self.model = model
        self.createdAt = createdAt
        self.updatedAt = createdAt
        self.messages = []
        self.attachments = []
    }

    var orderedMessages: [ChatMessage] {
        messages.sorted { $0.createdAt < $1.createdAt }
    }

    var orderedAttachments: [ChatAttachment] {
        attachments.sorted { $0.createdAt < $1.createdAt }
    }
}

@Model
final class ChatMessage {
    @Attribute(.unique) var id: UUID
    var roleValue: String
    var content: String
    var createdAt: Date
    var durationSeconds: Double?
    var timeToFirstTokenSeconds: Double?
    var generatedTokens: Int?
    var tokensPerSecond: Double?
    var wasCancelled: Bool
    var conversation: Conversation?

    init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        createdAt: Date = .now,
        conversation: Conversation? = nil
    ) {
        self.id = id
        self.roleValue = role.rawValue
        self.content = content
        self.createdAt = createdAt
        self.wasCancelled = false
        self.conversation = conversation
    }

    var role: MessageRole {
        get { MessageRole(rawValue: roleValue) ?? .user }
        set { roleValue = newValue.rawValue }
    }
}

@Model
final class ChatAttachment {
    @Attribute(.unique) var id: UUID
    var kindValue: String
    var displayName: String
    var importedPath: String
    var sourcePath: String
    var byteCount: Int64
    var createdAt: Date
    var conversation: Conversation?

    init(
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

    var kind: AttachmentKind {
        get { AttachmentKind(rawValue: kindValue) ?? .text }
        set { kindValue = newValue.rawValue }
    }
}

struct ServerConfiguration: Codable, Equatable {
    var lanEnabled = false
    var port = 11_435
    var replicas = 1
    var maxQueue = 64
    var queueTimeout = 300.0
    var launchAtLogin = false

    var bindHost: String { lanEnabled ? "0.0.0.0" : "127.0.0.1" }
    var clientHost: String { "127.0.0.1" }
    var endpoint: URL { URL(string: "http://\(clientHost):\(port)")! }
    var advertisedEndpoint: URL {
        guard lanEnabled else { return endpoint }
        var components = URLComponents()
        components.scheme = "http"
        components.host = ProcessInfo.processInfo.hostName
        components.port = port
        return components.url ?? endpoint
    }
}

struct CatalogResponse: Decodable {
    let schema: String
    let models: [CatalogModel]
}

struct CatalogModel: Codable, Identifiable, Hashable {
    let name: String
    let displayName: String
    let repository: String?
    let backend: String
    let capabilities: [String]
    let cached: Bool
    let cachedPath: String?
    let recommended: Bool
    let tested: Bool
    let downloadSizeGB: Double?
    let minimumMemoryGB: Double?
    let support: String

    var id: String { name }
    var supportsVision: Bool { capabilities.contains("vision") }

    enum CodingKeys: String, CodingKey {
        case name
        case displayName = "display_name"
        case repository
        case backend
        case capabilities
        case cached
        case cachedPath = "cached_path"
        case recommended
        case tested
        case downloadSizeGB = "download_size_gb"
        case minimumMemoryGB = "minimum_memory_gb"
        case support
    }
}

struct APIChatMessage: Encodable, Sendable {
    let role: String
    let content: String
    let images: [String]?
}

struct ChatRequest: Encodable, Sendable {
    let requestID: String
    let model: String
    let messages: [APIChatMessage]
    let context: [String]
    let stream = true
    let keepAlive = "forever"
    let options: Options

    struct Options: Encodable, Sendable {
        var maxTokens: Int
        var temperature: Double
        var affinityKey: String?

        enum CodingKeys: String, CodingKey {
            case maxTokens = "num_predict"
            case temperature
            case affinityKey = "affinity_key"
        }
    }

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case model
        case messages
        case context
        case stream
        case keepAlive = "keep_alive"
        case options
    }
}

struct ChatEvent: Decodable, Sendable {
    struct Message: Decodable, Sendable {
        let role: String?
        let content: String
    }

    struct MachBoost: Decodable, Sendable {
        let backend: String?
        let stats: GenerationStats?
        let timeToFirstTokenSeconds: Double?

        enum CodingKeys: String, CodingKey {
            case backend
            case stats
            case timeToFirstTokenSeconds = "time_to_first_token_seconds"
        }
    }

    let requestID: String?
    let message: Message?
    let done: Bool
    let doneReason: String?
    let totalDuration: Int64?
    let evalDuration: Int64?
    let evalCount: Int?
    let machboost: MachBoost?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case message
        case done
        case doneReason = "done_reason"
        case totalDuration = "total_duration"
        case evalDuration = "eval_duration"
        case evalCount = "eval_count"
        case machboost
        case error
    }
}

struct GenerationStats: Decodable, Sendable {
    let generatedTokens: Int?
    let generationSeconds: Double?
    let promptTokens: Int?

    enum CodingKeys: String, CodingKey {
        case generatedTokens = "generated_tokens"
        case generationSeconds = "generation_seconds"
        case promptTokens = "prompt_tokens"
    }
}

struct PullEvent: Decodable, Sendable {
    let requestID: String?
    let status: String?
    let file: String?
    let completed: Int64?
    let total: Int64?
    let unit: String?
    let done: Bool
    let path: String?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case status
        case file
        case completed
        case total
        case unit
        case done
        case path
        case error
    }
}

struct ModelInstance: Decodable, Identifiable {
    struct Scheduler: Decodable {
        let replicas: Int
        let activeRequests: Int
        let queuedRequests: Int

        enum CodingKeys: String, CodingKey {
            case replicas
            case activeRequests = "active_requests"
            case queuedRequests = "queued_requests"
        }
    }

    let model: String
    let backend: String
    let idleSeconds: Double
    let keepAliveSeconds: Double
    let requests: Int
    let capabilities: [String]
    let scheduler: Scheduler

    var id: String { "\(model)-\(backend)" }

    enum CodingKeys: String, CodingKey {
        case model
        case backend
        case idleSeconds = "idle_seconds"
        case keepAliveSeconds = "keep_alive_seconds"
        case requests
        case capabilities
        case scheduler
    }
}

struct ModelsResponse: Decodable {
    let models: [ModelInstance]
}

struct ServerMetrics: Decodable {
    struct Operations: Decodable {
        struct Totals: Decodable {
            let started: Int
            let completed: Int
            let cancelled: Int
            let failed: Int
            let generatedTokens: Int

            enum CodingKeys: String, CodingKey {
                case started
                case completed
                case cancelled
                case failed
                case generatedTokens = "generated_tokens"
            }
        }

        let activeCount: Int
        let totals: Totals
        let generationTokensPerSecond: Double

        enum CodingKeys: String, CodingKey {
            case activeCount = "active_count"
            case totals
            case generationTokensPerSecond = "generation_tokens_per_second"
        }
    }

    struct Scheduler: Decodable {
        let activeRequests: Int
        let queuedRequests: Int
        let rejectedRequests: Int

        enum CodingKeys: String, CodingKey {
            case activeRequests = "active_requests"
            case queuedRequests = "queued_requests"
            case rejectedRequests = "rejected_requests"
        }
    }

    struct ProcessInfo: Decodable {
        let peakResidentMemoryBytes: Int64

        enum CodingKeys: String, CodingKey {
            case peakResidentMemoryBytes = "peak_resident_memory_bytes"
        }
    }

    let schema: String
    let operations: Operations
    let models: [ModelInstance]
    let scheduler: Scheduler
    let process: ProcessInfo
}

struct ModelPreflightResponse: Decodable {
    struct Preflight: Decodable {
        let model: String
        let backend: String
        let capabilities: [String]
        let runtimeAvailable: Bool
        let cached: Bool
        let modelType: String?
        let supported: Bool
        let reason: String?

        enum CodingKeys: String, CodingKey {
            case model
            case backend
            case capabilities
            case runtimeAvailable = "runtime_available"
            case cached
            case modelType = "model_type"
            case supported
            case reason
        }
    }

    let preflight: Preflight
}
