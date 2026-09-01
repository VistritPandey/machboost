import Foundation
import MachBoostDaemonClient
import MachBoostPersistence

enum SidebarDestination: Hashable {
    case conversation(UUID)
    case apps
    case connections
    case extensions
    case models
    case server
    case settings
}

enum InferenceMode: String, Codable, CaseIterable, Sendable {
    case local
    case team
}

struct InferenceHostOption: Identifiable, Equatable, Sendable {
    static let automaticID = "automatic"
    static let localID = "local"

    let id: String
    let name: String
    let detail: String
    let isOnline: Bool
    let isLoaded: Bool
}

struct TeamHostProfile: Codable, Identifiable, Equatable, Sendable {
    let id: UUID
    var endpoint: URL
    var hostName: String
    var hostVersion: String
    var principalName: String
    var connectedAt: Date
}

struct TeamHostSnapshot: Identifiable, Sendable {
    let profile: TeamHostProfile
    var catalog: [CatalogModel]
    var loadedModels: [ModelInstance]
    var metrics: ServerMetrics?
    var roundTripSeconds: Double = 0
    var isOnline: Bool
    var lastError: String?
    var updatedAt: Date

    var id: UUID { profile.id }
    var activeRequests: Int { metrics?.scheduler.activeRequests ?? 0 }
    var queuedRequests: Int { metrics?.scheduler.queuedRequests ?? 0 }

    func supports(model: String) -> Bool {
        catalog.contains {
            ($0.name == model || $0.repository == model) && $0.cached && $0.support == "ready"
        }
    }

    func hasLoaded(model: String) -> Bool {
        loadedModels.contains { $0.model == model }
    }

    func scheduler(model: String) -> ModelInstance.Scheduler? {
        loadedModels.first { $0.model == model }?.scheduler
    }
}

enum HostRoutingPolicy {
    static func score(
        metrics: ServerMetrics?,
        modelLoaded: Bool,
        reservedRequests: Int = 0,
        roundTripSeconds: Double = 0,
        replicas: Int = 1,
        activeRequests: Int? = nil,
        queuedRequests: Int? = nil
    ) -> Double {
        let active = max(0, activeRequests ?? metrics?.scheduler.activeRequests ?? 0)
        let queued = max(0, queuedRequests ?? metrics?.scheduler.queuedRequests ?? 0)
        let capacity = max(1, replicas)
        let service = max(0.05, metrics?.operations.latencySeconds.p50 ?? 0.75)
        let demandAhead = queued + max(0, reservedRequests)
        let activeOverCapacity = max(0, active - capacity + 1)
        let queueDelay = Double(demandAhead + activeOverCapacity)
            * service / Double(capacity)
        let coldLoadPenalty = modelLoaded ? 0 : max(2, service * 4)
        return max(0, roundTripSeconds) + service + queueDelay + coldLoadPenalty
    }

    static func canFailOver(error: Error, emittedOutput: Bool) -> Bool {
        guard !emittedOutput else { return false }
        if error is CancellationError {
            return false
        }
        if let urlError = error as? URLError {
            return [
                .timedOut,
                .cannotFindHost,
                .cannotConnectToHost,
                .networkConnectionLost,
                .dnsLookupFailed,
                .notConnectedToInternet,
            ].contains(urlError.code)
        }
        if case let MachBoostAPIError.server(status, _) = error {
            return [408, 425, 429, 500, 502, 503, 504].contains(status)
        }
        if case let MachBoostAPIError.stream(message) = error {
            let normalized = message.lowercased()
            return ["timeout", "connection", "queue", "unavailable", "overloaded"]
                .contains { normalized.contains($0) }
        }
        return false
    }
}

enum AssistantTimelineKind: String, Codable, Sendable {
    case reasoning
    case content
    case tools
}

struct AssistantTimelineEntry: Codable, Identifiable, Sendable {
    var id = UUID()
    var kind: AssistantTimelineKind
    var text = ""
    var activities: [CodingToolActivity] = []

    mutating func append(_ value: String) {
        text += value
    }
}

extension Array where Element == AssistantTimelineEntry {
    mutating func appendText(_ text: String, kind: AssistantTimelineKind) {
        guard !text.isEmpty else { return }
        if !isEmpty, self[index(before: endIndex)].kind == kind {
            self[index(before: endIndex)].append(text)
        } else {
            append(AssistantTimelineEntry(kind: kind, text: text))
        }
    }
}

typealias MessageRole = MachBoostPersistence.MessageRole
typealias AttachmentKind = MachBoostPersistence.AttachmentKind
typealias Conversation = MachBoostPersistence.Conversation
typealias ChatMessage = MachBoostPersistence.ChatMessage
typealias ChatAttachment = MachBoostPersistence.ChatAttachment
typealias AttachmentStore = MachBoostPersistence.AttachmentStore
typealias ConversationExporter = MachBoostPersistence.ConversationExporter

typealias ServerConfiguration = MachBoostDaemonClient.ServerConfiguration
typealias CatalogResponse = MachBoostDaemonClient.CatalogResponse
typealias CatalogModel = MachBoostDaemonClient.CatalogModel
typealias APIChatMessage = MachBoostDaemonClient.APIChatMessage
typealias APIToolCall = MachBoostDaemonClient.APIToolCall
typealias APIToolDefinition = MachBoostDaemonClient.APIToolDefinition
typealias JSONValue = MachBoostDaemonClient.JSONValue
typealias ChatRequest = MachBoostDaemonClient.ChatRequest
typealias ChatEvent = MachBoostDaemonClient.ChatEvent
typealias GenerationStats = MachBoostDaemonClient.GenerationStats
typealias PullEvent = MachBoostDaemonClient.PullEvent
typealias ModelInstance = MachBoostDaemonClient.ModelInstance
typealias ModelsResponse = MachBoostDaemonClient.ModelsResponse
typealias ModelLoadResponse = MachBoostDaemonClient.ModelLoadResponse
typealias ServerMetrics = MachBoostDaemonClient.ServerMetrics
typealias TeamStatus = MachBoostDaemonClient.TeamStatus
typealias TeamSettings = MachBoostDaemonClient.TeamSettings
typealias TeamKey = MachBoostDaemonClient.TeamKey
typealias TeamConnectResponse = MachBoostDaemonClient.TeamConnectResponse
typealias TeamClient = MachBoostDaemonClient.TeamClient
typealias TeamModelRequest = MachBoostDaemonClient.TeamModelRequest
typealias TraceSummary = MachBoostDaemonClient.TraceSummary
typealias TraceEvaluation = MachBoostDaemonClient.TraceEvaluation
typealias MemorySummary = MachBoostDaemonClient.MemorySummary
typealias MemoriesResponse = MachBoostDaemonClient.MemoriesResponse
typealias CacheMetrics = MachBoostDaemonClient.CacheMetrics
typealias ProviderSummary = MachBoostDaemonClient.ProviderSummary
typealias ProvidersResponse = MachBoostDaemonClient.ProvidersResponse
typealias WorkspaceSummary = MachBoostDaemonClient.WorkspaceSummary
typealias WorkspaceResult = MachBoostDaemonClient.WorkspaceResult
typealias ModelPreflightResponse = MachBoostDaemonClient.ModelPreflightResponse
typealias MachBoostAPIError = MachBoostDaemonClient.MachBoostAPIError
typealias MachBoostAPIProtocol = MachBoostDaemonClient.MachBoostAPIProtocol
typealias MachBoostAPI = MachBoostDaemonClient.MachBoostAPI
