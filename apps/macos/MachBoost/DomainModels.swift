import Foundation
import MachBoostDaemonClient
import MachBoostPersistence

enum SidebarDestination: Hashable {
    case conversation(UUID)
    case models
    case server
    case settings
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
