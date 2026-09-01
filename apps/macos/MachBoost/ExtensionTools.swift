import Foundation
import MachBoostDaemonClient

@MainActor
enum ExtensionTools {
    static let names: Set<String> = ["search_mcp_tools", "call_mcp_tool"]

    static let definitions: [APIToolDefinition] = [
        APIToolDefinition(
            function: .init(
                name: "search_mcp_tools",
                description: "Find tools exposed by the user's enabled MCP connectors.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object(["type": .string("string")]),
                        "limit": .object([
                            "type": .string("integer"),
                            "minimum": .number(1),
                            "maximum": .number(25),
                        ]),
                    ]),
                    "required": .array([.string("query")]),
                ])
            )
        ),
        APIToolDefinition(
            function: .init(
                name: "call_mcp_tool",
                description: "Call one MCP tool returned by search_mcp_tools.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "server_id": .object(["type": .string("string")]),
                        "name": .object(["type": .string("string")]),
                        "arguments": .object(["type": .string("object")]),
                    ]),
                    "required": .array([
                        .string("server_id"), .string("name"), .string("arguments"),
                    ]),
                ])
            )
        ),
    ]

    static func supports(_ call: APIToolCall) -> Bool {
        names.contains(call.function.name)
    }

    static func requiresApproval(_ call: APIToolCall) -> Bool {
        call.function.name == "call_mcp_tool"
    }

    static func execute(_ call: APIToolCall, appState: AppState) async throws -> CodingToolResult {
        let arguments = object(call.function.arguments)
        switch call.function.name {
        case "search_mcp_tools":
            let query = try requiredString(arguments, "query")
            let limit = boundedInt(arguments["limit"], default: 8, range: 1 ... 25)
            let tools = try await appState.searchMCPTools(query: query, limit: limit)
            let data = try JSONEncoder().encode(tools)
            return CodingToolResult(
                callID: call.id,
                name: call.function.name,
                content: String(decoding: data, as: UTF8.self),
                changedPath: nil,
                changePatch: nil
            )
        case "call_mcp_tool":
            let serverID = try requiredString(arguments, "server_id")
            let name = try requiredString(arguments, "name")
            let toolArguments = arguments["arguments"] ?? .object([:])
            guard case .object = toolArguments else {
                throw CodingWorkspaceError.invalidArguments("arguments must be a JSON object")
            }
            let result = try await appState.callMCPTool(
                serverID: serverID,
                name: name,
                arguments: toolArguments
            )
            let content = json([
                "server": result.serverName,
                "tool": result.tool,
                "is_error": result.isError,
                "content": result.text,
            ])
            return CodingToolResult(
                callID: call.id,
                name: call.function.name,
                content: content,
                changedPath: nil,
                changePatch: nil
            )
        default:
            throw CodingWorkspaceError.invalidArguments("Unknown extension tool: \(call.function.name)")
        }
    }

    private static func object(_ value: JSONValue?) -> [String: JSONValue] {
        guard case let .object(object) = value else { return [:] }
        return object
    }

    private static func string(_ value: JSONValue?) -> String? {
        guard case let .string(string) = value else { return nil }
        return string
    }

    private static func requiredString(
        _ object: [String: JSONValue],
        _ key: String
    ) throws -> String {
        guard let value = string(object[key])?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty else {
            throw CodingWorkspaceError.invalidArguments("\(key) is required")
        }
        return value
    }

    private static func boundedInt(
        _ value: JSONValue?,
        default defaultValue: Int,
        range: ClosedRange<Int>
    ) -> Int {
        guard case let .number(number) = value else { return defaultValue }
        return min(max(Int(number), range.lowerBound), range.upperBound)
    }

    private static func json(_ object: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys, .withoutEscapingSlashes]
        ) else { return #"{"error":"tool result could not be encoded"}"# }
        return String(decoding: data, as: UTF8.self)
    }
}
