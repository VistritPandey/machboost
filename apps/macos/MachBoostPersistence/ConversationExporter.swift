import Foundation

public enum ConversationExporter {
    public static func markdown(_ conversation: Conversation) -> String {
        var lines = [
            "# \(conversation.title)",
            "",
            "Model: `\(conversation.model)`",
            "",
        ]
        if !conversation.attachments.isEmpty {
            lines.append(
                "Context: \(conversation.orderedAttachments.map(\.displayName).joined(separator: ", "))"
            )
            lines.append("")
        }
        for message in conversation.orderedMessages {
            lines.append("## \(message.role == .user ? "User" : "Assistant")")
            lines.append("")
            if let reasoning = message.reasoningContent, !reasoning.isEmpty {
                lines.append("<details>")
                lines.append("<summary>Reasoning</summary>")
                lines.append("")
                lines.append(reasoning)
                lines.append("")
                lines.append("</details>")
                lines.append("")
            }
            lines.append(message.content)
            lines.append("")
            if let toolCalls = message.toolCallsJSON, !toolCalls.isEmpty {
                lines.append("### Tool calls")
                lines.append("")
                lines.append("```json")
                lines.append(toolCalls)
                lines.append("```")
                lines.append("")
            }
        }
        return lines.joined(separator: "\n")
    }

    public static func fileName(for conversation: Conversation) -> String {
        let invalid = CharacterSet(charactersIn: "/:")
        let title = conversation.title
            .components(separatedBy: invalid)
            .joined(separator: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return "\(title.isEmpty ? "conversation" : title).md"
    }
}
