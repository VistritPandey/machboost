import AppKit
import SwiftUI

struct MessageContentView: View {
    let content: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(markdownBlocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case let .prose(text):
                    Text(markdown: text)
                        .textSelection(.enabled)
                        .lineSpacing(3)
                case let .code(language, code):
                    CodeBlockView(language: language, code: code)
                }
            }
        }
    }

    private var markdownBlocks: [MarkdownBlock] {
        MarkdownBlock.parse(content)
    }
}

private struct CodeBlockView: View {
    let language: String
    let code: String

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(language.isEmpty ? "Code" : language)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.plain)
                .help("Copy code")
            }
            .padding(.horizontal, 10)
            .frame(height: 30)
            .background(Color(nsColor: .controlBackgroundColor))

            Divider()

            ScrollView(.horizontal) {
                Text(code)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(nsColor: .textBackgroundColor))
        }
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        }
    }
}

private enum MarkdownBlock {
    case prose(String)
    case code(language: String, code: String)

    static func parse(_ source: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var prose: [String] = []
        var code: [String] = []
        var language = ""
        var insideCode = false

        func flushProse() {
            let value = prose.joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty { blocks.append(.prose(value)) }
            prose.removeAll(keepingCapacity: true)
        }

        func flushCode() {
            blocks.append(.code(language: language, code: code.joined(separator: "\n")))
            code.removeAll(keepingCapacity: true)
            language = ""
        }

        for line in source.components(separatedBy: .newlines) {
            if line.hasPrefix("```") {
                if insideCode {
                    flushCode()
                } else {
                    flushProse()
                    language = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                }
                insideCode.toggle()
            } else if insideCode {
                code.append(line)
            } else {
                prose.append(line)
            }
        }
        if insideCode {
            flushCode()
        } else {
            flushProse()
        }
        return blocks.isEmpty ? [.prose(source)] : blocks
    }
}

private extension Text {
    init(markdown: String) {
        if let attributed = try? AttributedString(
            markdown: markdown,
            options: .init(interpretedSyntax: .full)
        ) {
            self.init(attributed)
        } else {
            self.init(markdown)
        }
    }
}
