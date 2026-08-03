import AppKit
import SwiftUI

struct MessageContentView: View {
    let content: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(markdownBlocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case let .paragraph(text):
                    prose(text)
                case let .heading(level, text):
                    Text(markdown: text)
                        .font(headingFont(level))
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                case let .bullet(text):
                    HStack(alignment: .firstTextBaseline, spacing: 9) {
                        Image(systemName: "circle.fill")
                            .font(.system(size: 5))
                            .frame(width: 10)
                        prose(text)
                    }
                case let .numbered(marker, text):
                    HStack(alignment: .firstTextBaseline, spacing: 9) {
                        Text(marker)
                            .font(.body.monospacedDigit())
                            .foregroundStyle(.secondary)
                            .frame(minWidth: 22, alignment: .trailing)
                        prose(text)
                    }
                case let .quote(text):
                    HStack(alignment: .top, spacing: 10) {
                        Rectangle()
                            .fill(Color.teal)
                            .frame(width: 3)
                        prose(text)
                            .foregroundStyle(.secondary)
                    }
                case let .code(language, code):
                    CodeBlockView(language: language, code: code)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var markdownBlocks: [MarkdownBlock] {
        MarkdownBlock.parse(content)
    }

    private func prose(_ text: String) -> some View {
        Text(markdown: text)
            .font(.body)
            .textSelection(.enabled)
            .lineSpacing(4)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: .title2.weight(.semibold)
        case 2: .title3.weight(.semibold)
        default: .headline
        }
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
                    .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(nsColor: .textBackgroundColor))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        }
    }
}

private enum MarkdownBlock {
    case paragraph(String)
    case heading(level: Int, text: String)
    case bullet(String)
    case numbered(marker: String, text: String)
    case quote(String)
    case code(language: String, code: String)

    static func parse(_ source: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var prose: [String] = []
        var code: [String] = []
        var language = ""
        var insideCode = false

        func flushProse() {
            appendProse(prose, to: &blocks)
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
        return blocks.isEmpty ? [.paragraph(source)] : blocks
    }

    private static func appendProse(
        _ lines: [String],
        to blocks: inout [MarkdownBlock]
    ) {
        var paragraph: [String] = []

        func flushParagraph() {
            let text = paragraph.joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { blocks.append(.paragraph(text)) }
            paragraph.removeAll(keepingCapacity: true)
        }

        for rawLine in lines {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty {
                flushParagraph()
                continue
            }

            let headingMarks = line.prefix { $0 == "#" }.count
            if (1...6).contains(headingMarks) {
                let remainder = line.dropFirst(headingMarks)
                if remainder.first == " " {
                    flushParagraph()
                    blocks.append(
                        .heading(
                            level: headingMarks,
                            text: remainder.trimmingCharacters(in: .whitespaces)
                        )
                    )
                    continue
                }
            }

            if line.hasPrefix("- ") || line.hasPrefix("* ") || line.hasPrefix("+ ") {
                flushParagraph()
                blocks.append(.bullet(String(line.dropFirst(2))))
                continue
            }

            let pieces = line.split(separator: " ", maxSplits: 1)
            if
                pieces.count == 2,
                pieces[0].hasSuffix("."),
                Int(pieces[0].dropLast()) != nil
            {
                flushParagraph()
                blocks.append(
                    .numbered(marker: String(pieces[0]), text: String(pieces[1]))
                )
                continue
            }

            if line.hasPrefix("> ") {
                flushParagraph()
                blocks.append(.quote(String(line.dropFirst(2))))
                continue
            }

            paragraph.append(line)
        }
        flushParagraph()
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
