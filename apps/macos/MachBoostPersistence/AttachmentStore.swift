import CryptoKit
import Foundation
import UniformTypeIdentifiers

public enum AttachmentStore {
    public static let maximumTextFileBytes: Int64 = 2 * 1_024 * 1_024
    public static let maximumImageBytes: Int64 = 25 * 1_024 * 1_024
    public static let maximumFolderFiles = 200

    public static func importURLs(
        _ urls: [URL],
        conversation: Conversation
    ) throws -> [ChatAttachment] {
        var candidates: [URL] = []
        for url in urls {
            if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                candidates.append(contentsOf: try textFiles(in: url))
            } else {
                candidates.append(url)
            }
        }
        var importedPaths = Set(conversation.attachments.map(\.importedPath))
        var imported: [ChatAttachment] = []
        for candidate in candidates {
            let attachment = try importFile(candidate, conversation: conversation)
            if importedPaths.insert(attachment.importedPath).inserted {
                imported.append(attachment)
            }
        }
        return imported
    }

    public static func remove(_ attachment: ChatAttachment) {
        let path = URL(fileURLWithPath: attachment.importedPath)
        try? FileManager.default.removeItem(at: path)
    }

    private static func importFile(
        _ source: URL,
        conversation: Conversation
    ) throws -> ChatAttachment {
        let accessing = source.startAccessingSecurityScopedResource()
        defer {
            if accessing {
                source.stopAccessingSecurityScopedResource()
            }
        }

        let kind = try attachmentKind(for: source)
        let values = try source.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true else {
            throw AttachmentError.unsupported(source.lastPathComponent)
        }
        let byteCount = Int64(values.fileSize ?? 0)
        let maximum = kind == .image ? maximumImageBytes : maximumTextFileBytes
        guard byteCount <= maximum else {
            throw AttachmentError.tooLarge(
                name: source.lastPathComponent,
                maximumBytes: maximum
            )
        }

        let data = try Data(contentsOf: source, options: .mappedIfSafe)
        if kind == .text, String(data: data, encoding: .utf8) == nil {
            throw AttachmentError.notUTF8(source.lastPathComponent)
        }
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let extensionSuffix = source.pathExtension.isEmpty ? "" : ".\(source.pathExtension.lowercased())"
        let destination = try attachmentDirectory(conversationID: conversation.id)
            .appendingPathComponent("\(digest)\(extensionSuffix)", isDirectory: false)
        if !FileManager.default.fileExists(atPath: destination.path) {
            try data.write(to: destination, options: .atomic)
        }
        return ChatAttachment(
            kind: kind,
            displayName: source.lastPathComponent,
            importedPath: destination.path,
            sourcePath: source.path,
            byteCount: byteCount,
            conversation: conversation
        )
    }

    private static func attachmentKind(for url: URL) throws -> AttachmentKind {
        let type = UTType(filenameExtension: url.pathExtension)
        if type?.conforms(to: .image) == true {
            return .image
        }
        if
            type?.conforms(to: .plainText) == true
                || type?.conforms(to: .sourceCode) == true
                || textExtensions.contains(url.pathExtension.lowercased())
        {
            return .text
        }
        throw AttachmentError.unsupported(url.lastPathComponent)
    }

    private static func textFiles(in directory: URL) throws -> [URL] {
        let accessing = directory.startAccessingSecurityScopedResource()
        defer {
            if accessing {
                directory.stopAccessingSecurityScopedResource()
            }
        }
        let keys: [URLResourceKey] = [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey]
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return []
        }
        var files: [URL] = []
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: Set(keys))
            if values.isSymbolicLink == true {
                if values.isDirectory == true {
                    enumerator.skipDescendants()
                }
                continue
            }
            guard values.isRegularFile == true else { continue }
            guard (try? attachmentKind(for: url)) == .text else { continue }
            files.append(url)
            if files.count >= maximumFolderFiles {
                break
            }
        }
        return files.sorted { $0.path < $1.path }
    }

    private static func attachmentDirectory(conversationID: UUID) throws -> URL {
        let root = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        let directory = root
            .appendingPathComponent("MachBoost", isDirectory: true)
            .appendingPathComponent("Attachments", isDirectory: true)
            .appendingPathComponent(conversationID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }

    private static let textExtensions: Set<String> = [
        "c", "cc", "conf", "cpp", "css", "csv", "go", "h", "hpp", "html",
        "ini", "java", "js", "json", "jsx", "kt", "log", "md", "mjs", "mm",
        "php", "plist", "py", "rb", "rs", "sh", "sql", "swift", "toml", "ts",
        "tsx", "txt", "xml", "yaml", "yml",
    ]
}

public enum AttachmentError: LocalizedError {
    case unsupported(String)
    case tooLarge(name: String, maximumBytes: Int64)
    case notUTF8(String)

    public var errorDescription: String? {
        switch self {
        case let .unsupported(name):
            "\(name) is not a supported text, code, or image file."
        case let .tooLarge(name, maximumBytes):
            "\(name) exceeds the \(ByteCountFormatter.string(fromByteCount: maximumBytes, countStyle: .file)) limit."
        case let .notUTF8(name):
            "\(name) is not UTF-8 text."
        }
    }
}
