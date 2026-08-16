import Foundation
import MachBoostDaemonClient

enum CodingWorkspaceError: LocalizedError {
    case invalidArguments(String)
    case pathOutsideWorkspace
    case pathNotFound(String)
    case unsupportedFile(String)
    case ambiguousReplacement(Int)
    case fileAlreadyExists(String)

    var errorDescription: String? {
        switch self {
        case let .invalidArguments(message): message
        case .pathOutsideWorkspace: "The requested path is outside the selected repository."
        case let .pathNotFound(path): "No repository file exists at \(path)."
        case let .unsupportedFile(path): "\(path) is not a readable UTF-8 text file."
        case let .ambiguousReplacement(count):
            "The old text must occur exactly once; MachBoost found \(count) matches."
        case let .fileAlreadyExists(path): "A file already exists at \(path)."
        }
    }
}

struct CodingToolResult: Sendable {
    let callID: String?
    let name: String
    let content: String
    let changedPath: String?
    let changePatch: String?
}

enum CodingToolState: String, Codable, Hashable, Sendable {
    case requested
    case queued
    case running
    case succeeded
    case denied
    case failed
}

struct CodingToolActivity: Codable, Hashable, Identifiable, Sendable {
    let id: String
    let call: APIToolCall
    var state: CodingToolState
    var output: String?
    var changedPath: String?
    var changePatch: String?

    init(call: APIToolCall, state: CodingToolState = .queued) {
        self.id = call.id ?? "activity-\(UUID().uuidString.lowercased())"
        self.call = call
        self.state = state
        self.output = nil
        self.changedPath = nil
        self.changePatch = nil
    }
}

enum CodingPermissionMode: String, CaseIterable, Identifiable, Sendable {
    case automatic
    case manual
    case acceptEdits
    case plan
    case bypass

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic: "Auto"
        case .manual: "Manual"
        case .acceptEdits: "Accept edits"
        case .plan: "Plan"
        case .bypass: "Bypass permissions"
        }
    }

    var subtitle: String {
        switch self {
        case .automatic: "Approve small edits; ask before broader changes"
        case .manual: "Always ask before changing files"
        case .acceptEdits: "Automatically accept repository file edits"
        case .plan: "Inspect and propose changes without editing"
        case .bypass: "Approve every tool inside the selected repository"
        }
    }

    var icon: String {
        switch self {
        case .automatic: "wand.and.sparkles"
        case .manual: "hand.raised"
        case .acceptEdits: "checkmark.square"
        case .plan: "list.bullet.clipboard"
        case .bypass: "bolt.shield"
        }
    }
}

enum CodingPermissionDecision: Equatable, Sendable {
    case allow
    case ask
    case deny(String)
}

enum CodingWorkspace {
    static let maximumToolRounds = 8
    static let mutatingTools: Set<String> = ["replace_in_file", "create_file"]

    static let tools: [APIToolDefinition] = [
        definition(
            name: "list_files",
            description: "List repository files below an optional relative directory.",
            properties: [
                "path": stringProperty("Relative directory; defaults to the repository root"),
                "limit": numberProperty("Maximum files to return, from 1 to 500"),
            ]
        ),
        definition(
            name: "read_file",
            description: "Read a bounded line range from a UTF-8 repository file.",
            properties: [
                "path": stringProperty("Repository-relative file path"),
                "start_line": numberProperty("First one-based line; defaults to 1"),
                "end_line": numberProperty("Last one-based line; at most 400 lines are returned"),
            ],
            required: ["path"]
        ),
        definition(
            name: "search_code",
            description: "Search text files in the repository for a literal query.",
            properties: [
                "query": stringProperty("Literal text to find"),
                "path": stringProperty("Optional relative directory or file"),
                "limit": numberProperty("Maximum matching lines, from 1 to 100"),
            ],
            required: ["query"]
        ),
        definition(
            name: "replace_in_file",
            description: "Replace one exact, unique text block in a repository file.",
            properties: [
                "path": stringProperty("Repository-relative file path"),
                "old_text": stringProperty("Exact existing text; it must occur once"),
                "new_text": stringProperty("Replacement text"),
            ],
            required: ["path", "old_text", "new_text"]
        ),
        definition(
            name: "create_file",
            description: "Create a new UTF-8 text file without overwriting an existing file.",
            properties: [
                "path": stringProperty("Repository-relative new file path"),
                "content": stringProperty("Complete file contents"),
            ],
            required: ["path", "content"]
        ),
    ]

    static func isMutating(_ call: APIToolCall) -> Bool {
        mutatingTools.contains(call.function.name)
    }

    static func supports(_ call: APIToolCall) -> Bool {
        let name = call.function.name
        guard !name.isEmpty, name.count <= 64 else { return false }
        return name.range(
            of: #"^[A-Za-z0-9_-]+$"#,
            options: .regularExpression
        ) != nil
    }

    static func tools(for mode: CodingPermissionMode) -> [APIToolDefinition] {
        guard mode == .plan else { return tools }
        return tools.filter { !mutatingTools.contains($0.function.name) }
    }

    static func systemPrompt(for mode: CodingPermissionMode) -> String {
        let policy: String
        switch mode {
        case .plan:
            policy = "This is a read-only planning session. Inspect the repository and propose a concrete plan, but do not edit files or claim that changes were applied."
        case .manual:
            policy = "Every file change requires explicit user approval."
        case .automatic:
            policy = "Small exact edits may be approved automatically; broader changes require user approval."
        case .acceptEdits:
            policy = "Repository file edits are approved automatically."
        case .bypass:
            policy = "Repository tools are approved automatically, but the selected repository boundary remains mandatory."
        }
        return """
        Coding mode is active for a repository selected by the user. Use repository tools to inspect relevant files before answering questions about the code. Prefer targeted searches and small reads. Never claim a file was changed unless a write tool succeeds. \(policy) Do not request secrets, dependency caches, build output, or .git data. Keep changes focused and report the paths changed.
        """
    }

    static func permissionDecision(
        for call: APIToolCall,
        mode: CodingPermissionMode
    ) -> CodingPermissionDecision {
        guard isMutating(call) else { return .allow }
        switch mode {
        case .plan:
            return .deny("Plan mode does not allow repository changes.")
        case .manual:
            return .ask
        case .acceptEdits, .bypass:
            return .allow
        case .automatic:
            guard call.function.name == "replace_in_file" else { return .ask }
            let arguments = object(call.function.arguments)
            let oldText = string(arguments["old_text"]) ?? ""
            let newText = string(arguments["new_text"]) ?? ""
            let changedLines = max(
                oldText.components(separatedBy: .newlines).count,
                newText.components(separatedBy: .newlines).count
            )
            return max(oldText.count, newText.count) <= 4_000 && changedLines <= 80
                ? .allow
                : .ask
        }
    }

    static func summary(of call: APIToolCall) -> String {
        let arguments = object(call.function.arguments)
        let path = string(arguments["path"]) ?? "repository"
        switch call.function.name {
        case "replace_in_file": return "Edit \(path)"
        case "create_file": return "Create \(path)"
        case "read_file": return "Read \(path)"
        case "search_code": return "Search for \(string(arguments["query"]) ?? "text")"
        default: return "Run \(call.function.name)"
        }
    }

    static func activitySummary(of call: APIToolCall) -> String {
        let arguments = object(call.function.arguments)
        let path = string(arguments["path"]) ?? "repository root"
        switch call.function.name {
        case "list_files": return "Listed files in \(path)"
        case "read_file": return "Read \(path)"
        case "search_code": return "Searched for \(string(arguments["query"]) ?? "text")"
        case "replace_in_file": return "Edited \(path)"
        case "create_file": return "Created \(path)"
        default: return "Ran \(call.function.name)"
        }
    }

    static func activityDetails(of call: APIToolCall) -> [(String, String)] {
        let arguments = object(call.function.arguments)
        let path = string(arguments["path"])
        switch call.function.name {
        case "list_files":
            return [("Directory", path ?? "Repository root")]
        case "read_file":
            var rows = [("File", path ?? "Unknown")]
            if let start = int(arguments["start_line"]), let end = int(arguments["end_line"]) {
                rows.append(("Lines", "\(start)-\(end)"))
            }
            return rows
        case "search_code":
            return [
                ("Query", string(arguments["query"]) ?? ""),
                ("Location", path ?? "Entire repository"),
            ]
        case "replace_in_file", "create_file":
            return [("File", path ?? "Unknown")]
        default:
            return []
        }
    }

    static func displayResult(_ content: String) -> String {
        guard
            let data = content.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return content
        }
        if let error = object["error"] as? String { return error }
        if let files = object["files"] as? [String] {
            return files.isEmpty ? "No files found." : files.joined(separator: "\n")
        }
        if let text = object["content"] as? String { return text }
        if let matches = object["matches"] as? [[String: Any]] {
            if matches.isEmpty { return "No matches found." }
            return matches.map { match in
                let path = match["path"] as? String ?? "file"
                let line = match["line"] as? Int ?? 0
                let text = match["text"] as? String ?? ""
                return "\(path):\(line)  \(text)"
            }.joined(separator: "\n")
        }
        if let path = object["path"] as? String, let status = object["status"] as? String {
            return "\(status.capitalized) \(path)"
        }
        return content
    }

    static func visibleAssistantText(_ text: String) -> String {
        var value = text.replacingOccurrences(
            of: #"(?s)<atem:function_calls>.*?</atem:function_calls>"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        value = value.replacingOccurrences(
            of: #"(?s)<tool_call\b[^>]*>.*?</tool_call>"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        value = value.replacingOccurrences(
            of: #"<\|(?:start|message|end|call|channel|eom|eot)\|>"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        value = value.replacingOccurrences(
            of: #"^\s*(?:(?:assistant\s+)?to\s*=\s*user\s*)+"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        return value.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func fileURL(relativePath: String, workspaceRoot: String?) -> URL? {
        guard let workspaceRoot else { return nil }
        let root = URL(fileURLWithPath: workspaceRoot, isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        guard
            let file = try? safeURL(root: root, relativePath: relativePath, mustExist: true),
            contains(file, root: root)
        else {
            return nil
        }
        return file
    }

    static func execute(_ call: APIToolCall, workspaceRoot: String) throws -> CodingToolResult {
        let root = URL(fileURLWithPath: workspaceRoot, isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw CodingWorkspaceError.pathNotFound(workspaceRoot)
        }
        let arguments = object(call.function.arguments)
        let content: String
        let changedPath: String?
        let changePatch: String?
        switch call.function.name {
        case "list_files":
            content = try listFiles(
                root: root,
                path: string(arguments["path"]),
                limit: boundedInt(arguments["limit"], default: 200, range: 1 ... 500)
            )
            changedPath = nil
            changePatch = nil
        case "read_file":
            let path = try requiredString(arguments, "path")
            content = try readFile(
                root: root,
                path: path,
                startLine: boundedInt(arguments["start_line"], default: 1, range: 1 ... 1_000_000),
                endLine: boundedInt(arguments["end_line"], default: 400, range: 1 ... 1_000_000)
            )
            changedPath = nil
            changePatch = nil
        case "search_code":
            content = try searchCode(
                root: root,
                query: requiredString(arguments, "query"),
                path: string(arguments["path"]),
                limit: boundedInt(arguments["limit"], default: 50, range: 1 ... 100)
            )
            changedPath = nil
            changePatch = nil
        case "replace_in_file":
            let path = try requiredString(arguments, "path")
            let oldText = try requiredString(arguments, "old_text", allowEmpty: false)
            let newText = try requiredString(arguments, "new_text", allowEmpty: true)
            content = try replaceInFile(
                root: root,
                path: path,
                oldText: oldText,
                newText: newText
            )
            changedPath = path
            changePatch = patch(path: path, before: oldText, after: newText, created: false)
        case "create_file":
            let path = try requiredString(arguments, "path")
            let newContent = try requiredString(arguments, "content", allowEmpty: true)
            content = try createFile(
                root: root,
                path: path,
                content: newContent
            )
            changedPath = path
            changePatch = patch(path: path, before: "", after: newContent, created: true)
        default:
            throw CodingWorkspaceError.invalidArguments(
                "Unknown coding tool: \(call.function.name)"
            )
        }
        return CodingToolResult(
            callID: call.id,
            name: call.function.name,
            content: content,
            changedPath: changedPath,
            changePatch: changePatch
        )
    }

    private static func patch(
        path: String,
        before: String,
        after: String,
        created: Bool
    ) -> String {
        var lines = [
            "--- \(created ? "/dev/null" : "a/\(path)")",
            "+++ b/\(path)",
            "@@ changed block @@",
        ]
        let beforeLines = before.components(separatedBy: .newlines)
        let afterLines = after.components(separatedBy: .newlines)
        lines.append(contentsOf: beforeLines.prefix(200).map { "-\($0)" })
        lines.append(contentsOf: afterLines.prefix(200).map { "+\($0)" })
        if beforeLines.count > 200 || afterLines.count > 200 {
            lines.append("... patch preview truncated ...")
        }
        return String(lines.joined(separator: "\n").prefix(24_000))
    }

    private static func listFiles(
        root: URL,
        path: String?,
        limit: Int
    ) throws -> String {
        let start = try safeURL(root: root, relativePath: path ?? "", mustExist: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: start.path, isDirectory: &isDirectory) else {
            throw CodingWorkspaceError.pathNotFound(path ?? ".")
        }
        if !isDirectory.boolValue {
            return json(["files": [relativePath(start, root: root)], "truncated": false])
        }
        let keys: [URLResourceKey] = [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey]
        guard let enumerator = FileManager.default.enumerator(
            at: start,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return json(["files": [], "truncated": false])
        }
        var files: [String] = []
        var truncated = false
        for case let file as URL in enumerator {
            let values = try? file.resourceValues(forKeys: Set(keys))
            if values?.isDirectory == true, ignoredDirectory(file.lastPathComponent) {
                enumerator.skipDescendants()
                continue
            }
            if values?.isSymbolicLink == true || values?.isRegularFile != true { continue }
            files.append(relativePath(file, root: root))
            if files.count >= limit {
                truncated = true
                break
            }
        }
        return json(["files": files.sorted(), "truncated": truncated])
    }

    private static func readFile(
        root: URL,
        path: String,
        startLine: Int,
        endLine: Int
    ) throws -> String {
        let file = try safeURL(root: root, relativePath: path, mustExist: true)
        let text = try textFile(file, displayPath: path)
        let lines = text.components(separatedBy: .newlines)
        let start = min(max(1, startLine), max(1, lines.count))
        let requestedEnd = max(start, endLine)
        let end = min(lines.count, min(requestedEnd, start + 399))
        let numbered = (start ... end).map { "\($0):\(lines[$0 - 1])" }
        return json([
            "path": path,
            "start_line": start,
            "end_line": end,
            "total_lines": lines.count,
            "content": numbered.joined(separator: "\n"),
            "truncated": end < requestedEnd || end < lines.count,
        ])
    }

    private static func searchCode(
        root: URL,
        query: String,
        path: String?,
        limit: Int
    ) throws -> String {
        guard !query.isEmpty, query.count <= 500 else {
            throw CodingWorkspaceError.invalidArguments("query must contain 1 to 500 characters")
        }
        let start = try safeURL(root: root, relativePath: path ?? "", mustExist: true)
        let files = try candidateFiles(at: start)
        var matches: [[String: Any]] = []
        var truncated = false
        for file in files {
            guard let text = try? textFile(file, displayPath: relativePath(file, root: root)) else {
                continue
            }
            for (index, line) in text.components(separatedBy: .newlines).enumerated()
            where line.localizedCaseInsensitiveContains(query) {
                matches.append([
                    "path": relativePath(file, root: root),
                    "line": index + 1,
                    "text": String(line.prefix(1_000)),
                ])
                if matches.count >= limit {
                    truncated = true
                    break
                }
            }
            if truncated { break }
        }
        return json(["query": query, "matches": matches, "truncated": truncated])
    }

    private static func replaceInFile(
        root: URL,
        path: String,
        oldText: String,
        newText: String
    ) throws -> String {
        let file = try safeURL(root: root, relativePath: path, mustExist: true)
        let text = try textFile(file, displayPath: path)
        let count = text.components(separatedBy: oldText).count - 1
        guard count == 1 else { throw CodingWorkspaceError.ambiguousReplacement(count) }
        let updated = text.replacingOccurrences(of: oldText, with: newText)
        try Data(updated.utf8).write(to: file, options: .atomic)
        return json(["status": "updated", "path": path, "replacements": 1])
    }

    private static func createFile(root: URL, path: String, content: String) throws -> String {
        let file = try safeURL(root: root, relativePath: path, mustExist: false)
        guard !FileManager.default.fileExists(atPath: file.path) else {
            throw CodingWorkspaceError.fileAlreadyExists(path)
        }
        let parent = file.deletingLastPathComponent().resolvingSymlinksInPath()
        guard contains(parent, root: root) else { throw CodingWorkspaceError.pathOutsideWorkspace }
        try FileManager.default.createDirectory(
            at: parent,
            withIntermediateDirectories: true
        )
        try Data(content.utf8).write(to: file, options: .atomic)
        return json(["status": "created", "path": path, "bytes": content.utf8.count])
    }

    private static func candidateFiles(at start: URL) throws -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: start.path, isDirectory: &isDirectory) else {
            throw CodingWorkspaceError.pathNotFound(start.path)
        }
        if !isDirectory.boolValue { return [start] }
        let keys: [URLResourceKey] = [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]
        guard let enumerator = FileManager.default.enumerator(
            at: start,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return [] }
        var files: [URL] = []
        for case let file as URL in enumerator {
            let values = try? file.resourceValues(forKeys: Set(keys))
            if values?.isDirectory == true, ignoredDirectory(file.lastPathComponent) {
                enumerator.skipDescendants()
                continue
            }
            guard values?.isSymbolicLink != true,
                  values?.isRegularFile == true,
                  (values?.fileSize ?? 0) <= 1_000_000 else { continue }
            files.append(file)
            if files.count >= 5_000 { break }
        }
        return files.sorted { $0.path < $1.path }
    }

    private static func safeURL(
        root: URL,
        relativePath: String,
        mustExist: Bool
    ) throws -> URL {
        let trimmed = relativePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.hasPrefix("/"), !trimmed.contains("\0") else {
            throw CodingWorkspaceError.pathOutsideWorkspace
        }
        let components = NSString(string: trimmed).pathComponents
        guard !components.contains(".."), !components.contains(".git") else {
            throw CodingWorkspaceError.pathOutsideWorkspace
        }
        let standardized = root.appendingPathComponent(trimmed).standardizedFileURL
        let resolved: URL
        if mustExist || FileManager.default.fileExists(atPath: standardized.path) {
            resolved = standardized.resolvingSymlinksInPath()
        } else {
            let parent = standardized.deletingLastPathComponent().resolvingSymlinksInPath()
            guard contains(parent, root: root) else {
                throw CodingWorkspaceError.pathOutsideWorkspace
            }
            resolved = parent.appendingPathComponent(standardized.lastPathComponent)
        }
        guard contains(resolved, root: root) else {
            throw CodingWorkspaceError.pathOutsideWorkspace
        }
        if mustExist, !FileManager.default.fileExists(atPath: resolved.path) {
            throw CodingWorkspaceError.pathNotFound(trimmed)
        }
        return resolved
    }

    private static func contains(_ candidate: URL, root: URL) -> Bool {
        candidate.path == root.path || candidate.path.hasPrefix(root.path + "/")
    }

    private static func textFile(_ url: URL, displayPath: String) throws -> String {
        let data = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard data.count <= 1_000_000, !data.contains(0), let text = String(data: data, encoding: .utf8) else {
            throw CodingWorkspaceError.unsupportedFile(displayPath)
        }
        return text
    }

    private static func relativePath(_ file: URL, root: URL) -> String {
        String(file.path.dropFirst(min(file.path.count, root.path.count + 1)))
    }

    private static func ignoredDirectory(_ name: String) -> Bool {
        [".git", ".build", ".cache", ".venv", "build", "dist", "node_modules", "Pods"]
            .contains(name)
    }

    private static func requiredString(
        _ values: [String: JSONValue],
        _ key: String,
        allowEmpty: Bool = false
    ) throws -> String {
        guard let value = string(values[key]), allowEmpty || !value.isEmpty else {
            throw CodingWorkspaceError.invalidArguments("\(key) is required")
        }
        guard value.utf8.count <= 1_000_000 else {
            throw CodingWorkspaceError.invalidArguments("\(key) is too large")
        }
        return value
    }

    private static func object(_ value: JSONValue?) -> [String: JSONValue] {
        guard case let .object(object) = value else { return [:] }
        return object
    }

    private static func string(_ value: JSONValue?) -> String? {
        guard case let .string(string) = value else { return nil }
        return string
    }

    private static func int(_ value: JSONValue?) -> Int? {
        guard case let .number(number) = value else { return nil }
        return Int(number)
    }

    private static func boundedInt(
        _ value: JSONValue?,
        default defaultValue: Int,
        range: ClosedRange<Int>
    ) -> Int {
        let raw: Int
        if case let .number(number) = value { raw = Int(number) } else { raw = defaultValue }
        return min(range.upperBound, max(range.lowerBound, raw))
    }

    private static func definition(
        name: String,
        description: String,
        properties: [String: JSONValue],
        required: [String] = []
    ) -> APIToolDefinition {
        APIToolDefinition(
            function: .init(
                name: name,
                description: description,
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object(properties),
                    "required": .array(required.map(JSONValue.string)),
                    "additionalProperties": .boolean(false),
                ])
            )
        )
    }

    private static func stringProperty(_ description: String) -> JSONValue {
        .object(["type": .string("string"), "description": .string(description)])
    }

    private static func numberProperty(_ description: String) -> JSONValue {
        .object(["type": .string("integer"), "description": .string(description)])
    }

    private static func json(_ object: Any) -> String {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(
                  withJSONObject: object,
                  options: [.sortedKeys, .withoutEscapingSlashes]
              )
        else { return #"{"error":"tool result could not be encoded"}"# }
        return String(decoding: data, as: UTF8.self)
    }
}
