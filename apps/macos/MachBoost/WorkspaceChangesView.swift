import AppKit
import Foundation
import SwiftUI

struct WorkspaceChange: Identifiable, Hashable, Sendable {
    let path: String
    let status: String
    let additions: Int
    let deletions: Int
    let patch: String

    var id: String { path }
}

struct WorkspaceChangeSet: Equatable, Sendable {
    let branch: String
    let changes: [WorkspaceChange]
    let error: String?

    static let empty = WorkspaceChangeSet(branch: "", changes: [], error: nil)

    var additions: Int { changes.reduce(0) { $0 + $1.additions } }
    var deletions: Int { changes.reduce(0) { $0 + $1.deletions } }
}

enum WorkspaceChangeScope: String, CaseIterable, Identifiable {
    case conversation = "This Chat"
    case workingTree = "Working Tree"

    var id: String { rawValue }
}

enum WorkspaceChanges {
    static func load(workspaceRoot: String) -> WorkspaceChangeSet {
        let root = URL(fileURLWithPath: workspaceRoot, isDirectory: true)
        do {
            guard try git(root, ["rev-parse", "--is-inside-work-tree"]).trimmed == "true" else {
                return .init(
                    branch: "",
                    changes: [],
                    error: "The selected workspace is not a Git repository."
                )
            }
            let branch = (try? git(root, ["branch", "--show-current"]).trimmed)
                .flatMap { $0.isEmpty ? nil : $0 }
                ?? "detached"
            let status = try git(
                root,
                ["status", "--porcelain=v1", "--untracked-files=all"]
            )
            let changes = status.split(separator: "\n").compactMap { row -> WorkspaceChange? in
                let line = String(row)
                guard line.count >= 4 else { return nil }
                let code = String(line.prefix(2))
                var path = String(line.dropFirst(3))
                if let rename = path.range(of: " -> ") {
                    path = String(path[rename.upperBound...])
                }
                path = unquoted(path)
                guard !path.isEmpty else { return nil }
                return change(root: root, path: path, status: statusLabel(code))
            }
            return .init(branch: branch, changes: changes, error: nil)
        } catch {
            return .init(branch: "", changes: [], error: error.localizedDescription)
        }
    }

    static func session(
        workspaceRoot: String,
        activities: [CodingToolActivity]
    ) -> WorkspaceChangeSet {
        let root = URL(fileURLWithPath: workspaceRoot, isDirectory: true)
        let branch = (try? git(root, ["branch", "--show-current"]).trimmed)
            .flatMap { $0.isEmpty ? nil : $0 }
            ?? "detached"
        let successful = activities.filter {
            $0.state == .succeeded && $0.changedPath != nil && $0.changePatch != nil
        }
        var orderedPaths: [String] = []
        var changesByPath: [String: WorkspaceChange] = [:]
        for activity in successful {
            guard let path = activity.changedPath, let patch = activity.changePatch else {
                continue
            }
            let counts = diffCounts(patch)
            let status = patch.contains("--- /dev/null") ? "Created" : "Edited"
            if let existing = changesByPath[path] {
                changesByPath[path] = WorkspaceChange(
                    path: path,
                    status: existing.status == "Created" ? existing.status : status,
                    additions: existing.additions + counts.additions,
                    deletions: existing.deletions + counts.deletions,
                    patch: String((existing.patch + "\n\n" + patch).prefix(60_000))
                )
            } else {
                orderedPaths.append(path)
                changesByPath[path] = WorkspaceChange(
                    path: path,
                    status: status,
                    additions: counts.additions,
                    deletions: counts.deletions,
                    patch: String(patch.prefix(60_000))
                )
            }
        }
        return WorkspaceChangeSet(
            branch: branch,
            changes: orderedPaths.compactMap { changesByPath[$0] },
            error: nil
        )
    }

    private static func change(root: URL, path: String, status: String) -> WorkspaceChange {
        if status == "Untracked" {
            let url = root.appendingPathComponent(path)
            guard
                let data = try? Data(contentsOf: url, options: [.mappedIfSafe]),
                data.count <= 256_000,
                !data.contains(0),
                let content = String(data: data, encoding: .utf8)
            else {
                return .init(
                    path: path,
                    status: status,
                    additions: 0,
                    deletions: 0,
                    patch: "Binary or large untracked file"
                )
            }
            let lines = content.components(separatedBy: .newlines)
            let body = lines.prefix(400).map { "+\($0)" }.joined(separator: "\n")
            let suffix = lines.count > 400 ? "\n... diff truncated ..." : ""
            return .init(
                path: path,
                status: status,
                additions: lines.count,
                deletions: 0,
                patch: "--- /dev/null\n+++ b/\(path)\n@@ new file @@\n\(body)\(suffix)"
            )
        }

        let patch = (try? git(
            root,
            ["diff", "HEAD", "--no-ext-diff", "--unified=3", "--", path],
            acceptedExitCodes: [0]
        )) ?? ""
        let counts = (try? git(
            root,
            ["diff", "HEAD", "--numstat", "--", path],
            acceptedExitCodes: [0]
        ))?.split(separator: "\t") ?? []
        let additions = counts.first.flatMap { Int($0) } ?? 0
        let deletions = counts.dropFirst().first.flatMap { Int($0) } ?? 0
        return .init(
            path: path,
            status: status,
            additions: additions,
            deletions: deletions,
            patch: String(patch.prefix(60_000))
        )
    }

    private static func git(
        _ root: URL,
        _ arguments: [String],
        acceptedExitCodes: Set<Int32> = [0]
    ) throws -> String {
        let process = Process()
        let stdout = Pipe()
        let stderr = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["-C", root.path] + arguments
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        let output = stdout.fileHandleForReading.readDataToEndOfFile()
        let error = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard acceptedExitCodes.contains(process.terminationStatus) else {
            let message = String(decoding: error, as: UTF8.self).trimmed
            throw NSError(
                domain: "MachBoost.WorkspaceChanges",
                code: Int(process.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey: message.isEmpty ? "Git command failed." : message]
            )
        }
        return String(decoding: output, as: UTF8.self)
    }

    private static func statusLabel(_ code: String) -> String {
        if code == "??" { return "Untracked" }
        if code.contains("A") { return "Added" }
        if code.contains("D") { return "Deleted" }
        if code.contains("R") { return "Renamed" }
        return "Modified"
    }

    private static func unquoted(_ path: String) -> String {
        guard path.hasPrefix("\""), path.hasSuffix("\"") else { return path }
        let value = String(path.dropFirst().dropLast())
        return value.replacingOccurrences(of: "\\\"", with: "\"")
            .replacingOccurrences(of: "\\\\", with: "\\")
    }

    private static func diffCounts(_ patch: String) -> (additions: Int, deletions: Int) {
        var additions = 0
        var deletions = 0
        for line in patch.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("+") && !line.hasPrefix("+++") {
                additions += 1
            } else if line.hasPrefix("-") && !line.hasPrefix("---") {
                deletions += 1
            }
        }
        return (additions, deletions)
    }
}

struct WorkspaceChangesView: View {
    let snapshot: WorkspaceChangeSet
    let workspaceRoot: String
    let isRefreshing: Bool
    @Binding var scope: WorkspaceChangeScope
    let onRefresh: () -> Void
    let onClose: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .accessibilityIdentifier("workspace-changes-panel")
    }

    private var header: some View {
        VStack(spacing: 9) {
            HStack(spacing: 9) {
                Image(systemName: "arrow.triangle.branch")
                    .foregroundStyle(.green)
                VStack(alignment: .leading, spacing: 1) {
                    Text(snapshot.branch.isEmpty ? "Workspace changes" : snapshot.branch)
                        .font(.callout.weight(.semibold))
                    Text("\(snapshot.changes.count) changed files")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if snapshot.additions > 0 {
                    Text("+\(snapshot.additions)")
                        .foregroundStyle(.green)
                }
                if snapshot.deletions > 0 {
                    Text("−\(snapshot.deletions)")
                        .foregroundStyle(.red)
                }
                Button(action: onRefresh) {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .disabled(isRefreshing)
                .help("Refresh changes")
                Button(action: onClose) {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.borderless)
                .help("Close changes")
            }

            Picker("Change scope", selection: $scope) {
                ForEach(WorkspaceChangeScope.allCases) { value in
                    Text(value.rawValue).tag(value)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .accessibilityIdentifier("workspace-change-scope")
        }
        .font(.caption.monospacedDigit())
        .padding(12)
    }

    @ViewBuilder
    private var content: some View {
        if let error = snapshot.error {
            ContentUnavailableView(
                "Changes unavailable",
                systemImage: "exclamationmark.triangle",
                description: Text(error)
            )
        } else if snapshot.changes.isEmpty {
            ContentUnavailableView(
                scope == .conversation ? "No changes from this chat" : "Working tree is clean",
                systemImage: "checkmark.circle",
                description: Text(
                    scope == .conversation
                        ? "Files changed by other chats and tools stay hidden."
                        : "Repository changes will appear here."
                )
            )
        } else {
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(snapshot.changes) { change in
                        DisclosureGroup {
                            changeDetails(change)
                                .padding(.top, 8)
                        } label: {
                            changeLabel(change)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        Divider()
                    }
                }
            }
        }
    }

    private func changeLabel(_ change: WorkspaceChange) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "doc.text")
                .foregroundStyle(.secondary)
            Text(change.path)
                .font(.callout.monospaced())
                .lineLimit(1)
            Spacer()
            Text(change.status)
                .font(.caption)
                .foregroundStyle(.secondary)
            if change.additions > 0 {
                Text("+\(change.additions)").foregroundStyle(.green)
            }
            if change.deletions > 0 {
                Text("−\(change.deletions)").foregroundStyle(.red)
            }
        }
        .font(.caption.monospacedDigit())
    }

    private func changeDetails(_ change: WorkspaceChange) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Spacer()
                Button {
                    open(change.path)
                } label: {
                    Image(systemName: "arrow.up.forward.app")
                }
                .buttonStyle(.borderless)
                .help("Open file")
                Button {
                    reveal(change.path)
                } label: {
                    Image(systemName: "folder")
                }
                .buttonStyle(.borderless)
                .help("Reveal in Finder")
            }
            ScrollView(.horizontal) {
                VStack(alignment: .leading, spacing: 1) {
                    ForEach(Array(change.patch.split(separator: "\n", omittingEmptySubsequences: false).enumerated()), id: \.offset) { _, line in
                        Text(String(line))
                            .foregroundStyle(diffColor(String(line)))
                    }
                }
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 420)
        }
    }

    private func diffColor(_ line: String) -> Color {
        if line.hasPrefix("+++") || line.hasPrefix("---") { return .secondary }
        if line.hasPrefix("+") { return .green }
        if line.hasPrefix("-") { return .red }
        return .primary
    }

    private func open(_ path: String) {
        guard let url = CodingWorkspace.fileURL(relativePath: path, workspaceRoot: workspaceRoot) else {
            return
        }
        NSWorkspace.shared.open(url)
    }

    private func reveal(_ path: String) {
        guard let url = CodingWorkspace.fileURL(relativePath: path, workspaceRoot: workspaceRoot) else {
            return
        }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
}

private extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
