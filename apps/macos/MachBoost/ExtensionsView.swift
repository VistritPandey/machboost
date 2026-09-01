import MachBoostDaemonClient
import SwiftUI

private enum ExtensionSection: String, CaseIterable, Identifiable {
    case connectors = "Connectors"
    case instructions = "Instructions"

    var id: String { rawValue }
}

struct ExtensionsView: View {
    @Environment(AppState.self) private var appState
    @State private var section = ExtensionSection.connectors
    @State private var showsConnectorEditor = false
    @State private var editingConnector: MCPServerSummary?
    @State private var showsSkillEditor = false
    @State private var editingSkill: SkillSummary?
    @State private var testingConnectorID: String?
    @State private var testResult = ""

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Picker("Extension type", selection: $section) {
                        ForEach(ExtensionSection.allCases) { section in
                            Text(section.rawValue).tag(section)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 360)

                    if section == .connectors {
                        connectors
                    } else {
                        skills
                    }
                }
                .padding(22)
                .frame(maxWidth: 920, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
        }
        .sheet(isPresented: $showsConnectorEditor) {
            MCPConnectorEditor(connector: editingConnector) { draft in
                let saved = await appState.configureMCPServer(
                    id: editingConnector?.id,
                    name: draft.name,
                    transport: draft.transport,
                    url: draft.url,
                    command: draft.command,
                    args: draft.args,
                    environment: draft.environment,
                    headers: draft.headers,
                    enabled: draft.enabled
                )
                if saved { showsConnectorEditor = false }
                return saved
            }
        }
        .sheet(isPresented: $showsSkillEditor) {
            SkillEditor(skill: editingSkill) { name, instructions, enabled in
                let saved = await appState.configureSkill(
                    id: editingSkill?.id,
                    name: name,
                    instructions: instructions,
                    enabled: enabled
                )
                if saved { showsSkillEditor = false }
                return saved
            }
        }
        .alert("Connector test", isPresented: Binding(
            get: { !testResult.isEmpty },
            set: { if !$0 { testResult = "" } }
        )) {
            Button("OK") { testResult = "" }
        } message: {
            Text(testResult)
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Extensions")
                    .font(.title2.weight(.semibold))
                Text("Connect tools and add reusable guidance for every chat.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                Task { await appState.refreshMemoryAndProviders() }
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .help("Refresh extensions")
            Button {
                if section == .connectors {
                    editingConnector = nil
                    showsConnectorEditor = true
                } else {
                    editingSkill = nil
                    showsSkillEditor = true
                }
            } label: {
                Label(section == .connectors ? "Add connector" : "Add instructions", systemImage: "plus")
            }
        }
        .padding(18)
    }

    @ViewBuilder
    private var connectors: some View {
        if appState.mcpServers.isEmpty {
            ContentUnavailableView(
                "No connectors",
                systemImage: "puzzlepiece.extension",
                description: Text("Connect a local command or a remote MCP server. Tools stay behind a compact gateway until a chat needs them.")
            )
            .frame(maxWidth: .infinity, minHeight: 280)
        } else {
            VStack(spacing: 0) {
                ForEach(appState.mcpServers) { connector in
                    connectorRow(connector)
                    if connector.id != appState.mcpServers.last?.id { Divider() }
                }
            }
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.separator))
        }
    }

    private func connectorRow(_ connector: MCPServerSummary) -> some View {
        HStack(spacing: 14) {
            Image(systemName: connector.transport == "http" ? "network" : "terminal")
                .font(.title3)
                .foregroundStyle(connector.enabled ? Color.green : Color.secondary)
                .frame(width: 28)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(connector.name).fontWeight(.medium)
                    if connector.lastStatus == "ready" {
                        Label("Ready", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    } else if connector.lastStatus == "error" {
                        Label("Needs attention", systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    } else if !connector.enabled {
                        Text("Disabled").foregroundStyle(.secondary)
                    }
                }
                Text(connector.url ?? ([connector.command ?? "", connector.args.joined(separator: " ")].filter { !$0.isEmpty }.joined(separator: " ")))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                if let error = connector.lastError, !error.isEmpty {
                    Text(error).font(.caption).foregroundStyle(.orange).lineLimit(2)
                } else if connector.toolCount > 0 {
                    Text("\(connector.toolCount) tools available")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            if testingConnectorID == connector.id {
                ProgressView().controlSize(.small)
            }
            Button {
                test(connector)
            } label: {
                Image(systemName: "wave.3.right")
            }
            .disabled(testingConnectorID != nil || !connector.enabled)
            .help("Test connector")
            Menu {
                Button("Edit") {
                    editingConnector = connector
                    showsConnectorEditor = true
                }
                Button("Delete", role: .destructive) {
                    Task { await appState.deleteMCPServer(id: connector.id) }
                }
            } label: {
                Image(systemName: "ellipsis")
            }
            .menuStyle(.borderlessButton)
            .frame(width: 28)
        }
        .padding(14)
    }

    @ViewBuilder
    private var skills: some View {
        if appState.skills.isEmpty {
            ContentUnavailableView(
                "No reusable instructions",
                systemImage: "text.badge.plus",
                description: Text("Add project conventions, response preferences, or a repeatable role. Enabled instructions apply to local and shared chats.")
            )
            .frame(maxWidth: .infinity, minHeight: 280)
        } else {
            VStack(spacing: 0) {
                ForEach(appState.skills) { skill in
                    HStack(spacing: 14) {
                        Image(systemName: "text.book.closed")
                            .font(.title3)
                            .foregroundStyle(skill.enabled ? Color.green : Color.secondary)
                            .frame(width: 28)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(skill.name).fontWeight(.medium)
                            Text(skill.instructions)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                        Spacer()
                        Toggle("", isOn: Binding(
                            get: { skill.enabled },
                            set: { enabled in
                                Task {
                                    _ = await appState.configureSkill(
                                        id: skill.id,
                                        name: skill.name,
                                        instructions: skill.instructions,
                                        enabled: enabled
                                    )
                                }
                            }
                        ))
                        .labelsHidden()
                        Menu {
                            Button("Edit") {
                                editingSkill = skill
                                showsSkillEditor = true
                            }
                            Button("Delete", role: .destructive) {
                                Task { await appState.deleteSkill(id: skill.id) }
                            }
                        } label: {
                            Image(systemName: "ellipsis")
                        }
                        .menuStyle(.borderlessButton)
                        .frame(width: 28)
                    }
                    .padding(14)
                    if skill.id != appState.skills.last?.id { Divider() }
                }
            }
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.separator))
        }
    }

    private func test(_ connector: MCPServerSummary) {
        testingConnectorID = connector.id
        Task {
            let tools = await appState.testMCPServer(id: connector.id)
            testingConnectorID = nil
            if let tools {
                testResult = tools.isEmpty
                    ? "Connected, but this server did not publish any tools."
                    : "Connected. Found \(tools.count) tool\(tools.count == 1 ? "" : "s")."
            }
        }
    }
}

private struct MCPConnectorDraft {
    let name: String
    let transport: String
    let url: String?
    let command: String?
    let args: [String]
    let environment: [String: String]
    let headers: [String: String]
    let enabled: Bool
}

private struct MCPConnectorEditor: View {
    @Environment(\.dismiss) private var dismiss
    @State private var name: String
    @State private var transport: String
    @State private var url: String
    @State private var command: String
    @State private var arguments: String
    @State private var environment: String
    @State private var headers: String
    @State private var enabled: Bool
    @State private var isSaving = false
    let save: (MCPConnectorDraft) async -> Bool

    init(connector: MCPServerSummary?, save: @escaping (MCPConnectorDraft) async -> Bool) {
        _name = State(initialValue: connector?.name ?? "")
        _transport = State(initialValue: connector?.transport ?? "http")
        _url = State(initialValue: connector?.url ?? "")
        _command = State(initialValue: connector?.command ?? "")
        _arguments = State(initialValue: connector?.args.joined(separator: "\n") ?? "")
        _environment = State(initialValue: "")
        _headers = State(initialValue: "")
        _enabled = State(initialValue: connector?.enabled ?? true)
        self.save = save
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("MCP connector").font(.title3.weight(.semibold))
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") { saveConnector() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!isValid || isSaving)
            }
            .padding(18)
            Divider()
            Form {
                TextField("Name", text: $name)
                Picker("Connection", selection: $transport) {
                    Text("Remote URL").tag("http")
                    Text("Local command").tag("stdio")
                }
                .pickerStyle(.segmented)
                if transport == "http" {
                    TextField("Server URL", text: $url, prompt: Text("https://server.example/mcp"))
                    TextField("Request headers", text: $headers, axis: .vertical)
                        .lineLimit(2 ... 5)
                    Text("One NAME=VALUE header per line. Saved values are preserved when this is left blank and are never returned by the API.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    TextField("Command", text: $command, prompt: Text("npx"))
                    TextField("Arguments", text: $arguments, axis: .vertical)
                        .lineLimit(2 ... 5)
                    TextField("Environment", text: $environment, axis: .vertical)
                        .lineLimit(2 ... 5)
                    Text("Enter one argument per line and one NAME=VALUE environment variable per line. Saved environment values are preserved when left blank.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Toggle("Enabled", isOn: $enabled)
            }
            .formStyle(.grouped)
            .padding(.horizontal, 8)
        }
        .frame(width: 560, height: 500)
    }

    private var isValid: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && (transport == "http"
                ? URL(string: url)?.scheme?.hasPrefix("http") == true
                : !command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    private func saveConnector() {
        isSaving = true
        let draft = MCPConnectorDraft(
            name: name.trimmingCharacters(in: .whitespacesAndNewlines),
            transport: transport,
            url: transport == "http" ? url.trimmingCharacters(in: .whitespacesAndNewlines) : nil,
            command: transport == "stdio" ? command.trimmingCharacters(in: .whitespacesAndNewlines) : nil,
            args: lines(arguments),
            environment: pairs(environment),
            headers: pairs(headers),
            enabled: enabled
        )
        Task {
            _ = await save(draft)
            isSaving = false
        }
    }

    private func lines(_ value: String) -> [String] {
        value.split(whereSeparator: \.isNewline).map(String.init).filter { !$0.isEmpty }
    }

    private func pairs(_ value: String) -> [String: String] {
        var result: [String: String] = [:]
        for line in lines(value) {
            guard let separator = line.firstIndex(of: "=") else { continue }
            let key = String(line[..<separator]).trimmingCharacters(in: .whitespaces)
            let item = String(line[line.index(after: separator)...])
            if !key.isEmpty { result[key] = item }
        }
        return result
    }
}

private struct SkillEditor: View {
    @Environment(\.dismiss) private var dismiss
    @State private var name: String
    @State private var instructions: String
    @State private var enabled: Bool
    @State private var isSaving = false
    let save: (String, String, Bool) async -> Bool

    init(skill: SkillSummary?, save: @escaping (String, String, Bool) async -> Bool) {
        _name = State(initialValue: skill?.name ?? "")
        _instructions = State(initialValue: skill?.instructions ?? "")
        _enabled = State(initialValue: skill?.enabled ?? true)
        self.save = save
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Reusable instructions").font(.title3.weight(.semibold))
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Save") { saveSkill() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!isValid || isSaving)
            }
            .padding(18)
            Divider()
            VStack(alignment: .leading, spacing: 12) {
                TextField("Name", text: $name)
                    .textFieldStyle(.roundedBorder)
                Text("Instructions")
                    .font(.headline)
                TextEditor(text: $instructions)
                    .font(.body)
                    .padding(8)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(.separator))
                Toggle("Enabled in chats", isOn: $enabled)
            }
            .padding(18)
        }
        .frame(width: 620, height: 480)
    }

    private var isValid: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func saveSkill() {
        isSaving = true
        Task {
            _ = await save(
                name.trimmingCharacters(in: .whitespacesAndNewlines),
                instructions.trimmingCharacters(in: .whitespacesAndNewlines),
                enabled
            )
            isSaving = false
        }
    }
}
