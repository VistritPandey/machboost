import XCTest

final class MachBoostUITests: XCTestCase {
    @MainActor
    func testPrimaryWorkspaceAndNavigationAreReachable() {
        let app = launchApp()
        XCTAssertTrue(app.staticTexts["MachBoost"].waitForExistence(timeout: 10))

        if app.buttons["Continue"].exists {
            app.buttons["Continue"].click()
        }

        XCTAssertTrue(app.staticTexts["Chats"].exists)
        XCTAssertTrue(app.staticTexts["Connections"].exists)
        XCTAssertTrue(app.staticTexts["Models"].exists)
        XCTAssertTrue(app.staticTexts["Server"].exists)
        XCTAssertTrue(app.staticTexts["Settings"].exists)
    }

    @MainActor
    func testInferenceConnectionsExposeLocalAndTeamModes() {
        let app = launchApp()
        let connections = app.staticTexts["Connections"]
        XCTAssertTrue(connections.waitForExistence(timeout: 10))
        connections.click()

        XCTAssertTrue(app.staticTexts["Inference source"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.radioButtons["This Mac"].exists)
        XCTAssertTrue(app.radioButtons["Team host"].exists)
        XCTAssertTrue(app.staticTexts["Repository tools and model inference both run on this Mac."].exists)
    }

    @MainActor
    func testRepositoryPickerExposesWorkspaceActions() {
        let app = launchApp()
        let repository = app.descendants(matching: .any)["repository-picker"]

        XCTAssertTrue(repository.waitForExistence(timeout: 20))
        focus(repository)

        XCTAssertTrue(app.menuItems["No Repository"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.menuItems["Open Repository..."].exists)
    }

    @MainActor
    func testServerDeveloperSurfaceOpens() {
        let app = launchApp()
        XCTAssertTrue(app.staticTexts["Server"].waitForExistence(timeout: 10))
        app.staticTexts["Server"].click()
        let developerTab = app.radioButtons["Developer"]
        XCTAssertTrue(developerTab.waitForExistence(timeout: 3))
        developerTab.click()
        XCTAssertTrue(
            app.descendants(matching: .any)["developer-endpoint-section"]
                .waitForExistence(timeout: 3)
        )
        XCTAssertTrue(app.staticTexts["Local endpoint"].waitForExistence(timeout: 3))
        XCTAssertTrue(
            app.staticTexts["127.0.0.1 is reachable only from this Mac."]
                .waitForExistence(timeout: 3)
        )
        XCTAssertTrue(
            app.buttons["Enable authenticated LAN access"].waitForExistence(timeout: 3)
        )
        XCTAssertTrue(app.staticTexts["OpenAI Responses"].exists)
        XCTAssertTrue(app.staticTexts["Anthropic Messages"].exists)
        XCTAssertTrue(app.staticTexts["P50 latency"].exists)
        XCTAssertTrue(app.staticTexts["240 ms"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.staticTexts["P95 latency"].exists)
        XCTAssertTrue(app.staticTexts["380 ms"].exists)
    }

    @MainActor
    func testServerCanLoadAndWarmAResidentModel() {
        let app = launchApp()
        XCTAssertTrue(app.staticTexts["Server"].waitForExistence(timeout: 10))
        app.staticTexts["Server"].click()
        let developerTab = app.radioButtons["Developer"]
        XCTAssertTrue(developerTab.waitForExistence(timeout: 3))
        developerTab.click()

        let load = app.buttons["load-resident-model"]
        XCTAssertTrue(load.waitForExistence(timeout: 3))
        load.click()

        XCTAssertTrue(app.staticTexts["Resident model ready"].waitForExistence(timeout: 3))
    }

    @MainActor
    func testTeamGatewayAndTraceControlsOpen() {
        let app = launchApp()
        XCTAssertTrue(app.staticTexts["Server"].waitForExistence(timeout: 10))
        app.staticTexts["Server"].click()

        let teamTab = app.radioButtons["Team"]
        XCTAssertTrue(teamTab.waitForExistence(timeout: 3))
        teamTab.click()
        XCTAssertTrue(
            app.descendants(matching: .any)["team-key-section"]
                .waitForExistence(timeout: 3)
        )
        XCTAssertTrue(app.staticTexts["Create employee key"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.textFields["Name"].exists)
        XCTAssertTrue(app.staticTexts["Coding fleet readiness"].exists)
        XCTAssertTrue(app.staticTexts["Team environment"].exists)

        let logsTab = app.radioButtons["Logs & evals"]
        XCTAssertTrue(logsTab.exists)
        logsTab.click()
        XCTAssertTrue(app.staticTexts["Trace policy"].exists)
        XCTAssertTrue(app.staticTexts["Request traces"].exists)
        XCTAssertTrue(app.buttons["Save policy"].exists)
    }

    @MainActor
    func testChatStreamsResponseAndPerformanceStatistics() {
        let app = launchApp()
        let composer = app.textFields["Message MachBoost"]
        XCTAssertTrue(composer.waitForExistence(timeout: 10))

        focus(composer)
        composer.typeText("Hello from native UI automation")
        app.buttons["Send message"].click()

        XCTAssertTrue(app.staticTexts["Fixture response."].waitForExistence(timeout: 8))
        XCTAssertTrue(app.staticTexts["20.0 tok/s"].exists)
        XCTAssertTrue(app.staticTexts["0.12s TTFT"].exists)
    }

    @MainActor
    func testMuseChatShowsReasoningControlsAndToolCalls() {
        let app = launchApp(environment: [
            "MACHBOOST_UI_TEST_MODEL": "muse-glimmer:30b"
        ])
        let controls = app.buttons["Generation controls"]
        XCTAssertTrue(controls.waitForExistence(timeout: 10))
        controls.click()

        XCTAssertTrue(app.staticTexts["Reasoning"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.staticTexts["Context window"].exists)
        controls.click()

        send("Use Muse tools", in: app)

        XCTAssertTrue(
            app.disclosureTriangles["message-reasoning"].waitForExistence(timeout: 8)
        )
        XCTAssertTrue(
            app.descendants(matching: .any)["tool-call-search_repository"]
                .waitForExistence(timeout: 3)
        )
        XCTAssertTrue(app.staticTexts["Fixture response."].waitForExistence(timeout: 3))
    }

    @MainActor
    func testCodingAgentShowsMultiRoundActivityAndReviewableChanges() {
        let app = launchApp(environment: [
            "MACHBOOST_UI_TEST_MODEL": "muse-glimmer:30b",
            "MACHBOOST_UI_TEST_CODING": "1",
            "MACHBOOST_UI_TEST_PERMISSION_MODE": "manual",
        ])
        let repository = app.descendants(matching: .any)["repository-picker"]
        XCTAssertTrue(repository.waitForExistence(timeout: 10))
        focus(repository)
        let fixture = app.menuItems["MachBoost fixture"]
        XCTAssertTrue(fixture.waitForExistence(timeout: 3))
        fixture.click()
        let permissionMode = app.descendants(matching: .any)["coding-permission-mode"]
        XCTAssertTrue(permissionMode.waitForExistence(timeout: 3))
        XCTAssertEqual(permissionMode.value as? String, "Manual")

        send("Exercise coding agent", in: app)

        XCTAssertTrue(
            app.descendants(matching: .any)["tool-call-list_files"]
                .waitForExistence(timeout: 15)
        )
        XCTAssertTrue(
            app.descendants(matching: .any)["tool-call-read_file"]
                .waitForExistence(timeout: 10)
        )
        let approval = app.sheets.buttons["Apply Change"]
        XCTAssertTrue(approval.waitForExistence(timeout: 10))
        approval.click()

        XCTAssertTrue(
            app.descendants(matching: .any)["tool-call-replace_in_file"]
                .waitForExistence(timeout: 10)
        )
        XCTAssertTrue(
            app.staticTexts["Reviewed the repository after three tool results."]
                .waitForExistence(timeout: 10)
        )
        let changes = app.buttons["code-changes"]
        XCTAssertTrue(changes.waitForExistence(timeout: 3))
        changes.click()
        XCTAssertTrue(app.buttons["Open File"].waitForExistence(timeout: 3))
        XCTAssertTrue(app.buttons["Reveal"].exists)
        let patch = app.staticTexts["change-patch-edit-1"]
        XCTAssertTrue(patch.waitForExistence(timeout: 3))
        XCTAssertTrue(
            app.descendants(matching: .any)["workspace-changes-panel"]
                .waitForExistence(timeout: 5)
        )
        XCTAssertTrue(app.staticTexts["main → working tree"].exists)
        XCTAssertTrue(app.staticTexts["Sources/App.swift"].exists)
        XCTAssertFalse(
            app.staticTexts.containing(
                NSPredicate(format: "label CONTAINS %@", "<tool_call")
            ).firstMatch.exists
        )
    }

    @MainActor
    func testLongMarkdownStreamKeepsItsEndVisible() {
        let app = launchApp()
        send("Show a long Markdown response", in: app)

        let endMarker = app.staticTexts["STREAM END MARKER"]
        XCTAssertTrue(endMarker.waitForExistence(timeout: 8))
        let visible = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "hittable == true"),
            object: endMarker
        )
        XCTAssertEqual(XCTWaiter.wait(for: [visible], timeout: 3), .completed)
    }

    @MainActor
    func testChatGenerationCanBeStopped() {
        let app = launchApp()
        let composer = app.textFields["Message MachBoost"]
        XCTAssertTrue(composer.waitForExistence(timeout: 10))

        focus(composer)
        composer.typeText("Stop this fixture response")
        app.buttons["Send message"].click()
        let stop = app.buttons["stop-generation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 2))
        app.activate()
        let stopReady = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "hittable == true"),
            object: stop
        )
        XCTAssertEqual(XCTWaiter.wait(for: [stopReady], timeout: 5), .completed)
        stop.click()

        XCTAssertTrue(app.staticTexts["Stopped"].waitForExistence(timeout: 5))
    }

    @MainActor
    func testModelDownloadRequiresConfirmationAndUpdatesCatalog() {
        let app = launchApp()
        XCTAssertTrue(app.staticTexts["Models"].waitForExistence(timeout: 10))
        app.staticTexts["Models"].firstMatch.click()

        let download = app.buttons["Download Llama 3.2 1B"]
        XCTAssertTrue(download.waitForExistence(timeout: 3))
        download.click()
        let confirmation = app.sheets.buttons["Download"]
        XCTAssertTrue(confirmation.waitForExistence(timeout: 2))
        confirmation.click()

        let finished = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "exists == false"),
            object: download
        )
        XCTAssertEqual(XCTWaiter.wait(for: [finished], timeout: 5), .completed)
        XCTAssertTrue(app.staticTexts["Downloaded"].firstMatch.exists)
    }

    @MainActor
    func testChatModelBrowserSearchesNativeModels() {
        let app = launchApp()
        let picker = app.buttons["chat-model-picker"]
        XCTAssertTrue(picker.waitForExistence(timeout: 10))
        picker.click()

        let search = app.textFields["Search models"]
        XCTAssertTrue(search.waitForExistence(timeout: 3))
        XCTAssertTrue(app.staticTexts["MLX native models"].exists)
        search.typeText("Muse")

        XCTAssertTrue(app.buttons["Muse Glimmer 30B, ready"].waitForExistence(timeout: 3))
        XCTAssertFalse(app.buttons["Qwen2.5 3B, ready"].exists)
    }

    @MainActor
    func testFirstRunRequiresConfirmedModelDownload() {
        let app = launchApp(environment: ["MACHBOOST_UI_TEST_NO_CACHED_MODELS": "1"])
        XCTAssertTrue(app.staticTexts["Choose your first model"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.buttons["Continue"].exists)

        let download = app.buttons["onboarding-download-model"]
        XCTAssertTrue(download.waitForExistence(timeout: 3))
        download.click()
        let confirmation = app.sheets.buttons["Download"]
        XCTAssertTrue(confirmation.waitForExistence(timeout: 2))
        confirmation.click()

        let continueButton = app.buttons["Continue"]
        XCTAssertTrue(continueButton.waitForExistence(timeout: 5))
        continueButton.click()
        XCTAssertTrue(app.staticTexts["Chats"].waitForExistence(timeout: 3))
    }

    @MainActor
    func testVisionAndTextContextRemainPinnedAcrossPrompts() {
        let app = launchApp(environment: [
            "MACHBOOST_UI_TEST_MODEL": "qwen2.5-vl:3b",
            "MACHBOOST_UI_TEST_ATTACHMENTS": "image,text",
        ])
        XCTAssertTrue(app.staticTexts["fixture-image.png"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.staticTexts["fixture-context.txt"].exists)

        send("Inspect the attached material", in: app)
        let responses = app.staticTexts.matching(
            identifier: "Fixture context: 1 image, 1 file."
        )
        XCTAssertTrue(responses.firstMatch.waitForExistence(timeout: 8))

        send("Now answer a separate follow-up", in: app)
        XCTAssertTrue(responses.element(boundBy: 1).waitForExistence(timeout: 8))
    }

    @MainActor
    func testResponseCanBeRegenerated() {
        let app = launchApp()
        send("Generate this response twice", in: app)
        let response = app.staticTexts["Fixture response."]
        XCTAssertTrue(response.waitForExistence(timeout: 8))

        let regenerate = app.buttons["regenerate-response"]
        XCTAssertTrue(regenerate.waitForExistence(timeout: 2))
        regenerate.click()
        XCTAssertTrue(
            app.staticTexts["Regenerated fixture response."].waitForExistence(timeout: 8)
        )
    }

    @MainActor
    func testConversationHistoryCanBeSearched() {
        let app = launchApp()
        let title = "Quarterly roadmap needle"
        send(title, in: app)
        XCTAssertTrue(app.staticTexts["Fixture response."].waitForExistence(timeout: 8))

        let search = app.searchFields["Search chats"]
        XCTAssertTrue(search.waitForExistence(timeout: 3))
        focus(search)
        search.typeText("roadmap needle")
        XCTAssertTrue(app.staticTexts[title].firstMatch.waitForExistence(timeout: 3))
    }

    @MainActor
    func testSettingsExposeLoginAndUpdateControls() {
        let app = launchApp()
        XCTAssertTrue(app.staticTexts["Settings"].waitForExistence(timeout: 10))
        app.staticTexts["Settings"].click()

        let controls = app.descendants(matching: .any)
        XCTAssertTrue(controls["launch-at-login"].waitForExistence(timeout: 3))
        let automaticUpdates = app.switches["automatic-updates"]
        XCTAssertTrue(automaticUpdates.exists)
        XCTAssertEqual(String(describing: automaticUpdates.value ?? ""), "1")
        XCTAssertTrue(app.buttons["check-for-updates"].exists)
        XCTAssertTrue(
            app.staticTexts[
                "Checks GitHub Releases; community installation is manual"
            ].exists
        )
        XCTAssertTrue(app.staticTexts["Telemetry"].exists)
        XCTAssertTrue(app.staticTexts["Disabled"].exists)
    }

    @MainActor
    private func launchApp(environment: [String: String] = [:]) -> XCUIApplication {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchEnvironment["MACHBOOST_UI_TESTING"] = "1"
        app.launchEnvironment["MACHBOOST_SOURCE_ROOT"] = repositoryRoot()
        for (key, value) in environment {
            app.launchEnvironment[key] = value
        }
        app.launch()
        app.activate()
        return app
    }

    nonisolated private func repositoryRoot() -> String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .path
    }

    @MainActor
    private func focus(_ element: XCUIElement) {
        element.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5)).click()
    }

    @MainActor
    private func send(_ message: String, in app: XCUIApplication) {
        let composer = app.textFields["Message MachBoost"]
        XCTAssertTrue(composer.waitForExistence(timeout: 10))
        focus(composer)
        composer.typeText(message)
        app.buttons["Send message"].click()
    }
}
