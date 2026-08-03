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
        XCTAssertTrue(app.staticTexts["Models"].exists)
        XCTAssertTrue(app.staticTexts["Server"].exists)
        XCTAssertTrue(app.staticTexts["Settings"].exists)
    }

    @MainActor
    func testRepositoryPickerExposesWorkspaceActions() {
        let app = launchApp()
        let repository = app.descendants(matching: .any)["repository-picker"]

        XCTAssertTrue(repository.waitForExistence(timeout: 20))
        repository.click()

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
        XCTAssertTrue(app.staticTexts["Local endpoint"].exists)
        XCTAssertTrue(app.staticTexts["127.0.0.1 is reachable only from this Mac."].exists)
        XCTAssertTrue(app.buttons["Enable authenticated LAN access"].exists)
        XCTAssertTrue(app.staticTexts["OpenAI Python"].exists)
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
        XCTAssertTrue(app.staticTexts["Create employee key"].exists)
        XCTAssertTrue(app.textFields["Name"].exists)
        XCTAssertTrue(app.staticTexts["Coding agent environment"].exists)

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
    func testChatGenerationCanBeStopped() {
        let app = launchApp()
        let composer = app.textFields["Message MachBoost"]
        XCTAssertTrue(composer.waitForExistence(timeout: 10))

        focus(composer)
        composer.typeText("Stop this fixture response")
        app.buttons["Send message"].click()
        let stop = app.buttons["Stop generation"]
        XCTAssertTrue(stop.waitForExistence(timeout: 2))
        let stopReady = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "hittable == true"),
            object: stop
        )
        XCTAssertEqual(XCTWaiter.wait(for: [stopReady], timeout: 2), .completed)
        app.activate()
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
        XCTAssertTrue(controls["automatic-updates"].exists)
        XCTAssertTrue(app.buttons["View latest release"].exists)
        XCTAssertTrue(
            app.staticTexts["Community builds update through GitHub Releases"].exists
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
