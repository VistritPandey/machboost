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
    func testServerDeveloperSurfaceOpens() {
        let app = launchApp()
        XCTAssertTrue(app.staticTexts["Server"].waitForExistence(timeout: 10))
        app.staticTexts["Server"].click()
        let developerTab = app.radioButtons["Developer"]
        XCTAssertTrue(developerTab.waitForExistence(timeout: 3))
        developerTab.click()
        XCTAssertTrue(app.staticTexts["Endpoint"].exists)
        XCTAssertTrue(app.staticTexts["OpenAI Python"].exists)
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
    private func launchApp() -> XCUIApplication {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchEnvironment["MACHBOOST_UI_TESTING"] = "1"
        app.launchEnvironment["MACHBOOST_SOURCE_ROOT"] = repositoryRoot()
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
}
