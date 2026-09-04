const { test, expect } = require("@playwright/test");

test("customer can complete a full trade-in valuation", async ({ page }) => {
    // =========================================================
    // 1. CUSTOMER DETAILS
    // =========================================================

    await page.goto("/customer-detail");

    await expect(page).toHaveURL(/\/customer-detail$/);

    await page.locator("#customerName").fill("Playwright Test");
    await page.locator("#customerPhone").fill("0123456789");
    await page.locator("#customerEmail").fill("playwright@test.com");

    await page.locator("#customerContactMethod").selectOption("whatsapp");

    await page.getByRole("button", { name: /start my trade-in/i }).click();

    // Should redirect to the main trade-in page.
    await expect(page).toHaveURL(/\/$/);


    // =========================================================
    // 2. WAIT FOR DEVICE DATA
    // =========================================================

    await expect(page.locator("#deviceBoxGrid .device-box").first())
        .toBeVisible();


    // =========================================================
    // 3. SELECT DEVICE
    // =========================================================

    // Select iPhone for the baseline test.
    const iphoneBox = page.locator(
        '#deviceBoxGrid .device-box[data-device="iPhone"]'
    );

    await expect(iphoneBox).toBeVisible();

    await iphoneBox.click();


    // =========================================================
    // 4. SELECT PRODUCT LINE
    // =========================================================

    await expect(page.locator("#subDeviceBoxGrid .subdevice-box").first())
        .toBeVisible();

    // Select the first available product line.
    const firstSubDevice = page.locator(
        "#subDeviceBoxGrid .subdevice-box"
    ).first();

    await firstSubDevice.click();


    // =========================================================
    // 5. SELECT MODEL
    // =========================================================

    await expect(page.locator("#modelSelect"))
        .toBeEnabled();

    const modelOptions = page.locator(
        "#modelSelect option:not([disabled])"
    );

    await expect(modelOptions.first()).toBeAttached();

    const modelValue = await modelOptions.first().getAttribute("value");

    expect(modelValue).not.toBeNull();
    expect(modelValue).not.toBe("");

    await page.locator("#modelSelect").selectOption(modelValue);


    // =========================================================
    // 6. CONFIGURATION
    // =========================================================

    // Storage may or may not exist depending on the selected model.
    const storageGroup = page.locator("#storageGroup");

    if (!(await storageGroup.evaluate(el => el.classList.contains("hidden")))) {
        await expect(page.locator("#storageSelect"))
            .toBeEnabled();

        const storageOptions = page.locator(
            "#storageSelect option:not([disabled])"
        );

        const storageValue =
            await storageOptions.first().getAttribute("value");

        expect(storageValue).not.toBeNull();
        expect(storageValue).not.toBe("");

        await page.locator("#storageSelect")
            .selectOption(storageValue);
    }


    // Storage Type, if applicable.
    const storageTypeGroup = page.locator("#storageTypeGroup");

    if (
        !(await storageTypeGroup.evaluate(
            el => el.classList.contains("hidden")
        ))
    ) {
        const options = page.locator(
            "#storageTypeSelect option:not([disabled])"
        );

        const value =
            await options.first().getAttribute("value");

        expect(value).not.toBeNull();
        expect(value).not.toBe("");

        await page.locator("#storageTypeSelect")
            .selectOption(value);
    }


    // Connectivity, if applicable.
    const connectivityGroup = page.locator("#connectivityGroup");

    if (
        !(await connectivityGroup.evaluate(
            el => el.classList.contains("hidden")
        ))
    ) {
        const options = page.locator(
            "#connectivitySelect option:not([disabled])"
        );

        const value =
            await options.first().getAttribute("value");

        expect(value).not.toBeNull();
        expect(value).not.toBe("");

        await page.locator("#connectivitySelect")
            .selectOption(value);
    }


    // =========================================================
    // 7. CONDITION
    // =========================================================

    // Wait for condition section to become available.
    await expect(page.locator("#conditionSection"))
        .not.toHaveClass(/hidden/);

    // iPhone condition fields.
    await page.locator("#iphoneScreen").selectOption({
        index: 1
    });

    await page.locator("#iphoneBody").selectOption({
        index: 1
    });

    await page.locator("#iphoneBattery").selectOption({
        index: 1
    });


    // =========================================================
    // 8. SUBMIT
    // =========================================================

    // Watch the valuation API request.
    const predictRequestPromise = page.waitForRequest(
        request =>
            request.url().endsWith("/predict") &&
            request.method() === "POST"
    );

    // Watch the customer save request.
    const saveRequestPromise = page.waitForRequest(
        request =>
            request.url().endsWith("/customer/trade-in") &&
            request.method() === "POST"
    );

    await page.locator("#submitBtn").click();

    const predictRequest = await predictRequestPromise;
    const saveRequest = await saveRequestPromise;


    // =========================================================
    // 9. VERIFY /predict PAYLOAD
    // =========================================================

    const predictPayload =
        predictRequest.postDataJSON();

    expect(predictPayload.Device)
        .toBe("iPhone");

    expect(predictPayload.SubDevice)
        .not.toBe("");

    expect(predictPayload.Model)
        .not.toBe("");


    // =========================================================
    // 10. VERIFY CUSTOMER SAVE PAYLOAD
    // =========================================================

    const savePayload =
        saveRequest.postDataJSON();

    expect(savePayload.customer.name)
        .toBe("Playwright Test");

    expect(savePayload.customer.phone)
        .toBe("0123456789");

    expect(savePayload.customer.email)
        .toBe("playwright@test.com");

    expect(savePayload.device.device)
        .toBe("iPhone");

    expect(savePayload.device.subDevice)
        .not.toBe("");

    expect(savePayload.device.model)
        .not.toBe("");


    // =========================================================
    // 11. VERIFY RESULT
    // =========================================================

    await expect(page.locator("#resultCard"))
        .not.toHaveClass(/hidden/);

    await expect(page.locator("#resultValue"))
        .not.toContainText("Calculating...");

    await expect(page.locator("#resultValue"))
        .not.toContainText("Unable to calculate");

    console.log(
        "Final result:",
        await page.locator("#resultValue").innerText()
    );
});