const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
    testDir: "./tests",

    use: {
        baseURL: "http://127.0.0.1:8000",
        headless: false,
        screenshot: "only-on-failure",
        trace: "retain-on-failure"
    },

    timeout: 30_000
});