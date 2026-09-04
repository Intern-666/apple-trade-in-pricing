let deviceData = {};
let configData = {};

/* =========================================================
   CUSTOMER DETAILS
========================================================= */

const customerDetails = JSON.parse(
    sessionStorage.getItem("customerDetails") || "null"
);

if (!customerDetails) {
    window.location.href = "/customer-detail";
}

/* =========================================================
   DEVICE IMAGE MAP
   Keyed by Device -> Sub-device (not by exact model name),
   so new models under an existing sub-device need no changes.
   Keys are normalized via slugify() to tolerate casing/spacing
   differences in the CSV data. Each device has a "default"
   used when its sub-device isn't in the map yet.
========================================================= */

const DEVICE_IMAGE_MAP = {
    "iphone": {
        "standard": "iphone/standard.png",
        "plus": "iphone/plus.png",
        "pro": "iphone/pro.png",
        "pro-max": "iphone/pro-max.png",
        "mini": "iphone/mini.png",
        "se": "iphone/se.png",
        "air": "iphone/air.png",
        "default": "iphone/base.png"
    },
    "ipad": {
        // sub-device "iPad" (base line) also slugifies to "ipad" --
        // same as the device-level key below, but they're separate
        // map levels so this is intentional, not a collision.
        "ipad": "ipad/ipad.png",
        "ipad-air": "ipad/air.png",
        "ipad-pro": "ipad/pro.png",
        "ipad-mini": "ipad/mini.png",
        "default": "ipad/base.png"
    },
    "mac": {
        "macbook": "mac/macbook.png",
        "macbook-air": "mac/macbook-air.png",
        "macbook-pro": "mac/macbook-pro.png",
        "imac": "mac/imac.png",
        "imac-pro": "mac/imac-pro.png",
        "mac-mini": "mac/mac-mini.png",
        "mac-studio": "mac/mac-studio.png",
        "mac-pro": "mac/mac-pro.png",
        "default": "mac/base.png"
    },
    "apple-watch": {
        "watch-se": "watch/se.png",
        "watch-series": "watch/series.png",
        "watch-ultra": "watch/ultra.png",
        "default": "watch/base.png"
    },
    "airpods": {
        "airpods": "airpods/standard.png",
        "airpods-pro": "airpods/pro.png",
        "airpods-max": "airpods/max.png",
        "default": "airpods/base.png"
    }
};

const DEVICE_IMAGE_FALLBACK = "/assets/devices/unknown.png";

function slugify(text) {

    return String(text)
        .toLowerCase()
        .trim()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "");
}

/* Resolves an image path for a device (and optionally a sub-device).
   Device-only lookup (subDevice omitted) returns that device's
   "default" image -- used for the step-1 device boxes, before a
   sub-device is known. Returns null if the device itself isn't
   mapped, so callers can fall back to DEVICE_IMAGE_FALLBACK. */
function resolveDeviceImage(device, subDevice) {

    const deviceMap =
        DEVICE_IMAGE_MAP[slugify(device)];

    if (!deviceMap) {
        return null;
    }

    if (!subDevice) {
        return deviceMap["default"] || null;
    }

    return (
        deviceMap[slugify(subDevice)] ||
        deviceMap["default"] ||
        null
    );
}

function updateDeviceImagePreview(device, subDevice) {

    const preview = document.getElementById("deviceImagePreview");
    const img = document.getElementById("deviceImagePreviewImg");

    if (!device || !subDevice) {

        preview.classList.add("hidden");
        return;
    }

    const relativeSrc =
        resolveDeviceImage(device, subDevice);

    img.src = relativeSrc
        ? `/assets/devices/${relativeSrc}`
        : DEVICE_IMAGE_FALLBACK;

    img.alt = `${device} ${subDevice}`;

    img.onerror = function () {
        img.onerror = null;
        img.src = DEVICE_IMAGE_FALLBACK;
    };

    preview.classList.remove("hidden");
}

/* =========================================================
   ELEMENTS
========================================================= */

const deviceSelect = document.getElementById("deviceSelect");
const subDeviceSelect = document.getElementById("subDeviceSelect");
const modelSelect = document.getElementById("modelSelect");
const storageSelect = document.getElementById("storageSelect");

const storageTypeSelect = document.getElementById("storageTypeSelect");
const connectivitySelect = document.getElementById("connectivitySelect");

const storageGroup = document.getElementById("storageGroup");
const storageTypeGroup = document.getElementById("storageTypeGroup");
const connectivityGroup = document.getElementById("connectivityGroup");

const deviceBoxGrid = document.getElementById("deviceBoxGrid");
const subDeviceBoxGrid = document.getElementById("subDeviceBoxGrid");

const stepIndicator = document.getElementById("stepIndicator");
const stepViewport = document.getElementById("stepViewport");
const configurationSection = document.getElementById("configurationSection");
const tradeInCard = document.querySelector(".trade-in-card");

const tradeInForm = document.getElementById("tradeInForm");
const conditionSection = document.getElementById("conditionSection");

const resultCard = document.getElementById("resultCard");
const resultValue = document.getElementById("resultValue");
const submitBtn = document.getElementById("submitBtn");
const startOverBtn = document.getElementById("startOverBtn");

const productLineSection = document.getElementById("productLineSection");
const modelSection = document.getElementById("modelSection");


/* =========================================================
   STEP CONTROLLER
========================================================= */

let currentStep = 1;
let maxUnlockedStep = 1;

// Whether step 4 (Configuration) has anything to show for the
// currently selected device -- false for AirPods, or any device
// with no storage/type/connectivity options at all. Recomputed
// whenever a Model is chosen.
let step4Applicable = true;

function getStepPanel(step) {
    return document.querySelector(
        `.step-panel[data-step="${step}"]`
    );
}

function getStepIndicatorItem(step) {
    return stepIndicator.querySelector(
        `.step-indicator-item[data-step="${step}"]`
    );
}

/* Shows `targetStep`'s panel, hides the current one, and plays
   the slide+fade transition in the given direction. Does NOT
   change unlock state -- call unlockStep() separately when a
   step is newly reached for the first time. */
function goToStep(targetStep, direction) {

    const outgoing = getStepPanel(currentStep);
    const incoming = getStepPanel(targetStep);

    if (!incoming) {
        return;
    }

    if (outgoing && outgoing !== incoming) {
        outgoing.classList.remove(
            "step-panel-active",
            "step-panel-enter-right",
            "step-panel-enter-left"
        );
    }

    incoming.classList.remove(
        "step-panel-enter-right",
        "step-panel-enter-left"
    );

    incoming.classList.add("step-panel-active");

    // Force reflow so the enter animation replays even if the
    // same class was just removed (e.g. rapid back-and-forth).
    void incoming.offsetWidth;

    incoming.classList.add(
        direction === "back"
            ? "step-panel-enter-left"
            : "step-panel-enter-right"
    );

    currentStep = targetStep;

    refreshStepIndicator();

    scrollToElement(stepViewport);
}

function unlockStep(step) {

    if (step > maxUnlockedStep) {
        maxUnlockedStep = step;
    }

    refreshStepIndicator();
}

/* Clears all step data/selections AFTER `step` (exclusive) and
   locks the indicator back down to `step`. Used whenever an
   earlier choice invalidates everything downstream -- e.g.
   changing Device resets Product Line, Model, Configuration,
   and Condition. */
function resetFromStep(step) {

    if (step < 4) {

        storageTypeSelect.innerHTML =
            '<option value="" disabled selected>Select storage type...</option>';

        connectivitySelect.innerHTML =
            '<option value="" disabled selected>Select connectivity...</option>';

        storageTypeGroup.classList.add("hidden");
        connectivityGroup.classList.add("hidden");

        storageSelect.innerHTML =
            '<option value="" disabled selected>Select storage...</option>';

        storageSelect.disabled = true;
    }

    if (step < 5) {

        hideConditionProfiles();
    }

    maxUnlockedStep = Math.min(maxUnlockedStep, step);

    refreshStepIndicator();
}

/* Given the current device, returns the ordered list of step
   numbers actually reachable in the flow. Step 4 is omitted
   only when step4Applicable is false (AirPods today, or any
   device with no configuration fields at all) -- the indicator
   still shows all 5 slots per design, this list is just for
   deciding where Next/skip logic should land. */
function getReachableSteps() {

    const steps = [1, 2, 3, 5];

    if (step4Applicable) {
        steps.splice(3, 0, 4);
    }

    return steps.sort((a, b) => a - b);
}

function refreshStepIndicator() {

    for (let step = 1; step <= 5; step++) {

        const item = getStepIndicatorItem(step);

        if (!item) {
            continue;
        }

        const isUnlocked =
            step <= maxUnlockedStep &&
            (step !== 4 || step4Applicable);

        const isCurrent = step === currentStep;
        const isCompleted = isUnlocked && step < currentStep;

        item.classList.toggle("unlocked", isUnlocked);
        item.classList.toggle("current", isCurrent);
        item.classList.toggle("completed", isCompleted);

        const button = item.querySelector(
            ".step-indicator-button"
        );

        if (button) {
            button.disabled = !isUnlocked;
        }
    }
}

stepIndicator.addEventListener("click", function (event) {

    const button = event.target.closest(
        "[data-goto-step]"
    );

    if (!button || button.disabled) {
        return;
    }

    const targetStep =
        parseInt(button.dataset.gotoStep, 10);

    if (targetStep > maxUnlockedStep) {
        return;
    }

    if (targetStep === currentStep) {
        return;
    }

    goToStep(
        targetStep,
        targetStep < currentStep ? "back" : "forward"
    );
});


/* =========================================================
   INITIAL LOAD
========================================================= */

window.addEventListener("DOMContentLoaded", async function () {

    const customerDetails = sessionStorage.getItem("customerDetails");

    if (!customerDetails) {
        window.location.replace("/customer-detail");
        return;
    }

    try {

        const [modelsResponse, configResponse] = await Promise.all([
            fetch("/available-models"),
            fetch("/model-configuration")
        ]);

        if (!modelsResponse.ok) {
            throw new Error(
                `Failed to load models: ${modelsResponse.status}`
            );
        }

        deviceData = await modelsResponse.json();

        // Configuration data is additive -- if this call fails for
        // any reason, Configuration step simply won't show
        // Storage Type / Connectivity fields, but Storage and the
        // rest of the flow still work normally.
        configData = configResponse.ok
            ? await configResponse.json()
            : {};

        deviceSelect.innerHTML =
            '<option value="" disabled selected>Select a device...</option>';

        deviceBoxGrid.innerHTML = "";

        for (const device in deviceData) {

            const option = document.createElement("option");

            option.value = device;
            option.textContent = device;

            deviceSelect.appendChild(option);

            deviceBoxGrid.appendChild(
                buildDeviceBox(device)
            );
        }

        getStepPanel(1).classList.add("step-panel-active");

        refreshStepIndicator();

        updateProgress();

    } catch (error) {

        console.error(
            "Failed to load dynamic models:",
            error
        );

        deviceSelect.innerHTML =
            '<option value="" disabled selected>Unable to load devices</option>';

        deviceBoxGrid.innerHTML =
            '<p class="field-help">Unable to load devices.</p>';

        // Even on failure, step 1 must still be reachable --
        // otherwise the whole indicator stays permanently locked
        // with no way to retry or see what went wrong.
        getStepPanel(1).classList.add("step-panel-active");

        refreshStepIndicator();
    }
});


/* =========================================================
   BUILD A DEVICE BOX
========================================================= */

function buildDeviceBox(device) {

    const box = document.createElement("div");

    box.className = "device-box";
    box.setAttribute("role", "option");
    box.setAttribute("tabindex", "0");
    box.setAttribute("aria-selected", "false");
    box.dataset.device = device;

    const imageSrc = resolveDeviceImage(device);

    if (imageSrc) {

        const img = document.createElement("img");

        img.src = `/assets/devices/${imageSrc}`;
        img.alt = "";
        img.className = "device-box-image";

        img.addEventListener("error", function () {

            img.replaceWith(
                buildIconFallback(device)
            );

        });

        box.appendChild(img);

    } else {

        box.appendChild(
            buildIconFallback(device)
        );
    }

    const label = document.createElement("span");

    label.className = "device-box-label";
    label.textContent = device;

    box.appendChild(label);

    box.addEventListener("click", function () {
        selectDeviceBox(device);
    });

    box.addEventListener("keydown", function (event) {

        if (event.key === "Enter" || event.key === " ") {

            event.preventDefault();
            selectDeviceBox(device);
        }
    });

    return box;
}

function buildIconFallback(device) {

    const fallback = document.createElement("span");

    fallback.className = "device-box-icon-fallback";
    fallback.textContent = device.charAt(0);

    return fallback;
}

function selectDeviceBox(device) {

    deviceSelect.value = device;

    deviceBoxGrid
        .querySelectorAll(".device-box")
        .forEach(box => {

            const isSelected =
                box.dataset.device === device;

            box.classList.toggle(
                "selected",
                isSelected
            );

            box.setAttribute(
                "aria-selected",
                isSelected ? "true" : "false"
            );
        });

    deviceSelect.dispatchEvent(
        new Event("change")
    );
}


/* =========================================================
   DEVICE CHANGE
========================================================= */

deviceSelect.addEventListener("change", function (event) {

    const selectedDevice = event.target.value;

    const subDevices =
        deviceData[selectedDevice] || {};

    subDeviceSelect.innerHTML =
        '<option value="" disabled selected>Select a product line...</option>';

    subDeviceBoxGrid.innerHTML = "";

    for (const sub in subDevices) {

        const option = document.createElement("option");

        option.value = sub;
        option.textContent = sub;

        subDeviceSelect.appendChild(option);

        subDeviceBoxGrid.appendChild(
            buildSubDeviceBox(selectedDevice, sub)
        );
    }

    subDeviceSelect.disabled = false;
    subDeviceBoxGrid.classList.remove("disabled");

    modelSelect.innerHTML =
        '<option value="" disabled selected>Select a model...</option>';

    modelSelect.disabled = true;

    storageSelect.required =
        selectedDevice !== "AirPods";

    resultCard.classList.add("hidden");
    tradeInCard.classList.remove("result-shown");

    updateDeviceImagePreview(null, null);

    // Changing Device invalidates everything downstream --
    // Product Line, Model, Configuration, and Condition.
    resetFromStep(1);
    unlockStep(2);

    updateProgress();

    goToStep(2, "forward");
});


/* =========================================================
   PRODUCT LINE CHANGE
========================================================= */

subDeviceSelect.addEventListener("change", function (event) {

    const selectedDevice =
        deviceSelect.value;

    const selectedSub =
        event.target.value;

    const modelsObj =
        deviceData[selectedDevice][selectedSub] || {};

    updateDeviceImagePreview(
        selectedDevice,
        selectedSub
    );

    modelSelect.innerHTML =
        '<option value="" disabled selected>Select a model...</option>';

    Object.keys(modelsObj)
        .sort()
        .forEach(model => {

            const option =
                document.createElement("option");

            option.value = model;
            option.textContent = model;

            modelSelect.appendChild(option);
        });

    modelSelect.disabled = false;

    resultCard.classList.add("hidden");
    tradeInCard.classList.remove("result-shown");

    // Changing Product Line invalidates Model, Configuration,
    // and Condition.
    resetFromStep(2);
    unlockStep(3);

    updateProgress();

    goToStep(3, "forward");
});


/* =========================================================
   BUILD A PRODUCT LINE BOX
========================================================= */

function buildSubDeviceBox(device, sub) {

    const box = document.createElement("div");

    box.className = "subdevice-box";
    box.setAttribute("role", "option");
    box.setAttribute("tabindex", "0");
    box.setAttribute("aria-selected", "false");
    box.dataset.sub = sub;

    const imageSrc = resolveDeviceImage(device, sub);

    if (imageSrc) {

        const img = document.createElement("img");

        img.src = `/assets/devices/${imageSrc}`;
        img.alt = "";
        img.className = "subdevice-box-image";

        img.addEventListener("error", function () {

            img.replaceWith(
                buildSubDeviceIconFallback(sub)
            );

        });

        box.appendChild(img);

    } else {

        box.appendChild(
            buildSubDeviceIconFallback(sub)
        );
    }

    const label = document.createElement("span");

    label.className = "subdevice-box-label";
    label.textContent = sub;

    box.appendChild(label);

    box.addEventListener("click", function () {
        selectSubDeviceBox(sub);
    });

    box.addEventListener("keydown", function (event) {

        if (event.key === "Enter" || event.key === " ") {

            event.preventDefault();
            selectSubDeviceBox(sub);
        }
    });

    return box;
}

function buildSubDeviceIconFallback(sub) {

    const fallback = document.createElement("span");

    fallback.className = "subdevice-box-icon-fallback";
    fallback.textContent = sub.charAt(0);

    return fallback;
}

function selectSubDeviceBox(sub) {

    subDeviceSelect.value = sub;

    subDeviceBoxGrid
        .querySelectorAll(".subdevice-box")
        .forEach(box => {

            const isSelected =
                box.dataset.sub === sub;

            box.classList.toggle(
                "selected",
                isSelected
            );

            box.setAttribute(
                "aria-selected",
                isSelected ? "true" : "false"
            );
        });

    subDeviceSelect.dispatchEvent(
        new Event("change")
    );
}


/* =========================================================
   MODEL CHANGE
========================================================= */

modelSelect.addEventListener("change", function (event) {

    const selectedDevice =
        deviceSelect.value;

    const selectedSub =
        subDeviceSelect.value;

    const selectedModel =
        event.target.value;

    // Changing Model invalidates Configuration and Condition.
    resetFromStep(3);

    const validStorages =
        deviceData[selectedDevice][selectedSub][selectedModel] || [];

    storageSelect.innerHTML =
        '<option value="" disabled selected>Select storage...</option>';

    validStorages.forEach(storage => {

        const option =
            document.createElement("option");

        option.value = storage;

        let displayStorage;

        if (storage >= 1024) {

            const tbValue =
                storage / 1024;

            displayStorage =
                `${tbValue} TB`;

        } else {

            displayStorage =
                `${storage} GB`;
        }

        option.textContent =
            displayStorage;

        storageSelect.appendChild(option);
    });

    // --------------------------------------------------------
    // CONFIGURATION FIELDS (Storage Type / Connectivity)
    //
    // Sourced from /model-configuration, keyed the same way as
    // deviceData. Each field only appears if this exact
    // Device+SubDevice+Model actually has options for it.
    // --------------------------------------------------------

    const modelConfig =
        (
            configData[selectedDevice] &&
            configData[selectedDevice][selectedSub] &&
            configData[selectedDevice][selectedSub][selectedModel]
        ) || { storageTypes: [], connectivity: [] };

    storageTypeSelect.innerHTML =
        '<option value="" disabled selected>Select storage type...</option>';

    modelConfig.storageTypes.forEach(value => {

        const option = document.createElement("option");

        option.value = value;
        option.textContent = value;

        storageTypeSelect.appendChild(option);
    });

    connectivitySelect.innerHTML =
        '<option value="" disabled selected>Select connectivity...</option>';

    modelConfig.connectivity.forEach(value => {

        const option = document.createElement("option");

        option.value = value;
        option.textContent = value;

        connectivitySelect.appendChild(option);
    });

    const hasStorage = validStorages.length > 0;
    const hasStorageType = modelConfig.storageTypes.length > 0;
    const hasConnectivity = modelConfig.connectivity.length > 0;

    storageGroup.classList.toggle("hidden", !hasStorage);
    storageTypeGroup.classList.toggle("hidden", !hasStorageType);
    connectivityGroup.classList.toggle("hidden", !hasConnectivity);

    storageSelect.disabled = !hasStorage;
    storageSelect.required = hasStorage;

    storageTypeSelect.required = hasStorageType;
    connectivitySelect.required = hasConnectivity;

    step4Applicable =
        hasStorage || hasStorageType || hasConnectivity;

    resultCard.classList.add("hidden");
    tradeInCard.classList.remove("result-shown");

    if (step4Applicable) {

        unlockStep(4);
        goToStep(4, "forward");

    } else {

        // No configuration fields apply (e.g. AirPods) -- go
        // straight to Condition.
        unlockStep(5);
        showConditionProfile(selectedDevice, selectedSub);
        goToStep(5, "forward");
    }

    updateProgress();
});


/* =========================================================
   CONFIGURATION FIELD CHANGE
   (Storage / Storage Type / Connectivity -- any of the three
   that are visible for the current device/model)
========================================================= */

function checkConfigurationComplete() {

    const selectedDevice =
        deviceSelect.value;

    const selectedSub =
        subDeviceSelect.value;

    const storageOk =
        storageGroup.classList.contains("hidden") ||
        storageSelect.value !== "";

    const storageTypeOk =
        storageTypeGroup.classList.contains("hidden") ||
        storageTypeSelect.value !== "";

    const connectivityOk =
        connectivityGroup.classList.contains("hidden") ||
        connectivitySelect.value !== "";

    if (!(storageOk && storageTypeOk && connectivityOk)) {
        return;
    }

    unlockStep(5);
    showConditionProfile(selectedDevice, selectedSub);

    resultCard.classList.add("hidden");
    tradeInCard.classList.remove("result-shown");

    updateProgress();

    goToStep(5, "forward");
}

storageSelect.addEventListener("change", checkConfigurationComplete);
storageTypeSelect.addEventListener("change", checkConfigurationComplete);
connectivitySelect.addEventListener("change", checkConfigurationComplete);


/* =========================================================
   CONDITION PROFILE
========================================================= */

function showConditionProfile(
    selectedDevice,
    selectedSub
) {

    conditionSection.classList.remove("hidden");

    hideConditionProfiles();

    if (selectedDevice === "iPhone") {

        showProfile("profileIPhone");

    } else if (selectedDevice === "iPad") {

        showProfile("profileIPad");

    } else if (selectedDevice === "Apple Watch") {

        showProfile("profileAppleWatch");

    } else if (selectedDevice === "AirPods") {

        showProfile("profileAirPods");

    } else if (selectedDevice === "Mac") {

        const subLower =
            selectedSub.toLowerCase();

        if (
            subLower.includes("mini") ||
            subLower.includes("studio") ||
            subLower.includes("pro desktop")
        ) {

            showProfile("profileMacDesktop");

        } else if (
            subLower.includes("imac")
        ) {

            showProfile("profileMacDesktop");

            const screenGroup =
                document.getElementById(
                    "desktopScreenGroup"
                );

            if (screenGroup) {
                screenGroup.classList.remove("hidden");
            }

        } else {

            showProfile("profileMacLaptop");
        }
    }

    updateProgress();
}


function showProfile(id) {

    const profile =
        document.getElementById(id);

    if (profile) {
        profile.classList.remove("hidden");
    }
}


function hideConditionProfiles() {

    const profiles = [
        "profileIPhone",
        "profileIPad",
        "profileAppleWatch",
        "profileMacLaptop",
        "profileMacDesktop",
        "profileAirPods"
    ];

    profiles.forEach(id => {

        const profile =
            document.getElementById(id);

        if (profile) {
            profile.classList.add("hidden");
        }
    });

    const screenGroup =
        document.getElementById(
            "desktopScreenGroup"
        );

    if (screenGroup) {
        screenGroup.classList.add("hidden");
    }
}


/* =========================================================
   PROGRESS
========================================================= */

function updateProgress() {

    // The old linear progress bar (#progressBar / #progressText)
    // has been replaced by the step indicator. This function is
    // kept so existing call sites don't need to change, and now
    // simply keeps the indicator's unlocked/current/completed
    // states in sync.

    refreshStepIndicator();
}


/* =========================================================
   API
========================================================= */

async function fetchPredictedPrice(
    device,
    subDevice,
    model,
    storage,
    storageType,
    connectivity
) {

    const apiUrl =
        "/predict";

    try {

        const payload = {
            Device: device,
            SubDevice: subDevice,
            Model: model,
            Storage: storage
                ? parseFloat(storage)
                : null,
            StorageType: storageType || null,
            Connectivity: connectivity || null
        };

        console.log(
            "Sending API request:",
            payload
        );

        const response =
            await fetch(
                apiUrl,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",

                        "Accept":
                            "application/json"
                    },

                    body:
                        JSON.stringify(payload)
                }
            );

        if (!response.ok) {

            const errorData =
                await response.text();

            console.error(
                "API error:",
                response.status,
                errorData
            );

            throw new Error(
                `API failed with status ${response.status}`
            );
        }

        const data =
            await response.json();

        console.log(
            "API response:",
            data
        );

        return data;

    } catch (error) {

        console.error(
            "Error fetching price:",
            error
        );

        return null;
    }
}

/* =========================================================
   CONDITION VALIDATION
========================================================= */

function clearConditionFieldError(input) {

    if (!input) {
        return;
    }

    input.classList.remove("customer-input-error");

    const error =
        document.getElementById(`${input.id}-error`);

    if (error) {

        error.textContent = "";
        error.classList.remove("visible");
    }
}


function showConditionFieldError(input, message) {
    if (!input) return;

    input.classList.add("customer-input-error");

    let error = document.getElementById(`${input.id}-error`);

    if (!error) {
        error = document.createElement("p");
        error.id = `${input.id}-error`;
        error.className = "customer-field-error";
        error.setAttribute("aria-live", "polite");

        const wrapper = input.closest(".select-wrapper");

        if (wrapper) {
            wrapper.insertAdjacentElement("afterend", error);
        } else {
            input.parentElement.appendChild(error);
        }
    }

    error.textContent = message;
    error.classList.add("visible");
}


function validateConditionSelections() {

    const selectedDevice =
        deviceSelect.value;

    const selectedSub =
        subDeviceSelect.value;

    let requiredFields = [];


    /* =========================
       iPhone
    ========================= */

    if (selectedDevice === "iPhone") {

        requiredFields = [
            ["iphoneScreen", "screen condition"],
            ["iphoneBody", "body & frame condition"],
            ["iphoneBattery", "battery health"]
        ];
    }


    /* =========================
       iPad
    ========================= */

    else if (selectedDevice === "iPad") {

        requiredFields = [
            ["ipadScreen", "screen condition"],
            ["ipadBody", "body condition"],
            ["ipadBattery", "battery health"]
        ];
    }


    /* =========================
       Apple Watch
    ========================= */

    else if (selectedDevice === "Apple Watch") {

        requiredFields = [
            ["watchScreen", "screen condition"],
            ["watchBody", "case condition"],
            ["watchBattery", "battery health"]
        ];
    }


    /* =========================
       AirPods
    ========================= */

    else if (selectedDevice === "AirPods") {

        requiredFields = [
            ["airpodsCase", "charging case condition"],
            ["airpodsBuds", "earbuds condition"],
            ["airpodsBattery", "battery condition"]
        ];
    }


    /* =========================
       Mac
    ========================= */

    else if (selectedDevice === "Mac") {

        const subLower =
            selectedSub.toLowerCase();


        // Mac mini / Mac Studio / Mac Pro
        // have no screen or battery condition.
        if (
            subLower.includes("mini") ||
            subLower.includes("studio") ||
            subLower.includes("pro desktop")
        ) {

            requiredFields = [
                ["desktopBody", "body condition"]
            ];
        }


        // iMac has a screen and body,
        // but no battery field.
        else if (subLower.includes("imac")) {

            requiredFields = [
                ["desktopScreen", "screen condition"],
                ["desktopBody", "body condition"]
            ];
        }


        // MacBook / laptop
        else {

            requiredFields = [
                ["laptopScreen", "screen condition"],
                ["laptopBody", "body condition"],
                ["laptopBattery", "battery health"]
            ];
        }
    }


    // Clear previous condition errors first.
    requiredFields.forEach(([id]) => {

        clearConditionFieldError(
            document.getElementById(id)
        );
    });


    // Find fields that have not been selected.
    const missingFields =
        requiredFields.filter(([id]) => {

            const field =
                document.getElementById(id);

            return !field || !field.value;
        });


    if (missingFields.length === 0) {
        return true;
    }


    // Show an error for every missing condition.
    missingFields.forEach(([id, label]) => {

        const field =
            document.getElementById(id);

        if (field) {

            showConditionFieldError(
                field,
                `Please select your ${label}.`
            );
        }
    });


    // Focus the first missing field.
    const firstMissingField =
        document.getElementById(
            missingFields[0][0]
        );

    if (firstMissingField) {

        firstMissingField.focus();

        firstMissingField.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }

    return false;
}


/* =========================================================
   SUBMIT
========================================================= */

tradeInForm.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();

        if (!validateConditionSelections()) {
            return;
        }

        const device =
            deviceSelect.value;

        const subDevice =
            subDeviceSelect.value;

        const model =
            modelSelect.value;

        const storage =
            storageSelect.value;

        const storageType =
            storageTypeSelect.value;

        const connectivity =
            connectivitySelect.value;

        resultCard.classList.remove("hidden");
        tradeInCard.classList.add("result-shown");

        resultValue.innerText =
            "Calculating...";

        submitBtn.disabled = true;

        const buttonText =
            submitBtn.querySelector(
                ".button-text"
            );

        const buttonArrow =
            submitBtn.querySelector(
                ".button-arrow"
            );

        if (buttonText) {
            buttonText.textContent =
                "Calculating...";
        }

        if (buttonArrow) {
            buttonArrow.textContent =
                "…";
        }

        resultCard.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

        const valuation =
            await fetchPredictedPrice(
                device,
                subDevice,
                model,
                storage,
                storageType,
                connectivity
            );

        submitBtn.disabled = false;

        if (buttonText) {
            buttonText.textContent =
                "Get my valuation";
        }

        if (buttonArrow) {
            buttonArrow.textContent =
                "→";
        }

        if (valuation === null) {

            resultValue.innerText =
                "Unable to calculate";

            return;
        }

        if (
            valuation.status !== "resolved" ||
            valuation.estimated_value === null
        ) {

            resultValue.innerHTML = `
                <span style="
                    font-size: 1rem;
                    letter-spacing: 0;
                    line-height: 1.5;
                ">
                    This exact device configuration
                    is currently unavailable.
                </span>
            `;

            return;
        }

        const medianPrice =
            Number(
                valuation.estimated_value
            );

        /* =====================================================
           CONDITION SCORE
        ===================================================== */

        let score = 100;

        const profileIPhone =
            document.getElementById(
                "profileIPhone"
            );

        const profileIPad =
            document.getElementById(
                "profileIPad"
            );

        const profileAppleWatch =
            document.getElementById(
                "profileAppleWatch"
            );

        const profileMacLaptop =
            document.getElementById(
                "profileMacLaptop"
            );

        const profileMacDesktop =
            document.getElementById(
                "profileMacDesktop"
            );

        const profileAirPods =
            document.getElementById(
                "profileAirPods"
            );


        /* =========================
           iPhone
        ========================= */

        if (
            profileIPhone &&
            !profileIPhone.classList.contains(
                "hidden"
            )
        ) {

            score += parseInt(
                document.getElementById(
                    "iphoneScreen"
                ).value
            );

            score += parseInt(
                document.getElementById(
                    "iphoneBody"
                ).value
            );

            score += parseInt(
                document.getElementById(
                    "iphoneBattery"
                ).value
            );

            document
                .querySelectorAll(
                    'input[name="iphoneDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(
                        cb.value
                    );
                });
        }


        /* =========================
           iPad
        ========================= */

        else if (
            profileIPad &&
            !profileIPad.classList.contains(
                "hidden"
            )
        ) {

            score += parseInt(
                document.getElementById(
                    "ipadScreen"
                ).value
            );

            score += parseInt(
                document.getElementById(
                    "ipadBody"
                ).value
            );

            score += parseInt(
                document.getElementById(
                    "ipadBattery"
                ).value
            );

            document
                .querySelectorAll(
                    'input[name="ipadDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(
                        cb.value
                    );
                });
        }


        /* =========================
           Apple Watch
        ========================= */

        else if (
            profileAppleWatch &&
            !profileAppleWatch.classList.contains(
                "hidden"
            )
        ) {

            score += parseInt(
                document.getElementById(
                    "watchScreen"
                ).value
            );

            score += parseInt(
                document.getElementById(
                    "watchBody"
                ).value
            );

            score += parseInt(
                document.getElementById(
                    "watchBattery"
                ).value
            );

            document
                .querySelectorAll(
                    'input[name="watchDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(
                        cb.value
                    );
                });
        }


        /* =========================
           Mac Laptop
        ========================= */

        else if (
            profileMacLaptop &&
            !profileMacLaptop.classList.contains(
                "hidden"
            )
        ) {

            score += parseInt(
                document.getElementById(
                    "laptopScreen"
                ).value
            );

            score += parseInt(
                document.getElementById(
                    "laptopBody"
                ).value
            );

            score += parseInt(
                document.getElementById(
                    "laptopBattery"
                ).value
            );

            document
                .querySelectorAll(
                    'input[name="laptopDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(
                        cb.value
                    );
                });
        }


        /* =========================
           Mac Desktop
        ========================= */

        else if (
            profileMacDesktop &&
            !profileMacDesktop.classList.contains(
                "hidden"
            )
        ) {

            const selectedSub =
                subDeviceSelect.value;

            if (
                selectedSub &&
                selectedSub
                    .toLowerCase()
                    .includes("imac")
            ) {

                const desktopScreen =
                    document.getElementById(
                        "desktopScreen"
                    );

                if (desktopScreen) {

                    score += parseInt(
                        desktopScreen.value
                    );
                }
            }

            score += parseInt(
                document.getElementById(
                    "desktopBody"
                ).value
            );

            document
                .querySelectorAll(
                    'input[name="desktopDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(
                        cb.value
                    );
                });
        }


        /* =========================
           AirPods
        ========================= */

        else if (
            profileAirPods &&
            !profileAirPods.classList.contains(
                "hidden"
            )
        ) {

            score += parseInt(
                document.getElementById(
                    "airpodsCase"
                ).value
            );

            score += parseInt(
                document.getElementById(
                    "airpodsBuds"
                ).value
            );

            score += parseInt(
                document.getElementById(
                    "airpodsBattery"
                ).value
            );

            document
                .querySelectorAll(
                    'input[name="airpodsDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(
                        cb.value
                    );
                });
        }


        /* =====================================================
           CLAMP
        ===================================================== */

        score = Math.max(
            0,
            Math.min(
                100,
                score
            )
        );


        /* =====================================================
           GRADE
        ===================================================== */

        let grade = "F";
        let multiplier = 0.10;

        if (score >= 90) {

            grade = "A";
            multiplier = 1.00;

        } else if (score >= 71) {

            grade = "B";
            multiplier = 0.80;

        } else if (score >= 51) {

            grade = "C";
            multiplier = 0.40;
        }


        /* =====================================================
           FINAL VALUE
        ===================================================== */

        const finalPrice =
            Math.floor(
                medianPrice * multiplier
            );

        const formattedPrice =
            finalPrice.toLocaleString(
                "en-MY"
            );

        /* =====================================================
        TRADE-IN RECORD
        ===================================================== */

        const tradeInRecord = {
            customer: customerDetails,

            device: {
                device: device,
                subDevice: subDevice,
                model: model,
                storage: storage || null,
                storageType: storageType || null,
                connectivity: connectivity || null
            },

            valuation: {
                marketValue: medianPrice,
                conditionScore: score,
                grade: grade,
                multiplier: multiplier,
                finalValue: finalPrice
            },

            createdAt: new Date().toISOString()
        };

        sessionStorage.setItem(
            "tradeInRecord",
            JSON.stringify(tradeInRecord)
        );

        console.log(
            "Trade-in record:",
            tradeInRecord
        );

        /* =====================================================
        SAVE TRADE-IN RECORD
        ===================================================== */

        try {
            const saveResponse = await fetch("/customer/trade-in", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(tradeInRecord)
            });

            const saveResult = await saveResponse.json();

            if (!saveResponse.ok || saveResult.status !== "success") {
                console.error(
                    "Failed to save trade-in record:",
                    saveResult
                );
            } else {
                console.log(
                    "Trade-in record saved successfully:",
                    saveResult
                );
            }

        } catch (error) {
            console.error(
                "Customer trade-in save error:",
                error
            );
        }


        /* =====================================================
           DISPLAY
        ===================================================== */

        resultValue.innerHTML =
            `RM ${formattedPrice}`;

        resultCard.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }
);


/* =========================================================
   START OVER
========================================================= */

startOverBtn.addEventListener(
    "click",
    function () {

        tradeInForm.reset();

        subDeviceSelect.innerHTML =
            '<option value="" disabled selected>Select a product line...</option>';

        subDeviceBoxGrid.innerHTML = "";
        subDeviceBoxGrid.classList.add("disabled");

        deviceBoxGrid
            .querySelectorAll(".device-box.selected")
            .forEach(box => {
                box.classList.remove("selected");
                box.setAttribute("aria-selected", "false");
            });

        modelSelect.innerHTML =
            '<option value="" disabled selected>Select a model...</option>';

        storageTypeSelect.innerHTML =
            '<option value="" disabled selected>Select storage type...</option>';

        connectivitySelect.innerHTML =
            '<option value="" disabled selected>Select connectivity...</option>';

        storageTypeGroup.classList.add("hidden");
        connectivityGroup.classList.add("hidden");
        storageGroup.classList.remove("hidden");

        storageSelect.innerHTML =
            '<option value="" disabled selected>Select storage...</option>';

        subDeviceSelect.disabled = true;
        modelSelect.disabled = true;
        storageSelect.disabled = true;

        storageSelect.required = true;

        hideConditionProfiles();

        resultCard.classList.add(
            "hidden"
        );

        tradeInCard.classList.remove("result-shown");

        updateDeviceImagePreview(null, null);

        step4Applicable = true;

        maxUnlockedStep = 1;

        goToStep(1, "back");

        updateProgress();

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    }
);


/* =========================================================
   UTILITY
========================================================= */

function scrollToElement(element) {

    if (!element) {
        return;
    }

    setTimeout(() => {

        element.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

    }, 120);
}