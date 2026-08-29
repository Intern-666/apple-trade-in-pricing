let deviceData = {};

/* =========================================================
   ELEMENTS
========================================================= */

const deviceSelect = document.getElementById("deviceSelect");
const subDeviceSelect = document.getElementById("subDeviceSelect");
const modelSelect = document.getElementById("modelSelect");
const storageSelect = document.getElementById("storageSelect");

const tradeInForm = document.getElementById("tradeInForm");
const conditionSection = document.getElementById("conditionSection");

const resultCard = document.getElementById("resultCard");
const resultValue = document.getElementById("resultValue");
const submitBtn = document.getElementById("submitBtn");
const startOverBtn = document.getElementById("startOverBtn");

const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");

const productLineSection = document.getElementById("productLineSection");
const modelSection = document.getElementById("modelSection");
const storageSection = document.getElementById("storageSection");


/* =========================================================
   INITIAL LOAD
========================================================= */

window.addEventListener("DOMContentLoaded", async function () {

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/available-models"
        );

        if (!response.ok) {
            throw new Error(
                `Failed to load models: ${response.status}`
            );
        }

        deviceData = await response.json();

        deviceSelect.innerHTML =
            '<option value="" disabled selected>Select a device...</option>';

        for (const device in deviceData) {

            const option = document.createElement("option");

            option.value = device;
            option.textContent = device;

            deviceSelect.appendChild(option);
        }

        updateProgress();

    } catch (error) {

        console.error(
            "Failed to load dynamic models:",
            error
        );

        deviceSelect.innerHTML =
            '<option value="" disabled selected>Unable to load devices</option>';
    }
});


/* =========================================================
   DEVICE CHANGE
========================================================= */

deviceSelect.addEventListener("change", function (event) {

    const selectedDevice = event.target.value;

    const subDevices =
        deviceData[selectedDevice] || {};

    subDeviceSelect.innerHTML =
        '<option value="" disabled selected>Select a product line...</option>';

    for (const sub in subDevices) {

        const option = document.createElement("option");

        option.value = sub;
        option.textContent = sub;

        subDeviceSelect.appendChild(option);
    }

    subDeviceSelect.disabled = false;

    modelSelect.innerHTML =
        '<option value="" disabled selected>Select a model...</option>';

    modelSelect.disabled = true;

    storageSelect.innerHTML =
        '<option value="" disabled selected>Select storage...</option>';

    storageSelect.disabled = true;

    storageSelect.required =
        selectedDevice !== "AirPods";

    hideConditionProfiles();

    conditionSection.classList.add("hidden");

    resultCard.classList.add("hidden");

    updateProgress();

    scrollToElement(productLineSection);
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

    storageSelect.innerHTML =
        '<option value="" disabled selected>Select storage...</option>';

    storageSelect.disabled = true;

    hideConditionProfiles();

    conditionSection.classList.add("hidden");

    resultCard.classList.add("hidden");

    updateProgress();

    scrollToElement(modelSection);
});


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

    if (selectedDevice === "AirPods") {

        storageSelect.disabled = true;
        storageSelect.required = false;
        storageSelect.value = "";

        showConditionProfile(
            selectedDevice,
            selectedSub
        );

    } else {

        storageSelect.disabled = false;
        storageSelect.required = true;

        hideConditionProfiles();
        conditionSection.classList.add("hidden");
    }

    resultCard.classList.add("hidden");

    updateProgress();

    if (selectedDevice !== "AirPods") {
        scrollToElement(storageSection);
    }
});


/* =========================================================
   STORAGE CHANGE
========================================================= */

storageSelect.addEventListener("change", function () {

    const selectedDevice =
        deviceSelect.value;

    const selectedSub =
        subDeviceSelect.value;

    showConditionProfile(
        selectedDevice,
        selectedSub
    );

    resultCard.classList.add("hidden");

    updateProgress();

    scrollToElement(conditionSection);
});


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

    const device =
        deviceSelect.value;

    const subDevice =
        subDeviceSelect.value;

    const model =
        modelSelect.value;

    const storage =
        storageSelect.value;

    let step = 1;

    if (device) {
        step = 2;
    }

    if (subDevice) {
        step = 3;
    }

    if (model) {
        step = 4;
    }

    if (
        storage ||
        device === "AirPods"
    ) {
        step = 5;
    }

    const percentage =
        (step / 5) * 100;

    progressBar.style.width =
        `${percentage}%`;

    progressText.textContent =
        `Step ${step} of 5`;
}


/* =========================================================
   API
========================================================= */

async function fetchPredictedPrice(
    device,
    subDevice,
    model,
    storage
) {

    const apiUrl =
        "http://127.0.0.1:8000/predict";

    try {

        const payload = {
            Device: device,
            SubDevice: subDevice,
            Model: model,
            Storage: storage
                ? parseFloat(storage)
                : null
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
   SUBMIT
========================================================= */

tradeInForm.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();

        const device =
            deviceSelect.value;

        const subDevice =
            subDeviceSelect.value;

        const model =
            modelSelect.value;

        const storage =
            storageSelect.value;

        resultCard.classList.remove("hidden");

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
                storage
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

        modelSelect.innerHTML =
            '<option value="" disabled selected>Select a model...</option>';

        storageSelect.innerHTML =
            '<option value="" disabled selected>Select storage...</option>';

        subDeviceSelect.disabled = true;
        modelSelect.disabled = true;
        storageSelect.disabled = true;

        storageSelect.required = true;

        conditionSection.classList.add(
            "hidden"
        );

        hideConditionProfiles();

        resultCard.classList.add(
            "hidden"
        );

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