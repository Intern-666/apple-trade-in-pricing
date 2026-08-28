let deviceData = {}; // This will hold the data from the backend

// 1. Fetch data on load
window.addEventListener("DOMContentLoaded", async function() {
    try {
        const response = await fetch("http://127.0.0.1:8000/available-models");
        deviceData = await response.json();
        
        const deviceSelect = document.getElementById("deviceSelect");
        deviceSelect.innerHTML = '<option value="" disabled selected>Select a device...</option>';
        
        for (const device in deviceData) {
            const option = document.createElement("option");
            option.value = device;
            option.textContent = device;
            deviceSelect.appendChild(option);
        }
    } catch (error) {
        console.error("Failed to load dynamic models:", error);
    }
});

// 2. When Device changes -> Unlock & populate Product Line
// 2. When Device changes -> Unlock & populate Product Line
document.getElementById("deviceSelect").addEventListener("change", function(event) {
    const selectedDevice = event.target.value;
    const subDevices = deviceData[selectedDevice] || {};
    
    const subDeviceSelect = document.getElementById("subDeviceSelect");
    const modelSelect = document.getElementById("modelSelect");
    const storageSelect = document.getElementById("storageSelect");
    
    // Hide condition section until a product line is chosen
    document.getElementById("conditionSection").classList.add("hidden");

    // Populate Product Line
    subDeviceSelect.innerHTML = '<option value="" disabled selected>Select a product line...</option>';
    for (const sub in subDevices) {
        const option = document.createElement("option");
        option.value = sub;
        option.textContent = sub;
        subDeviceSelect.appendChild(option);
    }
    
    // Unlock Product Line, reset Model & Storage
    subDeviceSelect.disabled = false;
    modelSelect.innerHTML = '<option value="" disabled selected>Select a model...</option>';
    modelSelect.disabled = true;
    storageSelect.innerHTML = '<option value="" disabled selected>Select storage...</option>';
    storageSelect.disabled = true;

    // AirPods do not have storage
    if (selectedDevice === "AirPods") {

        storageSelect.required = false;
        storageSelect.disabled = true;
        storageSelect.value = "";

    } else {

        storageSelect.required = true;
    }
});

// 3. When Product Line changes -> Unlock Model & Show Correct Profile
document.getElementById("subDeviceSelect").addEventListener("change", function(event) {
    const selectedDevice = document.getElementById("deviceSelect").value;
    const selectedSub = event.target.value;
    
    const modelsObj = deviceData[selectedDevice][selectedSub] || {}; 
    
    const modelSelect = document.getElementById("modelSelect");
    const storageSelect = document.getElementById("storageSelect");

    // Populate Model using Object.keys() so we can sort them safely
    modelSelect.innerHTML = '<option value="" disabled selected>Select a model...</option>';
    Object.keys(modelsObj).sort().forEach(model => {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        modelSelect.appendChild(option);
    });
    
    modelSelect.disabled = false;
    storageSelect.innerHTML = '<option value="" disabled selected>Select storage...</option>';
    storageSelect.disabled = true;

    // --- CONDITION PROFILES SWITCHING ---
    const conditionSection = document.getElementById("conditionSection");
    conditionSection.classList.remove("hidden");

    // Grab all profile divs
    const profileIPhone = document.getElementById("profileIPhone");
    const profileIPad = document.getElementById("profileIPad");
    const profileAppleWatch = document.getElementById("profileAppleWatch");
    const profileMacLaptop = document.getElementById("profileMacLaptop");
    const profileMacDesktop = document.getElementById("profileMacDesktop");
    const profileAirPods = document.getElementById("profileAirPods");

    // Hide all profiles first
    [profileIPhone, profileIPad, profileAppleWatch, profileMacLaptop, profileMacDesktop, profileAirPods].forEach(profile => {
        if (profile) profile.classList.add("hidden");
    });

    // Reveal only the correct profile based on Device & Product Line
    if (selectedDevice === "iPhone") {
        if (profileIPhone) profileIPhone.classList.remove("hidden");
    } else if (selectedDevice === "iPad") {
        if (profileIPad) profileIPad.classList.remove("hidden");
    } else if (selectedDevice === "Apple Watch") {
        if (profileAppleWatch) profileAppleWatch.classList.remove("hidden");
    } else if (selectedDevice === "AirPods") {
        if (profileAirPods) profileAirPods.classList.remove("hidden");
    } else if (selectedDevice === "Mac") {

        const subLower = selectedSub.toLowerCase();

        if (
            subLower.includes("mini") ||
            subLower.includes("studio") ||
            subLower.includes("pro desktop")
        ) {

            // Mac mini / Mac Studio / Mac Pro
            // No built-in display
            if (profileMacDesktop) {
                profileMacDesktop.classList.remove("hidden");
            }

        } else if (
            subLower.includes("imac")
        ) {

            // iMac
            // Has built-in display
            if (profileMacDesktop) {
                profileMacDesktop.classList.remove("hidden");
            }

            const screenGroup =
                document.getElementById("desktopScreenGroup");

            if (screenGroup) {
                screenGroup.classList.remove("hidden");
            }

        } else {

            // MacBook / MacBook Air / MacBook Pro
            if (profileMacLaptop) {
                profileMacLaptop.classList.remove("hidden");
            }
        }
    }
});

// 4. When Model changes -> Unlock & populate exact Storage capacities
document.getElementById("modelSelect").addEventListener("change", function(event) {
    const selectedDevice = document.getElementById("deviceSelect").value;
    const selectedSub = document.getElementById("subDeviceSelect").value;
    const selectedModel = event.target.value;
    
    const validStorages = deviceData[selectedDevice][selectedSub][selectedModel] || [];
    const storageSelect = document.getElementById("storageSelect");
    
    storageSelect.innerHTML = '<option value="" disabled selected>Select storage...</option>';
    
    validStorages.forEach(storage => {
        const option = document.createElement("option");
        option.value = storage; // Keeps the raw number (e.g., 1024) for the backend API
        
        // --- NEW: Convert to TB if 1024 or greater ---
        let displayStorage = "";
        if (storage >= 1024) {
            const tbValue = storage / 1024;
            displayStorage = `${tbValue} TB`;
        } else {
            displayStorage = `${storage} GB`;
        }
        
        option.textContent = displayStorage;
        storageSelect.appendChild(option);
    });
    
    if (selectedDevice === "AirPods") {

        storageSelect.disabled = true;
        storageSelect.required = false;
        storageSelect.value = "";

    } else {

        storageSelect.disabled = false;
        storageSelect.required = true;
    }
});

// 5. The function that talks to your FastAPI server
// ============================================================
// 5. TALK TO FASTAPI SERVER
// ============================================================

async function fetchPredictedPrice(
    device,
    subDevice,
    model,
    storage
) {

    const apiUrl = "http://127.0.0.1:8000/predict";

    try {

        const payload = {
            Device: device,
            SubDevice: subDevice,
            Model: model,
            Storage: storage ? parseFloat(storage) : null
        };

        console.log("Sending API request:", payload);

        const response = await fetch(
            apiUrl,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },

                body: JSON.stringify(payload)
            }
        );


        // ----------------------------------------------------
        // API ERROR
        // ----------------------------------------------------

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


        // ----------------------------------------------------
        // SUCCESS
        // ----------------------------------------------------

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

// 6. The Event Listener for the Submit Button
document.getElementById("tradeInForm").addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        console.log("SUBMIT HANDLER FIRED");

        // ========================================================
        // 1. GET DEVICE SELECTIONS
        // ========================================================

        const device =
            document.getElementById("deviceSelect").value;

        const subDevice =
            document.getElementById("subDeviceSelect").value;

        const model =
            document.getElementById("modelSelect").value;

        const storage =
            document.getElementById("storageSelect").value;


        // ========================================================
        // 2. RESULT ELEMENTS
        // ========================================================

        const resultCard =
            document.getElementById("resultCard");

        const resultValue =
            document.getElementById("resultValue");


        resultCard.classList.remove("hidden");

        resultValue.innerText = "Calculating...";


        // ========================================================
        // 3. GET EXACT MARKET MEDIAN
        // ========================================================

        const valuation =
            await fetchPredictedPrice(
                device,
                subDevice,
                model,
                storage
            );


        if (valuation === null) {

            resultValue.innerText =
                "Unable to calculate trade-in value.";

            return;
        }


        if (
            valuation.status !== "resolved" ||
            valuation.estimated_value === null
        ) {

            resultValue.innerHTML = `
                <span style="font-size: 0.8em;">
                    This exact device configuration
                    is currently unavailable.
                </span>
            `;

            return;
        }


        const medianPrice =
            Number(valuation.estimated_value);


        // ========================================================
        // 4. CONDITION ASSESSMENT
        // ========================================================

        let score = 100;


        const profileIPhone =
            document.getElementById("profileIPhone");

        const profileIPad =
            document.getElementById("profileIPad");

        const profileAppleWatch =
            document.getElementById("profileAppleWatch");

        const profileMacLaptop =
            document.getElementById("profileMacLaptop");

        const profileMacDesktop =
            document.getElementById("profileMacDesktop");

        const profileAirPods =
            document.getElementById("profileAirPods");


        // --------------------------------------------------------
        // iPhone
        // --------------------------------------------------------

        if (
            profileIPhone &&
            !profileIPhone.classList.contains("hidden")
        ) {

            score += parseInt(
                document.getElementById("iphoneScreen").value
            );

            score += parseInt(
                document.getElementById("iphoneBody").value
            );

            score += parseInt(
                document.getElementById("iphoneBattery").value
            );

            document
                .querySelectorAll(
                    'input[name="iphoneDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(cb.value);
                });
        }


        // --------------------------------------------------------
        // iPad
        // --------------------------------------------------------

        else if (
            profileIPad &&
            !profileIPad.classList.contains("hidden")
        ) {

            score += parseInt(
                document.getElementById("ipadScreen").value
            );

            score += parseInt(
                document.getElementById("ipadBody").value
            );

            score += parseInt(
                document.getElementById("ipadBattery").value
            );

            document
                .querySelectorAll(
                    'input[name="ipadDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(cb.value);
                });
        }


        // --------------------------------------------------------
        // Apple Watch
        // --------------------------------------------------------

        else if (
            profileAppleWatch &&
            !profileAppleWatch.classList.contains("hidden")
        ) {

            score += parseInt(
                document.getElementById("watchScreen").value
            );

            score += parseInt(
                document.getElementById("watchBody").value
            );

            score += parseInt(
                document.getElementById("watchBattery").value
            );

            document
                .querySelectorAll(
                    'input[name="watchDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(cb.value);
                });
        }


        // --------------------------------------------------------
        // Mac Laptop
        // --------------------------------------------------------

        else if (
            profileMacLaptop &&
            !profileMacLaptop.classList.contains("hidden")
        ) {

            score += parseInt(
                document.getElementById("laptopScreen").value
            );

            score += parseInt(
                document.getElementById("laptopBody").value
            );

            score += parseInt(
                document.getElementById("laptopBattery").value
            );

            document
                .querySelectorAll(
                    'input[name="laptopDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(cb.value);
                });
        }


        // --------------------------------------------------------
        // Mac Desktop
        // --------------------------------------------------------

        else if (
            profileMacDesktop &&
            !profileMacDesktop.classList.contains("hidden")
        ) {

            // --------------------------------------------------------
            // iMac Screen
            // --------------------------------------------------------

            const selectedSub =
                document.getElementById("subDeviceSelect").value;

            if (
                selectedSub &&
                selectedSub.toLowerCase().includes("imac")
            ) {

                const desktopScreen =
                    document.getElementById("desktopScreen");

                if (desktopScreen) {
                    score += parseInt(
                        desktopScreen.value
                    );
                }
            }


            // --------------------------------------------------------
            // Body / Chassis
            // --------------------------------------------------------

            score += parseInt(
                document.getElementById("desktopBody").value
            );


            // --------------------------------------------------------
            // Functionality Issues
            // --------------------------------------------------------

            document
                .querySelectorAll(
                    'input[name="desktopDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(cb.value);
                });
        }


        // --------------------------------------------------------
        // AirPods
        // --------------------------------------------------------

        else if (
            profileAirPods &&
            !profileAirPods.classList.contains("hidden")
        ) {

            score += parseInt(
                document.getElementById("airpodsCase").value
            );

            score += parseInt(
                document.getElementById("airpodsBuds").value
            );

            score += parseInt(
                document.getElementById("airpodsBattery").value
            );

            document
                .querySelectorAll(
                    'input[name="airpodsDefect"]:checked'
                )
                .forEach(cb => {
                    score += parseInt(cb.value);
                });
        }


        // ========================================================
        // 5. CLAMP SCORE
        // ========================================================

        score = Math.max(
            0,
            Math.min(100, score)
        );


        // ========================================================
        // 6. GRADE + MULTIPLIER
        // ========================================================

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


        // ========================================================
        // 7. FINAL TRADE-IN VALUE
        // ========================================================

        // ========================================================
        // 7. FINAL TRADE-IN VALUE
        // ========================================================

        const finalPrice = Math.floor(
            medianPrice * multiplier
        );

        const formattedPrice =
            finalPrice.toLocaleString("en-MY");

        // ========================================================
        // 8. DISPLAY RESULT
        // ========================================================

        resultValue.innerHTML = `
            RM ${formattedPrice}
        `;
        }
);