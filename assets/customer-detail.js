document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("customerDetailForm");

    if (!form) {
        console.error("❌ customerDetailForm was not found.");
        return;
    }

    console.log("✅ Customer detail page loaded.");
    console.log("✅ Customer detail form found.");

    form.addEventListener("submit", (event) => {
        event.preventDefault();

        console.log("🟡 Start my trade-in button clicked.");

        const customerNameElement =
            document.getElementById("customerName");

        const customerPhoneElement =
            document.getElementById("customerPhone");

        const customerEmailElement =
            document.getElementById("customerEmail");

        const preferredContactElement =
            document.getElementById("customerContactMethod");

        if (
            !customerNameElement ||
            !customerPhoneElement ||
            !customerEmailElement ||
            !preferredContactElement
        ) {
            console.error(
                "❌ Customer detail form element is missing.",
                {
                    customerName: customerNameElement,
                    customerPhone: customerPhoneElement,
                    customerEmail: customerEmailElement,
                    preferredContact: preferredContactElement
                }
            );

            return;
        }

        const customerName =
            customerNameElement.value.trim();

        const customerPhone =
            customerPhoneElement.value.trim();

        const customerEmail =
            customerEmailElement.value.trim();

        const preferredContact =
            preferredContactElement.value;

        console.log("Customer details collected:", {
            name: customerName,
            phone: customerPhone,
            email: customerEmail,
            preferredContact: preferredContact
        });

        if (
            !customerName ||
            !customerPhone ||
            !customerEmail ||
            !preferredContact
        ) {
            console.warn(
                "⚠️ Customer details are incomplete."
            );

            return;
        }

        const customerDetails = {
            name: customerName,
            phone: customerPhone,
            email: customerEmail,
            preferredContact: preferredContact
        };

        sessionStorage.setItem(
            "customerDetails",
            JSON.stringify(customerDetails)
        );

        console.log(
            "✅ customerDetails saved to sessionStorage:",
            customerDetails
        );

        console.log(
            "➡️ Redirecting to /index.html..."
        );

        window.location.href = "/index.html";
    });
});
