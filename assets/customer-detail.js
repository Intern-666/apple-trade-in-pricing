document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("customerDetailForm");

    if (!form) {
        console.error("customerDetailForm was not found.");
        return;
    }

    const nameInput = document.getElementById("customerName");
    const phoneInput = document.getElementById("customerPhone");
    const emailInput = document.getElementById("customerEmail");
    const contactInput = document.getElementById("customerContactMethod");

    const submitButton =
        form.querySelector('button[type="submit"]');

    const submitText =
        submitButton?.querySelector("span");

    if (
        !nameInput ||
        !phoneInput ||
        !emailInput ||
        !contactInput ||
        !submitButton ||
        !submitText
    ) {
        console.error(
            "One or more customer form elements are missing."
        );
        return;
    }

    function clearFieldError(input) {
        input.classList.remove("customer-input-error");

        const error =
            document.getElementById(`${input.id}-error`);

        if (error) {
            error.textContent = "";
            error.classList.remove("visible");
        }
    }

    function showFieldError(input, message) {
        input.classList.add("customer-input-error");

        const error =
            document.getElementById(`${input.id}-error`);

        if (error) {
            error.textContent = message;
            error.classList.add("visible");
        }
    }

    function validateName() {
        const value = nameInput.value.trim();

        clearFieldError(nameInput);

        if (!value) {
            showFieldError(
                nameInput,
                "Please enter your full name."
            );
            return false;
        }

        if (value.length < 2) {
            showFieldError(
                nameInput,
                "Please enter a valid name."
            );
            return false;
        }

        return true;
    }

    function validatePhone() {
        const value = phoneInput.value.trim();

        clearFieldError(phoneInput);

        if (!value) {
            showFieldError(
                phoneInput,
                "Please enter your phone number."
            );
            return false;
        }

        const phoneDigits =
            value.replace(/\D/g, "");

        if (
            phoneDigits.length < 9 ||
            phoneDigits.length > 12
        ) {
            showFieldError(
                phoneInput,
                "Please enter a valid phone number."
            );
            return false;
        }

        return true;
    }

    function validateEmail() {
        const value = emailInput.value.trim();

        clearFieldError(emailInput);

        if (!value) {
            showFieldError(
                emailInput,
                "Please enter your email address."
            );
            return false;
        }

        const emailPattern =
            /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

        if (!emailPattern.test(value)) {
            showFieldError(
                emailInput,
                "Please enter a valid email address."
            );
            return false;
        }

        return true;
    }

    function validateContactMethod() {
        clearFieldError(contactInput);

        if (!contactInput.value) {
            showFieldError(
                contactInput,
                "Please select a preferred contact method."
            );
            return false;
        }

        return true;
    }

    nameInput.addEventListener("input", validateName);
    phoneInput.addEventListener("input", validatePhone);
    emailInput.addEventListener("input", validateEmail);
    contactInput.addEventListener(
        "change",
        validateContactMethod
    );

    form.addEventListener("submit", (event) => {
        event.preventDefault();

        const isNameValid = validateName();
        const isPhoneValid = validatePhone();
        const isEmailValid = validateEmail();
        const isContactValid = validateContactMethod();

        if (
            !isNameValid ||
            !isPhoneValid ||
            !isEmailValid ||
            !isContactValid
        ) {
            const firstInvalidField =
                form.querySelector(
                    ".customer-input-error"
                );

            firstInvalidField?.focus();

            return;
        }

        const customerDetails = {
            name: nameInput.value.trim(),
            phone: phoneInput.value.trim(),
            email: emailInput.value.trim(),
            preferredContact: contactInput.value
        };

        sessionStorage.setItem(
            "customerDetails",
            JSON.stringify(customerDetails)
        );

        submitButton.disabled = true;
        submitText.textContent = "Starting...";

        window.location.href = "/";
    });

    window.addEventListener("pageshow", () => {
        submitButton.disabled = false;
        submitText.textContent = "Start my trade-in";
    });
});