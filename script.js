const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("imagePreview");
const predictBtn = document.getElementById("predictBtn");
const prediction = document.getElementById("prediction");
const API_URL = "https://melanoma-cnn-backend.onrender.com/predict";

let selectedImage = null;

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;

    selectedImage = file;

    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);

    preview.innerHTML = "";
    preview.appendChild(img);

    prediction.innerHTML = "";
    predictBtn.disabled = false;
});

predictBtn.addEventListener("click", async () => {
    prediction.innerHTML = "Analyzing image...";

    const formData = new FormData();
    formData.append("image", selectedImage);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Prediction failed");
        }

        const result = await response.json();

        prediction.innerHTML = `
            <strong>${result.prediction}</strong><br>
            Confidence: ${(result.confidence * 100).toFixed(1)}%
        `;

    } catch (err) {
        prediction.innerHTML = "Error getting prediction. Try again later."
        console.error(err);
    }
});
