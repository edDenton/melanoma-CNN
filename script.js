const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("imagePreview");
const predictBtn = document.getElementById("predictBtn");
const prediction = document.getElementById("prediction");
const API_URL = "https://melanoma-cnn-backend.onrender.com/predict";

let selectedImage = null;

async function resizeImage(file) {
    const img = new Image();
    img.src = URL.createObjectURL(file);
    await img.decode();

    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 128;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, 128, 128);

    return new Promise(resolve => {
        canvas.toBlob(
            blob => resolve(blob),
            "image/jpeg",
            0.9
        );
    });
}

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

    const resized = await resizeImage(selectedImage);

    const formData = new FormData();
    formData.append("image", resized, "image.jpg");

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
