const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("imagePreview");
const predictBtn = document.getElementById("predictBtn");
const prediction = document.getElementById("prediction");

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

    // TEMP: fake model response
    const result = mockModelPrediction();

    prediction.innerHTML = `
        <strong>${result.label}</strong><br>
        Confidence: ${(result.probability * 100).toFixed(1)}%
    `;
});

function mockModelPrediction() {
    const probability = Math.random();
    return {
        label: probability > 0.5 ? "Benign" : "Malignant",
        probability: probability
    };
}
