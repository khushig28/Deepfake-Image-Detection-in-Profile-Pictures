// Deepfake Detector - Frontend JavaScript

const uploadCard = document.getElementById('uploadCard');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const resultSection = document.getElementById('resultSection');
const loadingOverlay = document.getElementById('loadingOverlay');
const uploadSection = document.querySelector('.upload-section');
const analyzeAnotherBtn = document.getElementById('analyzeAnother');

// Handle file browse
browseBtn.addEventListener('click', () => {
    fileInput.click();
});

// Handle file input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop
uploadCard.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadCard.classList.add('drag-over');
});

uploadCard.addEventListener('dragleave', () => {
    uploadCard.classList.remove('drag-over');
});

uploadCard.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadCard.classList.remove('drag-over');

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Handle file upload and prediction
async function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (PNG, JPG, BMP, WebP)');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }

    // Show loading
    loadingOverlay.classList.add('show');

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImage').src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Send to API
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        displayResult(result);

    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing image. Please try again.');
        loadingOverlay.classList.remove('show');
    }
}

// Display prediction result
function displayResult(result) {
    // Hide loading and upload, show result
    loadingOverlay.classList.remove('show');
    uploadSection.style.display = 'none';
    resultSection.classList.add('show');

    // Set result badge
    const resultBadge = document.getElementById('resultBadge');
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');

    if (result.is_fake) {
        resultBadge.classList.remove('real');
        resultBadge.classList.add('fake');
        resultIcon.textContent = '⚠️';
        resultText.textContent = 'AI-Generated (Deepfake)';
    } else {
        resultBadge.classList.remove('fake');
        resultBadge.classList.add('real');
        resultIcon.textContent = '✅';
        resultText.textContent = 'Authentic (Real)';
    }

    // Set confidence
    const confidence = (result.confidence * 100).toFixed(1);
    document.getElementById('confidenceFill').style.width = `${confidence}%`;
    document.getElementById('confidenceValue').textContent = `${confidence}%`;

    // Set probabilities
    const fakeProb = (result.probabilities.Fake * 100).toFixed(1);
    const realProb = (result.probabilities.Real * 100).toFixed(1);

    document.getElementById('fakeProb').style.width = `${fakeProb}%`;
    document.getElementById('fakePercent').textContent = `${fakeProb}%`;

    document.getElementById('realProb').style.width = `${realProb}%`;
    document.getElementById('realPercent').textContent = `${realProb}%`;

    // Display heatmap if available
    if (result.heatmap) {
        const heatmapContainer = document.getElementById('heatmapContainer');
        const heatmapImage = document.getElementById('heatmapImage');
        heatmapImage.src = result.heatmap;
        heatmapContainer.style.display = 'block';
    }

    // Set warning
    const warningSection = document.getElementById('warningSection');
    const warningContent = document.getElementById('warningContent');

    warningSection.className = 'warning-section ' + result.warning_level;

    if (result.warning_level === 'high') {
        warningContent.innerHTML = '<strong>⚠️ High Risk:</strong> This image is very likely AI-generated. Confidence is high that this is a deepfake.';
    } else if (result.warning_level === 'medium') {
        warningContent.innerHTML = '<strong>⚠️ Medium Risk:</strong> This image appears to be AI-generated, but with moderate confidence. Further verification recommended.';
    } else if (result.warning_level === 'low') {
        warningContent.innerHTML = '<strong>ℹ️ Low Risk:</strong> Weak indication of AI generation. Image may be authentic or the model is uncertain.';
    } else {
        warningContent.innerHTML = '<strong>✅ Authentic:</strong> This image appears to be a genuine photograph with high confidence.';
    }
}

// Analyze another image
analyzeAnotherBtn.addEventListener('click', () => {
    resultSection.classList.remove('show');
    uploadSection.style.display = 'block';
    fileInput.value = '';
});
