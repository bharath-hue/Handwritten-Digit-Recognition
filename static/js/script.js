
document.addEventListener('DOMContentLoaded', function() {
    // Canvas setup
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'black';

    // Handle Mouse
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });
    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });
    canvas.addEventListener('mouseup', () => isDrawing = false);
    canvas.addEventListener('mouseleave', () => isDrawing = false);

    // Handle Touch
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        lastX = e.touches[0].clientX - rect.left;
        lastY = e.touches[0].clientY - rect.top;
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        const x = e.touches[0].clientX - rect.left;
        const y = e.touches[0].clientY - rect.top;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();
        lastX = x;
        lastY = y;
    });
    canvas.addEventListener('touchend', () => isDrawing = false);

    // Clear button
    const clearButton = document.getElementById('clear-button');
    if (clearButton) {
        clearButton.addEventListener('click', function() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            resetResults();
        });
    }

    // Recognize (Predict) button
    const recognizeButton = document.getElementById('recognize-button');
    if (recognizeButton) {
        recognizeButton.addEventListener('click', function() {
            // Step 1: Get image data from canvas, downscale to 28x28
            const dataUrl28 = downscaleTo28x28(canvas);

            // Step 2: Build payload
            const payload = {
                image: dataUrl28.split(',')[1]  // remove 'data:image/png;base64,'
            };

            // Step 3: POST to /predict
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            })
            .then(res => res.json())
            .then(data => displayResults(data))
            .catch(err => {
                displayResults({error: "Prediction failed: " + err});
            });
        });
    }

    // Helper: Downscale canvas to 28x28
    function downscaleTo28x28(srcCanvas) {
        const tmp = document.createElement('canvas');
        tmp.width = 28;
        tmp.height = 28;
        const tmpCtx = tmp.getContext('2d');
        // Fill white background
        tmpCtx.fillStyle = "#fff";
        tmpCtx.fillRect(0,0,28,28);
        // Draw the source canvas scaled down
        tmpCtx.drawImage(srcCanvas, 0, 0, 28, 28);
        // Return as Base64 PNG
        return tmp.toDataURL('image/png');
    }

    // Helper: Reset results display
    function resetResults() {
        const predictedDigitElement = document.getElementById('predicted-digit');
        const confidenceElement = document.getElementById('confidence');
        const scoresGrid = document.querySelector('.scores-grid');
        if (predictedDigitElement) predictedDigitElement.textContent = '-';
        if (confidenceElement) confidenceElement.textContent = 'Confidence: -';
        if (scoresGrid) scoresGrid.innerHTML = '';
    }

    // Helper: Display prediction/results
    function displayResults(data) {
        const predictedDigitElement = document.getElementById('predicted-digit');
        const confidenceElement = document.getElementById('confidence');
        const scoresGrid = document.querySelector('.scores-grid');
        if ("error" in data) {
            if (predictedDigitElement) predictedDigitElement.textContent = '?';
            if (confidenceElement) confidenceElement.textContent = data.error;
            if (scoresGrid) scoresGrid.innerHTML = '';
            return;
        }
        if (predictedDigitElement) predictedDigitElement.textContent = data.digit ?? '?';
        if (confidenceElement) confidenceElement.textContent =
            ("confidence" in data) ? `Confidence: ${(data.confidence*100).toFixed(1)}%` : 'Confidence: -';

        // Show confidence scores for all digits if available
        if (scoresGrid && data.scores) {
            scoresGrid.innerHTML = '';
            for (let i = 0; i < data.scores.length; i++) {
                const el = document.createElement('div');
                el.textContent = `${i}: ${(data.scores[i]*100).toFixed(1)}%`;
                scoresGrid.appendChild(el);
            }
        }
    }

    // Optionally: Disable submit feedback until a digit is predicted
    // (LEFT FOR YOU to handle depending on how you want to implement feedback logic)
});
