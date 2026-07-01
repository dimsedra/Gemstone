document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const imagePreview = document.getElementById("image-preview");
    const resultContainer = document.getElementById("result-container");
    const topPrediction = document.getElementById("top-prediction");
    const confidenceBar = document.getElementById("confidence-bar");
    const confidenceText = document.getElementById("confidence-text");
    const altPredictionsList = document.getElementById("alt-predictions-list");
    const modelStatus = document.getElementById("model-status");
    const classesCount = document.getElementById("classes-count");
    const supportedClassesList = document.getElementById("supported-classes-list");

    let supportedClasses = [];

    // Fetch model info
    fetch("/info")
        .then(res => res.json())
        .then(data => {
            if (data.trained) {
                modelStatus.textContent = `Online (${data.total_classes} Classes)`;
                classesCount.textContent = data.total_classes;
                supportedClasses = data.classes;
                
                // Render supported classes
                supportedClassesList.innerHTML = data.classes.map(cls => `<li>${cls}</li>`).join('');
                
                // Reload training metrics image if available to bypass browser caching
                const metricsChart = document.getElementById("metrics-chart");
                const metricsPlaceholder = document.getElementById("metrics-placeholder");
                if (data.metrics_chart_available) {
                    metricsChart.src = `/models/training_metrics.png?t=${new Date().getTime()}`;
                    metricsChart.style.display = "block";
                    metricsPlaceholder.style.display = "none";
                } else {
                    metricsChart.style.display = "none";
                    metricsPlaceholder.style.display = "block";
                }
            } else {
                modelStatus.textContent = "Offline (Model not trained)";
                modelStatus.style.color = "#ef4444";
                document.getElementById("metrics-chart").style.display = "none";
                document.getElementById("metrics-placeholder").style.display = "block";
            }
        })
        .catch(err => {
            modelStatus.textContent = "Offline (Server error)";
            modelStatus.style.color = "#ef4444";
        });

    // Handle upload events
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    ["dragleave", "drop"].forEach(event => {
        dropZone.addEventListener(event, () => dropZone.classList.remove("dragover"));
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length) {
            fileInput.files = files;
            handleImageUpload(files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            handleImageUpload(fileInput.files[0]);
        }
    });

    function handleImageUpload(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file.");
            return;
        }

        resultContainer.style.display = "none";

        // Show Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = "block";
        };
        reader.readAsDataURL(file);

        // Upload to API
        const formData = new FormData();
        formData.append("file", file);

        modelStatus.textContent = "Analyzing Image...";
        modelStatus.style.color = "#3b82f6";

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(async res => {
            if (!res.ok) {
                const errorData = await res.json().catch(() => ({}));
                throw new Error(errorData.detail || "Classification request failed.");
            }
            return res.json();
        })
        .then(data => {
            modelStatus.textContent = "Online";
            modelStatus.style.color = "";
            displayResults(data);
        })
        .catch(err => {
            alert("Error identifying gemstone: " + err.message);
            modelStatus.textContent = "Online";
            modelStatus.style.color = "";
        });
    }

    function displayResults(data) {
        resultContainer.style.display = "block";
        topPrediction.textContent = data.prediction;
        
        const confidencePercent = (data.confidence * 100).toFixed(1);
        confidenceText.textContent = `${confidencePercent}%`;
        confidenceBar.style.width = `${confidencePercent}%`;

        // Dynamic colors based on confidence
        if (data.confidence > 0.8) {
            confidenceBar.style.background = "linear-gradient(135deg, #10b981, #06b6d4)"; // green gradient
            confidenceText.style.color = "#10b981";
        } else if (data.confidence > 0.5) {
            confidenceBar.style.background = "linear-gradient(135deg, #f59e0b, #d97706)"; // amber gradient
            confidenceText.style.color = "#f59e0b";
        } else {
            confidenceBar.style.background = "linear-gradient(135deg, #ef4444, #b91c1c)"; // red gradient
            confidenceText.style.color = "#ef4444";
        }

        // Build top 5 alternatives (skipping the first primary match)
        const alternatives = data.top_5.slice(1);
        altPredictionsList.innerHTML = alternatives.map(item => `
            <div class="alt-item">
                <span class="alt-name">${item.class}</span>
                <span class="alt-prob">${(item.confidence * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    }

    // Tab Switcher
    window.switchTab = function(tabName, event) {
        document.querySelectorAll(".tab-content").forEach(el => el.style.display = "none");
        document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
        
        document.getElementById(`${tabName}-tab`).style.display = "block";
        if (event && event.currentTarget) {
            event.currentTarget.classList.add("active");
        } else {
            const activeBtn = document.querySelector(`.tab-btn[onclick*="'${tabName}'"]`);
            if (activeBtn) {
                activeBtn.classList.add("active");
            }
        }
    };

    // Search filter for gemstone list
    window.filterClasses = function() {
        const query = document.getElementById("class-search").value.toLowerCase();
        const items = supportedClassesList.getElementsByTagName("li");
        
        for (let i = 0; i < items.length; i++) {
            const text = items[i].textContent || items[i].innerText;
            if (text.toLowerCase().indexOf(query) > -1) {
                items[i].style.display = "";
            } else {
                items[i].style.display = "none";
            }
        }
    };
});
