// Website Analysis Display

// Initialize the website analysis display
function initWebsiteAnalysis() {
    const analysisContainer = document.getElementById('website-analysis');
    if (!analysisContainer) return;
    
    // Add animation to the container
    analysisContainer.style.opacity = '0';
    analysisContainer.style.transform = 'translateY(20px)';
    
    // Animate in after a short delay
    setTimeout(() => {
        analysisContainer.style.opacity = '1';
        analysisContainer.style.transform = 'translateY(0)';
    }, 300);
    
    // Add click handlers for the raw data toggle
    const rawDataToggle = document.querySelector('[data-bs-toggle="collapse"][data-bs-target="#rawAnalysisData"]');
    if (rawDataToggle) {
        rawDataToggle.addEventListener('click', function() {
            const isCollapsed = this.getAttribute('aria-expanded') === 'false';
            this.textContent = isCollapsed ? 'Hide Raw Analysis Data' : 'Show Raw Analysis Data';
        });
    }
    
    // Highlight recommendations
    const recommendationsSection = document.querySelector('#website-analysis .alert-info');
    if (recommendationsSection) {
        setTimeout(() => {
            recommendationsSection.classList.add('highlight-pulse');
            setTimeout(() => {
                recommendationsSection.classList.remove('highlight-pulse');
            }, 1500);
        }, 1000);
    }
}

// Initialize when document is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize website analysis display
    initWebsiteAnalysis();
});
