// Status tracking and notification functionality

// Global variables to track status polling
let requestId = null;
let statusInterval = null;
let currentStage = null;
let isCompleted = false;
let pollCount = 0;
let isPollingActive = false;
const MAX_POLL_COUNT = 1; // Stop polling after just 1 completed status

// Force stop polling after 1 minute regardless of status
const MAX_POLLING_TIME = 60000; // 1 minute in milliseconds
let pollingStartTime = 0;

// Ensure polling stops when page is unloaded
window.addEventListener('beforeunload', function() {
    console.log('Page unloading, stopping status polling');
    if (statusInterval) {
        clearInterval(statusInterval);
        statusInterval = null;
    }
});

// Ensure polling stops when page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.visibilityState === 'hidden') {
        console.log('Page hidden, stopping status polling');
        if (statusInterval) {
            clearInterval(statusInterval);
            statusInterval = null;
        }
    }
});

// Initialize status tracking
function initStatusTracking(id) {
    // Reset variables
    requestId = id;
    currentStage = null;
    isCompleted = false;
    pollCount = 0;
    isPollingActive = true;
    pollingStartTime = Date.now();

    // Start polling for status updates
    if (statusInterval) {
        clearInterval(statusInterval);
        statusInterval = null;
    }

    console.log(`Starting status polling for request ${id}`);

    // Use a longer interval (2 seconds) to reduce the number of requests
    statusInterval = setInterval(checkStatus, 2000);

    // Immediately check status once
    checkStatus();

    // Force stop polling after MAX_POLLING_TIME
    setTimeout(() => {
        if (isPollingActive) {
            console.log(`Forcing stop of polling after ${MAX_POLLING_TIME/1000} seconds`);
            stopPolling();
        }
    }, MAX_POLLING_TIME);

    // Show the status container with animation
    const statusContainer = document.getElementById('status-container');
    statusContainer.style.display = 'block';
    statusContainer.style.opacity = '0';
    statusContainer.style.transform = 'translateY(-20px)';
    statusContainer.style.transition = 'opacity 0.5s, transform 0.5s';

    // Trigger animation
    setTimeout(() => {
        statusContainer.style.opacity = '1';
        statusContainer.style.transform = 'translateY(0)';
    }, 10);

    // Add close button functionality
    const closeButton = statusContainer.querySelector('.btn-close');
    if (closeButton) {
        closeButton.addEventListener('click', () => {
            statusContainer.style.opacity = '0';
            statusContainer.style.transform = 'translateY(-20px)';
            setTimeout(() => {
                statusContainer.style.display = 'none';
            }, 500);
        });
    }

    // Immediately check status
    checkStatus();
}

// Stop polling function
function stopPolling() {
    if (statusInterval) {
        console.log('Stopping status polling');
        clearInterval(statusInterval);
        statusInterval = null;
    }
    requestId = null;
    isPollingActive = false;

    // Update UI to show polling has stopped
    const statusMessage = document.getElementById('status-message');
    if (statusMessage && !isCompleted) {
        statusMessage.textContent += ' (Status updates stopped)';
    }
}

// Check the current status
function checkStatus() {
    // Safety checks to prevent zombie polling
    if (!requestId || !isPollingActive || !statusInterval) {
        stopPolling();
        return;
    }

    // Check if we've been polling too long
    const currentTime = Date.now();
    if (currentTime - pollingStartTime > MAX_POLLING_TIME) {
        console.log(`Polling timeout reached (${MAX_POLLING_TIME/1000} seconds)`);
        stopPolling();
        return;
    }

    // Use a timeout to ensure the fetch doesn't hang indefinitely
    const fetchPromise = fetch(`/api/status/${requestId}`);
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Status fetch timeout')), 5000);
    });

    Promise.race([fetchPromise, timeoutPromise])
        .then(response => {
            if (!response.ok) {
                throw new Error('Status check failed');
            }
            return response.json();
        })
        .then(data => {
            // Check if polling should still be active
            if (!isPollingActive || !statusInterval) {
                stopPolling();
                return;
            }

            updateStatusDisplay(data);

            // If process is completed or failed, handle accordingly
            if (data.status === 'completed' || data.status === 'failed') {
                if (!isCompleted) {
                    isCompleted = true;
                    // Show completion notification
                    showCompletionNotification(data.status === 'completed');
                    // Immediately increment poll count to speed up stopping
                    pollCount++;
                }

                // Increment poll count for completed status
                pollCount++;
                console.log(`Completed status poll count: ${pollCount}/${MAX_POLL_COUNT}`);

                // Stop polling after MAX_POLL_COUNT consecutive completed statuses
                if (pollCount >= MAX_POLL_COUNT) {
                    console.log('Stopping status polling after completion');
                    stopPolling();
                }
            } else {
                // Reset poll count if status is not completed
                pollCount = 0;
                isCompleted = false;
            }
        })
        .catch(error => {
            console.error('Error checking status:', error);
            // Stop polling on any error
            console.log('Stopping status polling due to error');
            stopPolling();
        });
}

// Update the status display
function updateStatusDisplay(statusData) {
    // Update progress bar
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = `${statusData.progress}%`;
    progressBar.setAttribute('aria-valuenow', statusData.progress);

    // Update progress color based on status
    if (statusData.status === 'completed') {
        progressBar.classList.remove('bg-danger', 'bg-warning', 'bg-info');
        progressBar.classList.add('bg-success');
    } else if (statusData.status === 'failed') {
        progressBar.classList.remove('bg-success', 'bg-warning', 'bg-info');
        progressBar.classList.add('bg-danger');
    } else if (statusData.progress > 70) {
        progressBar.classList.remove('bg-danger', 'bg-success', 'bg-info');
        progressBar.classList.add('bg-warning');
    } else {
        progressBar.classList.remove('bg-danger', 'bg-success', 'bg-warning');
        progressBar.classList.add('bg-info');
    }

    // Update status message
    const statusMessage = document.getElementById('status-message');
    statusMessage.textContent = statusData.message || 'Processing...';

    // Update status message style based on status
    if (statusData.status === 'completed') {
        statusMessage.className = 'mb-3 text-success fw-bold';
    } else if (statusData.status === 'failed') {
        statusMessage.className = 'mb-3 text-danger fw-bold';
    } else {
        statusMessage.className = 'mb-3';
    }

    // Check if stage has changed
    if (statusData.current_stage && statusData.current_stage !== currentStage) {
        currentStage = statusData.current_stage;
        showStageNotification(currentStage, statusData.stages);
    }

    // Update stages list
    updateStagesList(statusData.stages);

    // Auto-expand stages accordion if there are stages
    if (statusData.stages && statusData.stages.length > 0) {
        const collapseStages = document.getElementById('collapseStages');
        if (collapseStages && !collapseStages.classList.contains('show')) {
            const stagesButton = document.querySelector('[data-bs-target="#collapseStages"]');
            if (stagesButton && stagesButton.classList.contains('collapsed')) {
                // Only auto-expand once
                if (statusData.stages.length === 1) {
                    new bootstrap.Collapse(collapseStages, { toggle: true });
                }
            }
        }
    }
}

// Show a notification when a new stage begins
function showStageNotification(stageName, stages) {
    const stage = stages.find(s => s.name === stageName);
    if (!stage) return;

    const stageNames = {
        'setup': 'Setting up LLM models',
        'prompt_formatting': 'Formatting scraping prompt',
        'code_generation': 'Generating scraping code',
        'code_extraction': 'Extracting clean code',
        'code_execution': 'Executing the generated code',
        'data_validation': 'Validating extracted data',
        'refinement': 'Refining code'
    };

    const title = stageNames[stageName] || stageName;
    const message = stage.details || 'Processing...';

    // Create and show the notification
    const notification = document.createElement('div');
    notification.className = 'toast show';
    notification.setAttribute('role', 'alert');
    notification.setAttribute('aria-live', 'assertive');
    notification.setAttribute('aria-atomic', 'true');

    notification.innerHTML = `
        <div class="toast-header">
            <strong class="me-auto">${title}</strong>
            <small>just now</small>
            <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;

    const toastContainer = document.getElementById('toast-container');
    toastContainer.appendChild(notification);

    // Remove the notification after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 500);
    }, 5000);
}

// Show completion notification
function showCompletionNotification(success) {
    const title = success ? 'Process Completed' : 'Process Failed';
    const message = success ? 'The scraping process has completed successfully!' : 'The scraping process failed. Check the results for details.';
    const bgClass = success ? 'bg-success' : 'bg-danger';

    // Create and show the notification
    const notification = document.createElement('div');
    notification.className = 'toast show';
    notification.setAttribute('role', 'alert');
    notification.setAttribute('aria-live', 'assertive');
    notification.setAttribute('aria-atomic', 'true');

    notification.innerHTML = `
        <div class="toast-header ${bgClass} text-white">
            <strong class="me-auto">${title}</strong>
            <small>just now</small>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;

    const toastContainer = document.getElementById('toast-container');
    toastContainer.appendChild(notification);

    // Remove the notification after 10 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 500);
    }, 10000);

    // Add a visual indicator to the status container
    const statusContainer = document.getElementById('status-container');
    if (statusContainer) {
        if (success) {
            statusContainer.classList.add('border-success');
        } else {
            statusContainer.classList.add('border-danger');
        }
    }
}

// Update the stages list
function updateStagesList(stages) {
    const stagesList = document.getElementById('stages-list');
    stagesList.innerHTML = '';

    stages.forEach((stage, index) => {
        const li = document.createElement('li');
        li.className = 'list-group-item d-flex justify-content-between align-items-center';

        // Add animation for the current stage
        if (stage.status === 'in_progress') {
            li.classList.add('active', 'bg-light');
        }

        // Format stage name
        const stageName = stage.name.replace(/_/g, ' ');
        const formattedName = stageName.charAt(0).toUpperCase() + stageName.slice(1);

        // Calculate duration if available
        let durationText = '';
        if (stage.duration) {
            durationText = `${stage.duration.toFixed(2)}s`;
        }

        // Set status badge
        let badgeClass = 'bg-primary';
        let statusIcon = '<i class="bi bi-hourglass-split"></i>';

        if (stage.status === 'completed') {
            badgeClass = 'bg-success';
            statusIcon = '<i class="bi bi-check-circle-fill"></i>';
        } else if (stage.status === 'failed') {
            badgeClass = 'bg-danger';
            statusIcon = '<i class="bi bi-x-circle-fill"></i>';
        } else if (stage.status === 'in_progress') {
            badgeClass = 'bg-primary';
            statusIcon = '<i class="bi bi-arrow-repeat spin"></i>';
        }

        // Add a stage number
        const stageNumber = index + 1;

        li.innerHTML = `
            <div class="d-flex align-items-center">
                <span class="badge rounded-pill ${badgeClass} me-2">${stageNumber}</span>
                <div>
                    <strong>${formattedName}</strong>
                    ${stage.details ? `<div><small class="text-muted">${stage.details}</small></div>` : ''}
                </div>
            </div>
            <div>
                <span class="badge ${badgeClass}">${statusIcon} ${stage.status}</span>
                ${durationText ? `<small class="ms-2">${durationText}</small>` : ''}
            </div>
        `;

        // Add hover effect
        li.style.transition = 'background-color 0.3s';
        li.addEventListener('mouseover', () => {
            if (stage.status !== 'in_progress') {
                li.style.backgroundColor = '#f8f9fa';
            }
        });
        li.addEventListener('mouseout', () => {
            if (stage.status !== 'in_progress') {
                li.style.backgroundColor = '';
            }
        });

        stagesList.appendChild(li);
    });
}

// Display token usage information
function displayTokenUsage(helperUsage, codingUsage, totalCost) {
    const tokenUsageContainer = document.getElementById('token-usage');
    if (!tokenUsageContainer) return;

    // Prepare for animation
    tokenUsageContainer.style.display = 'block';
    tokenUsageContainer.style.opacity = '0';
    tokenUsageContainer.style.transform = 'translateY(20px)';
    tokenUsageContainer.style.transition = 'opacity 0.5s, transform 0.5s';

    // Helper LLM usage
    document.getElementById('helper-tokens').textContent = helperUsage.total_tokens.toLocaleString();
    document.getElementById('helper-cost').textContent = `$${helperUsage.cost.toFixed(4)}`;

    // Coding LLM usage
    document.getElementById('coding-tokens').textContent = codingUsage.total_tokens.toLocaleString();
    document.getElementById('coding-cost').textContent = `$${codingUsage.cost.toFixed(4)}`;

    // Total cost
    const totalCostElement = document.getElementById('total-cost');
    totalCostElement.textContent = `$${totalCost.toFixed(4)}`;

    // Add cost color based on amount
    if (totalCost > 0.1) {
        totalCostElement.className = 'text-danger fw-bold';
    } else if (totalCost > 0.05) {
        totalCostElement.className = 'text-warning fw-bold';
    } else {
        totalCostElement.className = 'text-success fw-bold';
    }

    // Create pie chart for token distribution
    createTokenDistributionChart(helperUsage, codingUsage);

    // Animate in after a short delay
    setTimeout(() => {
        tokenUsageContainer.style.opacity = '1';
        tokenUsageContainer.style.transform = 'translateY(0)';
    }, 500);

    // Add a visual indicator to highlight the cost information
    setTimeout(() => {
        const costAlert = document.querySelector('.alert-info');
        if (costAlert) {
            costAlert.classList.add('highlight-pulse');
            setTimeout(() => {
                costAlert.classList.remove('highlight-pulse');
            }, 2000);
        }
    }, 1000);
}

// Create a simple token distribution chart
function createTokenDistributionChart(helperUsage, codingUsage) {
    // Check if chart container exists
    const chartContainer = document.getElementById('token-chart-container');
    if (!chartContainer) return;

    // Calculate percentages
    const helperTokens = helperUsage.total_tokens;
    const codingTokens = codingUsage.total_tokens;
    const totalTokens = helperTokens + codingTokens;

    const helperPercentage = (helperTokens / totalTokens * 100).toFixed(1);
    const codingPercentage = (codingTokens / totalTokens * 100).toFixed(1);

    // Create chart HTML
    chartContainer.innerHTML = `
        <h6 class="mb-3">Token Distribution</h6>
        <div class="progress mb-2" style="height: 25px;">
            <div class="progress-bar bg-info" role="progressbar" style="width: ${helperPercentage}%"
                aria-valuenow="${helperPercentage}" aria-valuemin="0" aria-valuemax="100">
                Helper LLM ${helperPercentage}%
            </div>
            <div class="progress-bar bg-primary" role="progressbar" style="width: ${codingPercentage}%"
                aria-valuenow="${codingPercentage}" aria-valuemin="0" aria-valuemax="100">
                Coding LLM ${codingPercentage}%
            </div>
        </div>
        <div class="small text-muted text-center">
            Total Tokens: ${totalTokens.toLocaleString()}
        </div>
    `;
}
