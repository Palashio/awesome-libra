// Interactive Demo JavaScript Functionality

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDemo();
});

// Initialize all demo functionality
function initializeDemo() {
    setupCodeExampleSwitcher();
    setupCopyToClipboard();
    setupSmoothScrolling();
    setupDynamicContentLoading();
    setupNavigationHighlighting();
}

// Code Example Switcher Functionality
function setupCodeExampleSwitcher() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const exampleTabs = document.querySelectorAll('.example-tab');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetExample = this.getAttribute('onclick').match(/'([^']+)'/)[1];
            showExample(targetExample);
        });
    });
}

// Show specific example tab with animation
function showExample(exampleType) {
    // Remove active class from all tabs and buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.example-tab').forEach(tab => tab.classList.remove('active'));
    
    // Add active class to selected button and tab
    const activeButton = document.querySelector(`[onclick*="${exampleType}"]`);
    const activeTab = document.getElementById(exampleType);
    
    if (activeButton && activeTab) {
        activeButton.classList.add('active');
        activeTab.classList.add('active');
        
        // Trigger syntax highlighting for the newly shown code
        if (typeof Prism !== 'undefined') {
            Prism.highlightAllUnder(activeTab);
        }
    }
}

// Copy to Clipboard Functionality
function setupCopyToClipboard() {
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const codeId = this.getAttribute('onclick').match(/'([^']+)'/)[1];
            copyCode(codeId);
        });
    });
}

// Copy code to clipboard with visual feedback
function copyCode(codeId) {
    const codeElement = document.getElementById(codeId);
    const copyButton = document.querySelector(`[onclick*="${codeId}"]`);
    
    if (codeElement && copyButton) {
        const codeText = codeElement.textContent || codeElement.innerText;
        
        // Use modern clipboard API if available
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(codeText).then(() => {
                showCopyFeedback(copyButton, 'Copied!');
            }).catch(() => {
                fallbackCopyToClipboard(codeText, copyButton);
            });
        } else {
            fallbackCopyToClipboard(codeText, copyButton);
        }
    }
}

// Fallback copy method for older browsers
function fallbackCopyToClipboard(text, button) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-999999px';
    textArea.style.top = '-999999px';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    try {
        document.execCommand('copy');
        showCopyFeedback(button, 'Copied!');
    } catch (err) {
        showCopyFeedback(button, 'Copy failed');
    }
    
    document.body.removeChild(textArea);
}

// Show visual feedback for copy action
function showCopyFeedback(button, message) {
    const originalText = button.textContent;
    button.textContent = message;
    button.style.background = '#10b981'; // Success color
    
    setTimeout(() => {
        button.textContent = originalText;
        button.style.background = ''; // Reset to original
    }, 2000);
}

// Smooth Scrolling Navigation
function setupSmoothScrolling() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            scrollToSection(targetId);
        });
    });
}

// Smooth scroll to section with offset for fixed navbar
function scrollToSection(sectionId) {
    const targetSection = document.getElementById(sectionId);
    const navbar = document.querySelector('.navbar');
    
    if (targetSection) {
        const navbarHeight = navbar ? navbar.offsetHeight : 0;
        const targetPosition = targetSection.offsetTop - navbarHeight - 20;
        
        window.scrollTo({
            top: targetPosition,
            behavior: 'smooth'
        });
    }
}

// Dynamic Content Loading for Different ML Scenarios
function setupDynamicContentLoading() {
    // Add loading states and dynamic content updates
    const exampleTabs = document.querySelectorAll('.example-tab');
    
    exampleTabs.forEach(tab => {
        // Add intersection observer for lazy loading
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    loadDynamicContent(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });
        
        observer.observe(tab);
    });
}

// Load dynamic content for ML scenarios
function loadDynamicContent(tabElement) {
    const tabId = tabElement.id;
    
    // Add dynamic content based on the ML scenario
    switch(tabId) {
        case 'classification':
            enhanceClassificationExample(tabElement);
            break;
        case 'regression':
            enhanceRegressionExample(tabElement);
            break;
        case 'neural-network':
            enhanceNeuralNetworkExample(tabElement);
            break;
        case 'clustering':
            enhanceClusteringExample(tabElement);
            break;
    }
}

// Enhance classification example with dynamic content
function enhanceClassificationExample(tab) {
    // Add dynamic metrics or additional info
    addDynamicMetrics(tab, 'Classification Accuracy: 94.2%');
}

// Enhance regression example with dynamic content
function enhanceRegressionExample(tab) {
    addDynamicMetrics(tab, 'R² Score: 0.87 | RMSE: 12.3k');
}

// Enhance neural network example with dynamic content
function enhanceNeuralNetworkExample(tab) {
    addDynamicMetrics(tab, 'Training Epochs: 50 | Validation Loss: 0.023');
