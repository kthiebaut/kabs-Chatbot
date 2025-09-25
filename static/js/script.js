// // Global state
// let selectedFiles = [];
// let chatHistory = [];

// // DOM elements
// const uploadArea = document.getElementById('uploadArea');
// const fileInput = document.getElementById('fileInput');
// const selectedFilesDiv = document.getElementById('selectedFiles');
// const fileList = document.getElementById('fileList');
// const processBtn = document.getElementById('processBtn');
// const uploadedFilesList = document.getElementById('uploadedFilesList');
// const statusIndicator = document.getElementById('statusIndicator');
// const chatContainer = document.getElementById('chatContainer');
// const chatInput = document.getElementById('chatInput');
// const sendBtn = document.getElementById('sendBtn');
// const clearChatBtn = document.getElementById('clearChatBtn');
// const clearDataBtn = document.getElementById('clearDataBtn');

// // File type icons
// const fileIcons = {
//     pdf: 'fas fa-file-pdf',
//     csv: 'fas fa-file-csv',
//     xlsx: 'fas fa-file-excel',
//     xls: 'fas fa-file-excel',
//     txt: 'fas fa-file-alt',
//     doc: 'fas fa-file-word',
//     docx: 'fas fa-file-word'
// };

// // Enhanced message formatting function
// function formatMessageContent(content) {
//     // Convert markdown-style formatting to HTML
//     let formatted = content;
    
//     // Handle line breaks
//     formatted = formatted.replace(/\n/g, '<br>');
    
//     // Handle bold text
//     formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
//     // Handle bullet points (more comprehensive pattern)
//     formatted = formatted.replace(/^[•\-\*]\s+(.+)$/gm, '<div class="bullet-point">• $1</div>');
    
//     // Handle numbered lists
//     formatted = formatted.replace(/^(\d+)\.\s+(.+)$/gm, '<div class="numbered-item">$1. $2</div>');
    
//     // Handle price formatting (look for currency symbols - enhanced pattern)
//     formatted = formatted.replace(/(\$[\d,]+\.?\d*|\$\d+(?:\.\d{2})?)/g, '<span class="price">$1</span>');
    
//     // Handle product codes (pattern like WRH3027, RH4224, etc.)
//     formatted = formatted.replace(/\b([A-Z]{2,4}\d{3,5}(?:\s+[A-Z]{1,3})?)\b/g, '<span class="product-name">$1</span>');
    
//     // Handle sources section
//     formatted = formatted.replace(/\*\*Sources:\*\*/g, '<strong>Sources:</strong>');
    
//     return formatted;
// }

// // Initialize
// function init() {
//     setupEventListeners();
//     loadUploadedFiles();
//     updateStatus();
// }

// function setupEventListeners() {
//     // Upload area events
//     uploadArea.addEventListener('click', () => fileInput.click());
//     uploadArea.addEventListener('dragover', handleDragOver);
//     uploadArea.addEventListener('dragleave', handleDragLeave);
//     uploadArea.addEventListener('drop', handleDrop);

//     // File input
//     fileInput.addEventListener('change', handleFileSelect);

//     // Process button
//     processBtn.addEventListener('click', processFiles);

//     // Chat input events
//     chatInput.addEventListener('keypress', (e) => {
//         if (e.key === 'Enter') {
//             e.preventDefault();
//             sendMessage();
//         }
//     });
    
//     // Send button
//     sendBtn.addEventListener('click', sendMessage);

//     // Control buttons
//     clearChatBtn.addEventListener('click', clearChat);
//     clearDataBtn.addEventListener('click', clearAllData);
// }

// function handleDragOver(e) {
//     e.preventDefault();
//     uploadArea.classList.add('dragover');
// }

// function handleDragLeave(e) {
//     e.preventDefault();
//     uploadArea.classList.remove('dragover');
// }

// function handleDrop(e) {
//     e.preventDefault();
//     uploadArea.classList.remove('dragover');
//     const files = Array.from(e.dataTransfer.files);
//     addFilesToSelection(files);
// }

// function handleFileSelect(e) {
//     const files = Array.from(e.target.files);
//     addFilesToSelection(files);
// }

// function addFilesToSelection(files) {
//     files.forEach(file => {
//         if (!selectedFiles.find(f => f.name === file.name)) {
//             selectedFiles.push(file);
//         }
//     });
//     updateSelectedFiles();
// }

// function updateSelectedFiles() {
//     if (selectedFiles.length === 0) {
//         selectedFilesDiv.style.display = 'none';
//         processBtn.style.display = 'none';
//         return;
//     }

//     selectedFilesDiv.style.display = 'block';
//     processBtn.style.display = 'block';

//     fileList.innerHTML = selectedFiles.map((file, index) => {
//         const ext = file.name.split('.').pop().toLowerCase();
//         const icon = fileIcons[ext] || 'fas fa-file';
//         const size = formatFileSize(file.size);

//         return `
//             <div class="file-item">
//                 <div class="file-icon ${ext}">
//                     <i class="${icon}"></i>
//                 </div>
//                 <div class="file-info">
//                     <div class="file-name">${file.name}</div>
//                     <div class="file-size">${size}</div>
//                 </div>
//                 <div class="remove-file" onclick="removeFile(${index})">
//                     <i class="fas fa-times"></i>
//                 </div>
//             </div>
//         `;
//     }).join('');
// }

// function removeFile(index) {
//     selectedFiles.splice(index, 1);
//     updateSelectedFiles();
// }

// function formatFileSize(bytes) {
//     if (bytes === 0) return '0 Bytes';
//     const k = 1024;
//     const sizes = ['Bytes', 'KB', 'MB', 'GB'];
//     const i = Math.floor(Math.log(bytes) / Math.log(k));
//     return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
// }

// async function processFiles() {
//     if (selectedFiles.length === 0) return;

//     // Disable button and show loading state
//     processBtn.disabled = true;
//     processBtn.innerHTML = '<div class="spinner"></div> Processing...';

//     const formData = new FormData();
//     selectedFiles.forEach(file => {
//         formData.append('files', file);
//     });

//     try {
//         const response = await fetch('/upload', {
//             method: 'POST',
//             body: formData
//         });

//         const result = await response.json();

//         if (result.success) {
//             // Clear selection
//             selectedFiles = [];
//             fileInput.value = '';
//             updateSelectedFiles();
//             loadUploadedFiles();
//             updateStatus();

//             addMessage('assistant', `✅ ${result.message}`);
//         } else {
//             addMessage('assistant', `❌ Error: ${result.message}`);
//         }
//     } catch (error) {
//         console.error('Upload error:', error);
//         addMessage('assistant', '❌ Failed to upload files. Please try again.');
//     } finally {
//         // Re-enable button
//         processBtn.disabled = false;
//         processBtn.innerHTML = '<i class="fas fa-rocket"></i> Process Files';
//     }
// }

// async function loadUploadedFiles() {
//     try {
//         const response = await fetch('/files');
//         const files = await response.json();

//         if (files.length === 0) {
//             uploadedFilesList.innerHTML = `
//                 <div style="color: #6c757d; text-align: center; padding: 20px;">
//                     No files uploaded yet
//                 </div>
//             `;
//             return;
//         }

//         uploadedFilesList.innerHTML = files.map(file => {
//             const icon = fileIcons[file.type] || 'fas fa-file';

//             return `
//                 <div class="file-item">
//                     <div class="file-icon ${file.type}">
//                         <i class="${icon}"></i>
//                     </div>
//                     <div class="file-info">
//                         <div class="file-name">${file.name}</div>
//                         <div class="file-size">${file.type.toUpperCase()}</div>
//                     </div>
//                 </div>
//             `;
//         }).join('');
//     } catch (error) {
//         console.error('Error loading files:', error);
//         uploadedFilesList.innerHTML = `
//             <div style="color: #dc3545; text-align: center; padding: 20px;">
//                 Error loading files
//             </div>
//         `;
//     }
// }

// async function updateStatus() {
//     try {
//         const response = await fetch('/status');
//         const status = await response.json();

//         if (status.count > 0) {
//             statusIndicator.className = 'status-indicator status-ready';
//             statusIndicator.innerHTML = `
//                 <i class="fas fa-check-circle"></i>
//                 Ready • ${status.count} documents indexed
//             `;
//             chatInput.disabled = false;
//             sendBtn.disabled = false;
//             chatInput.placeholder = "Ask about single or multiple items from your documents...";
//         } else {
//             statusIndicator.className = 'status-indicator status-waiting';
//             statusIndicator.innerHTML = `
//                 <i class="fas fa-clock"></i>
//                 Upload files to get started
//             `;
//             chatInput.disabled = true;
//             sendBtn.disabled = true;
//             chatInput.placeholder = "Upload documents first to start chatting...";
//         }
//     } catch (error) {
//         console.error('Error updating status:', error);
//         statusIndicator.className = 'status-indicator status-waiting';
//         statusIndicator.innerHTML = `
//             <i class="fas fa-exclamation-triangle"></i>
//             Error connecting to server
//         `;
//         chatInput.disabled = true;
//         sendBtn.disabled = true;
//     }
// }

// async function sendMessage() {
//     const message = chatInput.value.trim();
//     if (!message || chatInput.disabled) return;

//     // Add user message to chat
//     addMessage('user', message);
//     chatInput.value = '';

//     // Show typing indicator
//     const typingDiv = document.createElement('div');
//     typingDiv.className = 'message assistant';
//     typingDiv.id = 'typing-indicator';
//     typingDiv.innerHTML = `
//         <div class="message-avatar">🤖</div>
//         <div class="message-content">
//             <div class="loading">
//                 <div class="spinner"></div>
//                 Analyzing your query for multiple items...
//             </div>
//         </div>
//     `;
//     chatContainer.appendChild(typingDiv);
//     chatContainer.scrollTop = chatContainer.scrollHeight;

//     try {
//         const response = await fetch('/chat', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({ message: message })
//         });

//         if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//         }

//         const result = await response.json();

//         // Remove typing indicator
//         const typingIndicator = document.getElementById('typing-indicator');
//         if (typingIndicator) {
//             chatContainer.removeChild(typingIndicator);
//         }

//         if (result.success) {
//             addMessage('assistant', result.response);
//         } else {
//             addMessage('assistant', `❌ Error: ${result.message || 'Unknown error occurred'}`);
//         }
//     } catch (error) {
//         console.error('Chat error:', error);
        
//         // Remove typing indicator if it exists
//         const typingIndicator = document.getElementById('typing-indicator');
//         if (typingIndicator) {
//             chatContainer.removeChild(typingIndicator);
//         }
        
//         addMessage('assistant', '❌ Failed to get response. Please check your connection and try again.');
//     }
// }

// function addMessage(role, content) {
//     const message = { role, content, timestamp: new Date() };
//     chatHistory.push(message);

//     const messageDiv = document.createElement('div');
//     messageDiv.className = `message ${role}`;
    
//     const avatar = role === 'assistant' ? '🤖' : '👤';
    
//     // Format the content for better display (only for assistant messages)
//     const formattedContent = role === 'assistant' ? formatMessageContent(content) : escapeHtml(content);
    
//     messageDiv.innerHTML = `
//         <div class="message-avatar">${avatar}</div>
//         <div class="message-content">${formattedContent}</div>
//     `;

//     chatContainer.appendChild(messageDiv);
//     chatContainer.scrollTop = chatContainer.scrollHeight;
// }

// function escapeHtml(text) {
//     const div = document.createElement('div');
//     div.textContent = text;
//     return div.innerHTML;
// }

// function clearChat() {
//     chatHistory = [];
//     chatContainer.innerHTML = `
//         <div class="message assistant">
//             <div class="message-avatar">🤖</div>
//             <div class="message-content">
//                 <strong>Hello! I'm K&B Scout AI</strong>, your enhanced enterprise document assistant with improved formatting.<br><br>
//                 I can help you find information from your uploaded files with better structured responses. What would you like to know?<br><br>
//                 <em>Example: "What is the price of WRH3027 SV, WRH3624 DM, RH4224 RP?"</em>
//             </div>
//         </div>
//     `;
// }

// async function clearAllData() {
//     const confirmMessage = 'Are you sure you want to clear all data? This will remove all uploaded files and chat history.';
    
//     if (!confirm(confirmMessage)) {
//         return;
//     }

//     try {
//         const response = await fetch('/clear', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             }
//         });

//         if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//         }

//         const result = await response.json();

//         if (result.success) {
//             // Clear local state
//             selectedFiles = [];
//             chatHistory = [];
//             fileInput.value = '';
            
//             // Update UI
//             updateSelectedFiles();
//             loadUploadedFiles();
//             updateStatus();
//             clearChat();
            
//             addMessage('assistant', '🗑️ All data has been cleared successfully!');
//         } else {
//             addMessage('assistant', `❌ Error clearing data: ${result.message || 'Unknown error occurred'}`);
//         }
//     } catch (error) {
//         console.error('Clear data error:', error);
//         addMessage('assistant', '❌ Failed to clear data. Please check your connection and try again.');
//     }
// }

// // Utility function to handle errors gracefully
// function handleError(error, context = '') {
//     console.error(`Error in ${context}:`, error);
    
//     // Show user-friendly error message
//     const errorMessage = `❌ An error occurred${context ? ` in ${context}` : ''}. Please try again.`;
//     addMessage('assistant', errorMessage);
// }

// // Enhanced error handling for fetch requests
// async function safeFetch(url, options = {}) {
//     try {
//         const response = await fetch(url, {
//             ...options,
//             headers: {
//                 'Content-Type': 'application/json',
//                 ...options.headers
//             }
//         });

//         if (!response.ok) {
//             throw new Error(`HTTP ${response.status}: ${response.statusText}`);
//         }

//         return await response.json();
//     } catch (error) {
//         console.error('Fetch error:', error);
//         throw error;
//     }
// }

// // Initialize the application when DOM is ready
// document.addEventListener('DOMContentLoaded', function() {
//     init();
// });

// // Also initialize immediately if DOM is already loaded
// if (document.readyState === 'loading') {
//     document.addEventListener('DOMContentLoaded', init);
// } else {
//     init();
// }




// Global state
let selectedFiles = [];
let chatHistory = [];
let uploadedFiles = [];

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const selectedFilesDiv = document.getElementById('selectedFiles');
const fileList = document.getElementById('fileList');
const processBtn = document.getElementById('processBtn');
const uploadedFilesList = document.getElementById('uploadedFilesList');
const statusIndicator = document.getElementById('statusIndicator');
const chatContainer = document.getElementById('chatContainer');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const clearChatBtn = document.getElementById('clearChatBtn');
const clearDataBtn = document.getElementById('clearDataBtn');

// New elements for enhanced features
const querySuggestions = document.getElementById('querySuggestions');
const analyzePdfBtn = document.getElementById('analyzePdfBtn');
const multiQuoteBtn = document.getElementById('multiQuoteBtn');
const pdfModal = document.getElementById('pdfModal');
const multiQuoteModal = document.getElementById('multiQuoteModal');
const loadingOverlay = document.getElementById('loadingOverlay');

// File type icons
const fileIcons = {
    pdf: 'fas fa-file-pdf',
    csv: 'fas fa-file-csv',
    xlsx: 'fas fa-file-excel',
    xls: 'fas fa-file-excel',
    txt: 'fas fa-file-alt',
    doc: 'fas fa-file-word',
    docx: 'fas fa-file-word'
};

// Enhanced message formatting function
function formatMessageContent(content) {
    // Convert markdown-style formatting to HTML
    let formatted = content;
    
    // Handle line breaks
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Handle bold text
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Handle bullet points with enhanced detection
    formatted = formatted.replace(/^[•\-\*]\s+(.+)$/gm, '<div class="bullet-point">• $1</div>');
    
    // Handle numbered lists
    formatted = formatted.replace(/^(\d+)\.\s+(.+)$/gm, '<div class="numbered-item">$1. $2</div>');
    
    // Enhanced price formatting with better detection
    formatted = formatted.replace(/(\$[\d,]+\.?\d*|\$\d+(?:\.\d{2})?)/g, '<span class="price">$1</span>');
    
    // Handle total/summary prices specially
    formatted = formatted.replace(/(Total|Sum|Grand Total):\s*(\$[\d,]+\.?\d*)/gi, 
        '$1: <span class="total-price">$2</span>');
    
    // Handle product codes with enhanced detection
    formatted = formatted.replace(/\b([A-Z]{2,4}\d{3,5}(?:\s+[A-Z]{1,3})?)\b/g, 
        '<span class="product-name">$1</span>');
    
    // Handle table-like content (simple detection)
    formatted = formatted.replace(/^(.+\|.+\|.+)$/gm, '<div class="table-row">$1</div>');
    
    // Handle sources section
    formatted = formatted.replace(/\*\*Sources:\*\*/g, '<strong>Sources:</strong>');
    
    return formatted;
}

// Initialize application
function init() {
    setupEventListeners();
    loadUploadedFiles();
    updateStatus();
    setupQuerySuggestions();
}

function setupEventListeners() {
    // Upload area events
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // File input
    fileInput.addEventListener('change', handleFileSelect);

    // Process button
    processBtn.addEventListener('click', processFiles);

    // Chat input events
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Send button
    sendBtn.addEventListener('click', sendMessage);

    // Control buttons
    clearChatBtn.addEventListener('click', clearChat);
    clearDataBtn.addEventListener('click', clearAllData);

    // Enhanced feature buttons
    analyzePdfBtn.addEventListener('click', showPdfAnalysisModal);
    multiQuoteBtn.addEventListener('click', showMultiQuoteModal);

    // Modal event listeners
    setupModalEventListeners();
}

function setupModalEventListeners() {
    // PDF Analysis Modal
    document.getElementById('closePdfModal').addEventListener('click', () => {
        pdfModal.style.display = 'none';
    });
    
    document.getElementById('cancelPdfAnalysis').addEventListener('click', () => {
        pdfModal.style.display = 'none';
    });
    
    document.getElementById('analyzePdfConfirm').addEventListener('click', analyzePdfPricing);

    // Multi-Quote Modal
    document.getElementById('closeMultiModal').addEventListener('click', () => {
        multiQuoteModal.style.display = 'none';
    });
    
    document.getElementById('cancelMultiQuote').addEventListener('click', () => {
        multiQuoteModal.style.display = 'none';
    });
    
    document.getElementById('generateQuote').addEventListener('click', generateMultiQuote);

    // Close modals when clicking outside
    [pdfModal, multiQuoteModal].forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    });
}

function setupQuerySuggestions() {
    const suggestionItems = document.querySelectorAll('.suggestion-item');
    suggestionItems.forEach(item => {
        item.addEventListener('click', () => {
            const query = item.getAttribute('data-query');
            chatInput.value = query;
            chatInput.focus();
        });
    });
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    addFilesToSelection(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    addFilesToSelection(files);
}

function addFilesToSelection(files) {
    files.forEach(file => {
        if (!selectedFiles.find(f => f.name === file.name)) {
            selectedFiles.push(file);
        }
    });
    updateSelectedFiles();
}

function updateSelectedFiles() {
    if (selectedFiles.length === 0) {
        selectedFilesDiv.style.display = 'none';
        processBtn.style.display = 'none';
        return;
    }

    selectedFilesDiv.style.display = 'block';
    processBtn.style.display = 'block';

    fileList.innerHTML = selectedFiles.map((file, index) => {
        const ext = file.name.split('.').pop().toLowerCase();
        const icon = fileIcons[ext] || 'fas fa-file';
        const size = formatFileSize(file.size);

        return `
            <div class="file-item">
                <div class="file-icon ${ext}">
                    <i class="${icon}"></i>
                </div>
                <div class="file-info">
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${size}</div>
                </div>
                <div class="remove-file" onclick="removeFile(${index})">
                    <i class="fas fa-times"></i>
                </div>
            </div>
        `;
    }).join('');
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateSelectedFiles();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function processFiles() {
    if (selectedFiles.length === 0) return;

    showLoadingOverlay('Processing files...');
    
    // Disable button and show loading state
    processBtn.disabled = true;
    processBtn.innerHTML = '<div class="spinner"></div> Processing...';

    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            // Clear selection
            selectedFiles = [];
            fileInput.value = '';
            updateSelectedFiles();
            loadUploadedFiles();
            updateStatus();

            addMessage('assistant', `✅ ${result.message}`);
        } else {
            addMessage('assistant', `❌ Error: ${result.message}`);
        }
    } catch (error) {
        console.error('Upload error:', error);
        addMessage('assistant', '❌ Failed to upload files. Please try again.');
    } finally {
        // Re-enable button
        processBtn.disabled = false;
        processBtn.innerHTML = '<i class="fas fa-rocket"></i> Process Files';
        hideLoadingOverlay();
    }
}

async function loadUploadedFiles() {
    try {
        const response = await fetch('/files');
        const files = await response.json();
        uploadedFiles = files;

        if (files.length === 0) {
            uploadedFilesList.innerHTML = `
                <div style="color: #6c757d; text-align: center; padding: 20px;">
                    No files uploaded yet
                </div>
            `;
            return;
        }

        uploadedFilesList.innerHTML = files.map(file => {
            const icon = fileIcons[file.type] || 'fas fa-file';

            return `
                <div class="file-item clickable" data-filename="${file.name}" data-type="${file.type}">
                    <div class="file-icon ${file.type}">
                        <i class="${icon}"></i>
                    </div>
                    <div class="file-info">
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${file.type.toUpperCase()}</div>
                    </div>
                </div>
            `;
        }).join('');

        // Add click handlers for file items
        document.querySelectorAll('.file-item.clickable').forEach(item => {
            item.addEventListener('click', () => {
                const filename = item.getAttribute('data-filename');
                const type = item.getAttribute('data-type');
                
                if (type === 'pdf') {
                    askAboutFile(filename, 'Give me pricing for all items in this PDF');
                } else {
                    askAboutFile(filename, 'Show me the contents of this file');
                }
            });
        });

    } catch (error) {
        console.error('Error loading files:', error);
        uploadedFilesList.innerHTML = `
            <div style="color: #dc3545; text-align: center; padding: 20px;">
                Error loading files
            </div>
        `;
    }
}

function askAboutFile(filename, defaultQuery) {
    const query = `${defaultQuery}: ${filename}`;
    chatInput.value = query;
    sendMessage();
}

async function updateStatus() {
    try {
        const response = await fetch('/status');
        const status = await response.json();

        if (status.count > 0) {
            statusIndicator.className = 'status-indicator status-ready';
            statusIndicator.innerHTML = `
                <i class="fas fa-check-circle"></i>
                Ready • ${status.count} documents indexed
            `;
            chatInput.disabled = false;
            sendBtn.disabled = false;
            analyzePdfBtn.disabled = false;
            multiQuoteBtn.disabled = false;
            chatInput.placeholder = "Ask about pricing, multiple items, or complete PDF analysis...";
            
            // Show query suggestions
            querySuggestions.style.display = 'block';
        } else {
            statusIndicator.className = 'status-indicator status-waiting';
            statusIndicator.innerHTML = `
                <i class="fas fa-clock"></i>
                Upload files to get started
            `;
            chatInput.disabled = true;
            sendBtn.disabled = true;
            analyzePdfBtn.disabled = true;
            multiQuoteBtn.disabled = true;
            chatInput.placeholder = "Upload documents first to start chatting...";
            
            // Hide query suggestions
            querySuggestions.style.display = 'none';
        }
    } catch (error) {
        console.error('Error updating status:', error);
        statusIndicator.className = 'status-indicator status-waiting';
        statusIndicator.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            Error connecting to server
        `;
        chatInput.disabled = true;
        sendBtn.disabled = true;
        analyzePdfBtn.disabled = true;
        multiQuoteBtn.disabled = true;
    }
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || chatInput.disabled) return;

    // Add user message to chat
    addMessage('user', message);
    chatInput.value = '';

    // Show typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="loading">
                <div class="spinner"></div>
                Analyzing query and searching documents...
            </div>
        </div>
    `;
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Remove typing indicator
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            chatContainer.removeChild(typingIndicator);
        }

        if (result.success) {
            addMessage('assistant', result.response);
        } else {
            addMessage('assistant', `❌ Error: ${result.message || 'Unknown error occurred'}`);
        }
    } catch (error) {
        console.error('Chat error:', error);
        
        // Remove typing indicator if it exists
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            chatContainer.removeChild(typingIndicator);
        }
        
        addMessage('assistant', '❌ Failed to get response. Please check your connection and try again.');
    }
}

function addMessage(role, content) {
    const message = { role, content, timestamp: new Date() };
    chatHistory.push(message);

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = role === 'assistant' ? '🤖' : '👤';
    
    // Format the content for better display (only for assistant messages)
    const formattedContent = role === 'assistant' ? formatMessageContent(content) : escapeHtml(content);
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${formattedContent}</div>
    `;

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function clearChat() {
    chatHistory = [];
    chatContainer.innerHTML = `
        <div class="message assistant">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <strong>Hello! I'm K&B Scout AI</strong>, your advanced pricing assistant.<br><br>
                I specialize in:<br>
                • <strong>Multi-item pricing</strong> - Get quotes for multiple products at once<br>
                • <strong>PDF pricing analysis</strong> - Complete cost breakdowns of entire documents<br>
                • <strong>Flexible queries</strong> - Ask naturally: "How much for...", "What's the cost of...", etc.<br><br>
                <em>Try: "Price for WRH3027, RH4224, and WRH3624" or "Give me pricing for all items in my PDF"</em>
            </div>
        </div>
    `;
}

async function clearAllData() {
    const confirmMessage = 'Are you sure you want to clear all data? This will remove all uploaded files and chat history.';
    
    if (!confirm(confirmMessage)) {
        return;
    }

    showLoadingOverlay('Clearing all data...');

    try {
        const response = await fetch('/clear', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
            // Clear local state
            selectedFiles = [];
            chatHistory = [];
            uploadedFiles = [];
            fileInput.value = '';
            
            // Update UI
            updateSelectedFiles();
            loadUploadedFiles();
            updateStatus();
            clearChat();
            
            addMessage('assistant', '🗑️ All data has been cleared successfully!');
        } else {
            addMessage('assistant', `❌ Error clearing data: ${result.message || 'Unknown error occurred'}`);
        }
    } catch (error) {
        console.error('Clear data error:', error);
        addMessage('assistant', '❌ Failed to clear data. Please check your connection and try again.');
    } finally {
        hideLoadingOverlay();
    }
}

// Enhanced Features Functions

function showPdfAnalysisModal() {
    const pdfFiles = uploadedFiles.filter(file => file.type === 'pdf');
    
    if (pdfFiles.length === 0) {
        addMessage('assistant', '❌ No PDF files found. Please upload a PDF document first.');
        return;
    }

    // Populate PDF selector
    const pdfSelect = document.getElementById('pdfSelect');
    pdfSelect.innerHTML = '<option value="">Choose a PDF...</option>';
    
    pdfFiles.forEach(file => {
        const option = document.createElement('option');
        option.value = file.name;
        option.textContent = file.name;
        pdfSelect.appendChild(option);
    });

    pdfModal.style.display = 'flex';
}

function showMultiQuoteModal() {
    if (uploadedFiles.length === 0) {
        addMessage('assistant', '❌ No files uploaded. Please upload documents first.');
        return;
    }

    document.getElementById('multiItemInput').value = '';
    multiQuoteModal.style.display = 'flex';
}

async function analyzePdfPricing() {
    const selectedPdf = document.getElementById('pdfSelect').value;
    
    if (!selectedPdf) {
        alert('Please select a PDF file first.');
        return;
    }

    pdfModal.style.display = 'none';
    showLoadingOverlay('Analyzing PDF pricing...');

    try {
        const response = await fetch('/analyze-pdf', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filename: selectedPdf })
        });

        const result = await response.json();

        if (result.success) {
            addMessage('user', `Analyze pricing for all items in: ${selectedPdf}`);
            addMessage('assistant', result.response);
        } else {
            addMessage('assistant', `❌ Error analyzing PDF: ${result.message}`);
        }
    } catch (error) {
        console.error('PDF analysis error:', error);
        addMessage('assistant', '❌ Failed to analyze PDF. Please try again.');
    } finally {
        hideLoadingOverlay();
    }
}

async function generateMultiQuote() {
    const itemsInput = document.getElementById('multiItemInput').value.trim();
    
    if (!itemsInput) {
        alert('Please enter product codes first.');
        return;
    }

    multiQuoteModal.style.display = 'none';
    
    // Format the query for multi-item pricing
    const query = `Please provide pricing and details for these items: ${itemsInput}`;
    
    addMessage('user', query);
    chatInput.value = query;
    sendMessage();
}

// Loading overlay functions
function showLoadingOverlay(text = 'Processing...', subtext = 'Please wait...') {
    document.querySelector('.loading-text').textContent = text;
    document.querySelector('.loading-subtext').textContent = subtext;
    loadingOverlay.style.display = 'flex';
}

function hideLoadingOverlay() {
    loadingOverlay.style.display = 'none';
}

// Enhanced query suggestions
function updateQuerySuggestions() {
    if (uploadedFiles.length === 0) {
        querySuggestions.style.display = 'none';
        return;
    }

    const haspdfs = uploadedFiles.some(file => file.type === 'pdf');
    const suggestions = querySuggestions.querySelectorAll('.suggestion-item');
    
    // Update suggestions based on available files
    suggestions.forEach(item => {
        const query = item.getAttribute('data-query');
        if (query.includes('PDF') && !haspdfs) {
            item.style.display = 'none';
        } else {
            item.style.display = 'flex';
        }
    });
}

// Utility functions for enhanced error handling
function handleError(error, context = '') {
    console.error(`Error in ${context}:`, error);
    
    const errorMessage = `❌ An error occurred${context ? ` in ${context}` : ''}. Please try again.`;
    addMessage('assistant', errorMessage);
}

async function safeFetch(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Fetch error:', error);
        throw error;
    }
}

// Enhanced product code detection for input assistance
function detectProductCodes(text) {
    const productPattern = /\b[A-Z]{2,4}\d{3,5}(?:\s+[A-Z]{1,3})?\b/gi;
    return text.match(productPattern) || [];
}

// Input enhancement for better UX
function enhanceInput() {
    chatInput.addEventListener('input', function(e) {
        const value = e.target.value;
        const codes = detectProductCodes(value);
        
        // Visual feedback for detected product codes
        if (codes.length > 1) {
            chatInput.style.borderColor = '#28a745';
            chatInput.title = `Detected ${codes.length} product codes: ${codes.join(', ')}`;
        } else if (codes.length === 1) {
            chatInput.style.borderColor = '#667eea';
            chatInput.title = `Detected product code: ${codes[0]}`;
        } else {
            chatInput.style.borderColor = '#e9ecef';
            chatInput.title = '';
        }
    });
}

// Keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to send message
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (!chatInput.disabled && chatInput.value.trim()) {
                sendMessage();
            }
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            pdfModal.style.display = 'none';
            multiQuoteModal.style.display = 'none';
        }
    });
}

// Initialize enhanced features
function initializeEnhancedFeatures() {
    enhanceInput();
    setupKeyboardShortcuts();
    updateQuerySuggestions();
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    init();
    initializeEnhancedFeatures();
});

// Also initialize immediately if DOM is already loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        init();
        initializeEnhancedFeatures();
    });
} else {
    init();
    initializeEnhancedFeatures();
}
    