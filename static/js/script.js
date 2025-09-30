// Global state
let selectedFiles = [];
let chatHistory = [];
let uploadedFiles = [];
let sidebarExpanded = true; // Start with sidebar expanded
let filesExpandedSidebar = false;

// DOM elements
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const toggleIcon = document.getElementById('toggleIcon');
const compactUploadBtn = document.getElementById('compactUploadBtn');
const fileInput = document.getElementById('fileInput');
const selectedFilesDiv = document.getElementById('selectedFiles');
const fileList = document.getElementById('fileList');
const processBtn = document.getElementById('processBtn');
const uploadedFilesListSidebar = document.getElementById('uploadedFilesListSidebar');
const filesToggleSidebar = document.getElementById('filesToggleSidebar');
const toggleIconSidebar = document.getElementById('toggleIconSidebar');
const fileCountSidebar = document.getElementById('fileCountSidebar');
const statusIndicator = document.getElementById('statusIndicator');
const chatOutput = document.getElementById('chatOutput');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const chatHistoryList = document.getElementById('chatHistoryList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
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

// Initialize application
function init() {
    setupEventListeners();
    loadUploadedFiles();
    updateStatus();
    loadChatHistory();
}

function setupEventListeners() {
    // Sidebar toggle
    sidebarToggle.addEventListener('click', toggleSidebar);

    // Files toggle in sidebar
    filesToggleSidebar.addEventListener('click', toggleFilesSidebar);

    // Upload functionality
    compactUploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    processBtn.addEventListener('click', processFiles);

    // Chat functionality
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });
    sendBtn.addEventListener('click', sendMessage);

    // History functionality
    clearHistoryBtn.addEventListener('click', clearChatHistory);

    // Drag and drop on chat input for files
    chatInput.addEventListener('dragover', handleDragOver);
    chatInput.addEventListener('drop', handleDrop);
}

function handleDragOver(e) {
    e.preventDefault();
    chatInput.style.borderColor = '#667eea';
}

function handleDrop(e) {
    e.preventDefault();
    chatInput.style.borderColor = '#e9ecef';
    const files = Array.from(e.dataTransfer.files);
    addFilesToSelection(files);
}

function toggleSidebar() {
    sidebarExpanded = !sidebarExpanded;
    if (sidebarExpanded) {
        sidebar.classList.remove('collapsed');
        toggleIcon.className = 'fas fa-chevron-left';
    } else {
        sidebar.classList.add('collapsed');
        toggleIcon.className = 'fas fa-chevron-right';
    }
}

function toggleFilesSidebar() {
    filesExpandedSidebar = !filesExpandedSidebar;
    
    if (filesExpandedSidebar) {
        uploadedFilesListSidebar.classList.add('expanded');
        toggleIconSidebar.classList.add('rotated');
    } else {
        uploadedFilesListSidebar.classList.remove('expanded');
        toggleIconSidebar.classList.remove('rotated');
    }
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


// Add these functions to your existing script.js file

// Enhanced file loading with S3 sync
async function loadUploadedFiles() {
    try {
        const response = await fetch('/files');
        const result = await response.json();
        
        if (!result.success) return;
        
        const files = result.files;
        uploadedFiles = Object.values(files);
        
        fileCountSidebar.textContent = `(${uploadedFiles.length})`;

        if (uploadedFiles.length === 0) {
            uploadedFilesListSidebar.innerHTML = `
                <div style="color: #bdc3c7; text-align: center; padding: 15px; font-size: 12px;">
                    No files uploaded yet
                    <br><br>
                    <button onclick="syncWithS3()" style="background: #667eea; color: white; border: none; padding: 8px 12px; border-radius: 4px; font-size: 11px; cursor: pointer;">
                        <i class="fas fa-sync"></i> Check S3
                    </button>
                </div>
            `;
            return;
        }

        const fileListHTML = uploadedFiles.map((file, index) => {
            const filename = Object.keys(files)[index];
            const extension = getFileExtension(file.type);
            const uploadDate = new Date(file.uploaded_at).toLocaleDateString();
            
            // Determine file status
            let statusIcon = '';
            let statusColor = '#28a745';
            let statusText = 'Ready';
            let actionButton = '';
            
            if (file.status === 'in_s3_only') {
                statusIcon = '<i class="fas fa-cloud"></i>';
                statusColor = '#ffc107';
                statusText = 'In S3 Only';
                actionButton = `<button onclick="reprocessFile('${filename}')" style="background: #ffc107; color: #212529; border: none; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin-top: 4px; cursor: pointer;" title="Reprocess from S3">
                    <i class="fas fa-redo"></i> Process
                </button>`;
            } else if (file.s3_missing) {
                statusIcon = '<i class="fas fa-exclamation-triangle"></i>';
                statusColor = '#dc3545';
                statusText = 'S3 Missing';
            } else if (file.chunks > 0) {
                statusIcon = '<i class="fas fa-check-circle"></i>';
                statusColor = '#28a745';
                statusText = 'Processed';
            }

            return `
                <div class="file-item-sidebar" onclick="askAboutFile('${filename}')">
                    <div class="file-icon-sidebar ${extension}">
                        <i class="fas fa-file-${extension === 'pdf' ? 'pdf' : extension === 'csv' || extension === 'xlsx' ? 'excel' : 'alt'}"></i>
                    </div>
                    <div class="file-info-sidebar">
                        <div class="file-name-sidebar">${filename}</div>
                        <div class="file-size-sidebar">
                            ${file.chunks || 0} chunks • ${uploadDate}
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 4px;">
                            <div style="color: ${statusColor}; font-size: 10px; display: flex; align-items: center; gap: 4px;">
                                ${statusIcon} ${statusText}
                            </div>
                            ${actionButton}
                        </div>
                    </div>
                    <div class="file-actions-sidebar">
                        <button onclick="deleteFile(event, '${filename}')" style="background: #dc3545; color: white; border: none; border-radius: 3px; width: 20px; height: 20px; font-size: 10px; cursor: pointer;" title="Delete file">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            `;
        }).join('');

        // Add sync button at the top
        const syncButton = `
            <div style="text-align: center; padding: 10px; border-bottom: 1px solid #eee; margin-bottom: 10px;">
                <button onclick="syncWithS3()" style="background: #667eea; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-size: 11px; cursor: pointer; margin-right: 8px;">
                    <i class="fas fa-sync"></i> Sync S3
                </button>
                <button onclick="refreshFiles()" style="background: #28a745; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-size: 11px; cursor: pointer;">
                    <i class="fas fa-refresh"></i> Refresh
                </button>
            </div>
        `;

        uploadedFilesListSidebar.innerHTML = syncButton + fileListHTML;

    } catch (error) {
        console.error('Error loading files:', error);
        uploadedFilesListSidebar.innerHTML = `
            <div style="color: #e74c3c; text-align: center; padding: 15px; font-size: 12px;">
                Error loading files
                <br><br>
                <button onclick="syncWithS3()" style="background: #667eea; color: white; border: none; padding: 8px 12px; border-radius: 4px; font-size: 11px; cursor: pointer;">
                    <i class="fas fa-sync"></i> Try Sync S3
                </button>
            </div>
        `;
    }
}

// New function to sync with S3
async function syncWithS3() {
    try {
        showLoadingOverlay('Syncing with S3...');
        
        const response = await fetch('/sync-s3', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const result = await response.json();

        if (result.success) {
            addMessage('assistant', `✅ S3 Sync Complete!\n\n${result.message}\n\nFound ${result.synced_files} new files in S3.`);
            
            // Refresh the file list
            await loadUploadedFiles();
            await updateStatus();
        } else {
            addMessage('assistant', `❌ S3 Sync Failed: ${result.message}`);
        }
    } catch (error) {
        console.error('Sync error:', error);
        addMessage('assistant', 'Failed to sync with S3. Please check your connection and try again.');
    } finally {
        hideLoadingOverlay();
    }
}

// New function to reprocess files from S3
async function reprocessFile(filename) {
    try {
        showLoadingOverlay(`Reprocessing ${filename} from S3...`);
        
        const response = await fetch(`/reprocess/${encodeURIComponent(filename)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const result = await response.json();

        if (result.success) {
            addMessage('assistant', `✅ Successfully reprocessed ${filename}!\n\nGenerated ${result.chunks} chunks for search.`);
            
            // Refresh the file list and status
            await loadUploadedFiles();
            await updateStatus();
        } else {
            addMessage('assistant', `❌ Failed to reprocess ${filename}: ${result.message || result.error}`);
        }
    } catch (error) {
        console.error('Reprocess error:', error);
        addMessage('assistant', `Failed to reprocess ${filename}. Please try again.`);
    } finally {
        hideLoadingOverlay();
    }
}

// Enhanced delete function with event handling
async function deleteFile(event, filename) {
    // Prevent triggering the parent click event
    if (event) {
        event.stopPropagation();
    }
    
    if (!confirm(`Are you sure you want to delete "${filename}"?\n\nThis will remove it from both the database and S3 storage.`)) {
        return;
    }

    try {
        showLoadingOverlay(`Deleting ${filename}...`);
        
        const response = await fetch(`/files/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });

        const result = await response.json();

        if (result.success) {
            addMessage('assistant', `✅ Successfully deleted ${filename}\n\n${result.message}`);
            
            // Refresh the file list and status
            await loadUploadedFiles();
            await updateStatus();
        } else {
            addMessage('assistant', `❌ Failed to delete ${filename}: ${result.message}`);
        }
    } catch (error) {
        console.error('Delete error:', error);
        addMessage('assistant', `Failed to delete ${filename}. Please try again.`);
    } finally {
        hideLoadingOverlay();
    }
}

// New function to refresh files
async function refreshFiles() {
    showLoadingOverlay('Refreshing files...');
    try {
        await loadUploadedFiles();
        await updateStatus();
        addMessage('assistant', '🔄 File list refreshed!');
    } catch (error) {
        addMessage('assistant', 'Failed to refresh files.');
    } finally {
        hideLoadingOverlay();
    }
}

// Enhanced status update with S3 info
async function updateStatus() {
    try {
        const response = await fetch('/files'); 
        const status = await response.json();

        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.placeholder = "Ask questions about your documents...";
        
        if (status.success && status.total_vectors_in_db > 0) {
            statusIndicator.className = 'status-indicator status-ready';
            statusIndicator.innerHTML = `
                <i class="fas fa-check-circle"></i>
                Ready • ${status.total_vectors_in_db} vectors indexed • ${status.total_files} files
                ${status.s3_connected ? '<i class="fas fa-cloud" style="margin-left: 8px;" title="S3 Connected"></i>' : ''}
            `;
        } else if (status.local_files > 0) {
            statusIndicator.className = 'status-indicator status-warning';
            statusIndicator.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                ${status.local_files} files found but may need reprocessing
                ${status.s3_connected ? '<i class="fas fa-cloud" style="margin-left: 8px;" title="S3 Connected"></i>' : ''}
            `;
        } else {
            statusIndicator.className = 'status-indicator status-waiting';
            statusIndicator.innerHTML = `
                <i class="fas fa-cloud-upload-alt"></i>
                Upload files or sync with S3 to get started
                ${status.s3_connected ? '<i class="fas fa-cloud" style="margin-left: 8px;" title="S3 Connected"></i>' : ''}
            `;
        }
    } catch (error) {
        console.error('Error updating status:', error);
        
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.placeholder = "Ask questions about your documents...";
        
        statusIndicator.className = 'status-indicator status-error';
        statusIndicator.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            Error connecting to server
        `;
    }
}

// Add keyboard shortcut for S3 sync
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to send message
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (chatInput.value.trim()) {
                sendMessage();
            }
        }
        
        // Escape to close loading overlay
        if (e.key === 'Escape') {
            hideLoadingOverlay();
        }
        
        // Ctrl/Cmd + U to focus upload button
        if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
            e.preventDefault();
            compactUploadBtn.click();
        }
        
        // F key to toggle files list
        if (e.key === 'f' && !chatInput.matches(':focus')) {
            e.preventDefault();
            toggleFilesSidebar();
        }
        
        // S key to toggle sidebar
        if (e.key === 's' && !chatInput.matches(':focus')) {
            e.preventDefault();
            toggleSidebar();
        }
        
        // Ctrl/Cmd + R to sync with S3
        if ((e.ctrlKey || e.metaKey) && e.key === 'r' && !chatInput.matches(':focus')) {
            e.preventDefault();
            syncWithS3();
        }
    });
}

// Enhanced initialization
function init() {
    setupEventListeners();
    loadUploadedFiles();
    updateStatus();
    loadChatHistory();
    
    // Auto-sync with S3 on page load if no files are present
    setTimeout(async () => {
        try {
            const response = await fetch('/files');
            const result = await response.json();
            
            if (result.success && Object.keys(result.files || {}).length === 0) {
                console.log('No local files found, attempting S3 sync...');
                await syncWithS3();
            }
        } catch (error) {
            console.log('Auto-sync check failed:', error);
        }
    }, 2000); // Wait 2 seconds after page load
}

function updateSelectedFiles() {
    if (selectedFiles.length === 0) {
        selectedFilesDiv.style.display = 'none';
        processBtn.style.display = 'none';
        return;
    }

    selectedFilesDiv.style.display = 'block';
    processBtn.style.display = 'block';

    fileList.innerHTML = selectedFiles.map((file, index) => `
        <div style="display: flex; align-items: center; gap: 10px; padding: 8px; background: #f8f9fa; border-radius: 4px; margin: 4px 0;">
            <span style="font-size: 12px; font-weight: 600;">${file.name}</span>
            <button onclick="removeFile(${index})" style="background: #dc3545; color: white; border: none; border-radius: 50%; width: 20px; height: 20px; font-size: 10px; cursor: pointer;">×</button>
        </div>
    `).join('');
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

    showLoadingOverlay();
    
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
            selectedFiles = [];
            fileInput.value = '';
            updateSelectedFiles();
            loadUploadedFiles();
            updateStatus();
            addMessage('assistant', result.message);
        } else {
            addMessage('assistant', `Error: ${result.message}`);
        }
    } catch (error) {
        addMessage('assistant', 'Failed to upload files. Please try again.');
    } finally {
        hideLoadingOverlay();
    }
}

async function loadUploadedFiles() {
    try {
        const response = await fetch('/files');
        const result = await response.json();
        
        if (!result.success) return;
        
        const files = result.files;
        uploadedFiles = Object.values(files);
        
        fileCountSidebar.textContent = `(${uploadedFiles.length})`;

        if (uploadedFiles.length === 0) {
            uploadedFilesListSidebar.innerHTML = `
                <div style="color: #bdc3c7; text-align: center; padding: 15px; font-size: 12px;">
                    No files uploaded yet
                </div>
            `;
            return;
        }

        const fileListHTML = uploadedFiles.map((file, index) => {
            const filename = Object.keys(files)[index];
            const extension = getFileExtension(file.type);
            const uploadDate = new Date(file.uploaded_at).toLocaleDateString();

            return `
                <div class="file-item-sidebar" onclick="askAboutFile('${filename}')">
                    <div class="file-icon-sidebar ${extension}">
                        <i class="fas fa-file-${extension === 'pdf' ? 'pdf' : extension === 'csv' || extension === 'xlsx' ? 'excel' : 'alt'}"></i>
                    </div>
                    <div class="file-info-sidebar">
                        <div class="file-name-sidebar">${filename}</div>
                        <div class="file-size-sidebar">${file.chunks} chunks • ${uploadDate}</div>
                    </div>
                </div>
            `;
        }).join('');

        uploadedFilesListSidebar.innerHTML = fileListHTML;

    } catch (error) {
        console.error('Error loading files:', error);
        uploadedFilesListSidebar.innerHTML = `
            <div style="color: #e74c3c; text-align: center; padding: 15px; font-size: 12px;">
                Error loading files
            </div>
        `;
    }
}

function getFileExtension(fileType) {
    const typeMap = {
        'PDF Document': 'pdf',
        'CSV Spreadsheet': 'csv',
        'Text Document': 'txt'
    };
    
    if (fileType.includes('Excel')) {
        return 'xlsx';
    }
    
    return typeMap[fileType] || 'file';
}

function askAboutFile(filename) {
    const query = `What information can you find in ${filename}?`;
    chatInput.value = query;
    sendMessage();
}

async function updateStatus() {
    try {
        const response = await fetch('/status');
        const status = await response.json();

        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.placeholder = "Ask questions about your documents...";
        
        if (status.connected && status.count > 0) {
            statusIndicator.className = 'status-indicator status-ready';
            statusIndicator.innerHTML = `
                <i class="fas fa-check-circle"></i>
                Ready • ${status.count} documents indexed • ${status.files} files
            `;
        } else {
            statusIndicator.className = 'status-indicator status-waiting';
            statusIndicator.innerHTML = `
                <i class="fas fa-clock"></i>
                Upload files to get started
            `;
        }
    } catch (error) {
        console.error('Error updating status:', error);
        
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.placeholder = "Ask questions about your documents...";
        
        statusIndicator.className = 'status-indicator status-waiting';
        statusIndicator.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            Error connecting to server
        `;
    }
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

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
            <div style="display: flex; align-items: center; gap: 10px; color: #667eea;">
                <div class="loading-spinner" style="width: 20px; height: 20px; border: 2px solid #f3f3f3; border-top: 2px solid #667eea;"></div>
                Searching documents...
            </div>
        </div>
    `;
    chatOutput.appendChild(typingDiv);
    chatOutput.scrollTop = chatOutput.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        const result = await response.json();

        // Remove typing indicator
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            chatOutput.removeChild(typingIndicator);
        }

        let responseText;
        if (result.success) {
            responseText = result.response;
            addMessage('assistant', responseText);
        } else {
            responseText = `Error: ${result.message || 'Unknown error occurred'}`;
            addMessage('assistant', responseText);
        }

        // Add to chat history with both question and response
        saveChatToHistory(message, responseText);

    } catch (error) {
        console.error('Chat error:', error);
        
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            chatOutput.removeChild(typingIndicator);
        }
        
        const errorResponse = 'Failed to get response. Please check your connection and try again.';
        addMessage('assistant', errorResponse);
        
        // Add to chat history with error response
        saveChatToHistory(message, errorResponse);
    }
}

function addMessage(role, content) {
    const message = { role, content, timestamp: new Date() };
    chatHistory.push(message);

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = role === 'assistant' ? '🤖' : '👤';
    
    // Format the content for better display
    const formattedContent = role === 'assistant' ? formatMessageContent(content) : escapeHtml(content);
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">${formattedContent}</div>
    `;

    chatOutput.appendChild(messageDiv);
    chatOutput.scrollTop = chatOutput.scrollHeight;
}

function formatMessageContent(content) {
    let formatted = content;
    
    // Handle line breaks
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Handle bold text
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Handle bullet points
    formatted = formatted.replace(/^[•\-\*]\s+(.+)$/gm, '<div style="margin: 8px 0; padding-left: 20px; position: relative;"><span style="position: absolute; left: 0; color: #667eea;">•</span>$1</div>');
    
    // Handle numbered lists
    formatted = formatted.replace(/^(\d+)\.\s+(.+)$/gm, '<div style="margin: 8px 0; padding-left: 5px;">$1. $2</div>');
    
    // Enhanced price formatting
    formatted = formatted.replace(/(\$[\d,]+\.?\d*|\$\d+(?:\.\d{2})?)/g, '<span style="color: #155724; font-weight: 700; background: linear-gradient(135deg, #d4edda, #c3e6cb); padding: 3px 8px; border-radius: 6px; border: 1px solid #c3e6cb;">$1</span>');
    
    // Handle product codes
    formatted = formatted.replace(/\b([A-Z]{2,4}\d{3,5}(?:\s+[A-Z]{1,3})?)\b/g, '<span style="color: #495057; font-weight: 600; background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 3px 8px; border-radius: 6px; font-family: monospace; border: 1px solid #dee2e6;">$1</span>');
    
    return formatted;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function saveChatToHistory(message, response) {
    const historyItem = {
        id: Date.now(),
        title: message.length > 50 ? message.substring(0, 50) + '...' : message,
        question: message,
        answer: response,
        timestamp: new Date()
    };

    // Get existing history from localStorage
    let history = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    
    // Add new item to beginning
    history.unshift(historyItem);
    
    // Keep only last 10 items
    history = history.slice(0, 10);
    
    // Save back to localStorage
    localStorage.setItem('chatHistory', JSON.stringify(history));
    
    // Update UI
    loadChatHistory();
}

function loadChatHistory() {
    const history = JSON.parse(localStorage.getItem('chatHistory') || '[]');
    
    if (history.length === 0) {
        chatHistoryList.innerHTML = `
            <div style="color: #bdc3c7; text-align: center; padding: 15px; font-size: 12px;">
                No chat history yet
            </div>
        `;
        return;
    }

    chatHistoryList.innerHTML = history.map(item => `
        <div class="history-item" onclick="loadHistoryItem('${escapeForAttribute(item.question)}', '${escapeForAttribute(item.answer || 'No response recorded')}')">
            <div class="history-title">${item.title}</div>
            <div class="history-time">${formatDate(new Date(item.timestamp))}</div>
        </div>
    `).join('');
}

function loadHistoryItem(question, answer) {
    // Clear current chat and display the historical conversation
    chatOutput.innerHTML = `
        <div class="message assistant">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <strong>Hello! I'm K&B Scout AI</strong>, your enhanced enterprise document assistant.<br><br>
                I can help you find information from your uploaded files. What would you like to know?
            </div>
        </div>
    `;
    
    // Add the historical question and answer
    addMessage('user', question);
    addMessage('assistant', answer);
}

function escapeForAttribute(str) {
    if (!str) return '';
    return str.replace(/'/g, '&#39;')
                 .replace(/"/g, '&quot;')
                 .replace(/\n/g, ' ')
                 .replace(/\r/g, ' ');
}

function clearChatHistory() {
    if (confirm('Are you sure you want to clear chat history?')) {
        localStorage.removeItem('chatHistory');
        loadChatHistory();
    }
}

function formatDate(date) {
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (minutes < 60) {
        return `${minutes}m ago`;
    } else if (hours < 24) {
        return `${hours}h ago`;
    } else {
        return `${days}d ago`;
    }
}

function showLoadingOverlay(text = 'Processing...') {
    document.querySelector('.loading-text').textContent = text;
    loadingOverlay.style.display = 'flex';
}

function hideLoadingOverlay() {
    loadingOverlay.style.display = 'none';
}

// Enhanced input assistance
function detectProductCodes(text) {
    const productPattern = /\b[A-Z]{2,4}\d{3,5}(?:\s+[A-Z]{1,3})?\b/gi;
    return text.match(productPattern) || [];
}

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
            if (chatInput.value.trim()) {
                sendMessage();
            }
        }
        
        // Escape to close loading overlay
        if (e.key === 'Escape') {
            hideLoadingOverlay();
        }
        
        // Ctrl/Cmd + U to focus upload button
        if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
            e.preventDefault();
            compactUploadBtn.click();
        }
        
        // F key to toggle files list
        if (e.key === 'f' && !chatInput.matches(':focus')) {
            e.preventDefault();
            toggleFilesSidebar();
        }
        
        // S key to toggle sidebar
        if (e.key === 's' && !chatInput.matches(':focus')) {
            e.preventDefault();
            toggleSidebar();
        }
    });
}

// Auto-scroll chat container
function autoScrollChat() {
    const isNearBottom = chatOutput.scrollTop + chatOutput.clientHeight >= chatOutput.scrollHeight - 100;
    if (isNearBottom) {
        chatOutput.scrollTop = chatOutput.scrollHeight;
    }
}

// Initialize enhanced features
function initializeEnhancedFeatures() {
    enhanceInput();
    setupKeyboardShortcuts();
    
    // Set up mutation observer for auto-scroll
    const observer = new MutationObserver(autoScrollChat);
    observer.observe(chatOutput, { childList: true, subtree: true });
}

// Performance monitoring
function trackPerformance() {
    // Track page load time
    window.addEventListener('load', () => {
        const loadTime = performance.now();
        console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
    });
    
    // Track API response times
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
        const start = performance.now();
        return originalFetch.apply(this, args).then(response => {
            const end = performance.now();
            console.log(`API call to ${args[0]} took ${(end - start).toFixed(2)}ms`);
            return response;
        });
    };
}

// Error handling and retry mechanism
async function retryOperation(operation, maxRetries = 3, delay = 1000) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await operation();
        } catch (error) {
            if (i === maxRetries - 1) throw error;
            await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)));
        }
    }
}

// Initialize application
function initializeApp() {
    init();
    initializeEnhancedFeatures();
    trackPerformance();
}

// Initialize once document is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
