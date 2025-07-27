// RAGalyze Web Interface JavaScript

class RAGalyzeApp {
    constructor() {
        this.queries = [];
        this.selectedQueryIndex = null;
        this.autoRefreshInterval = null;
        this.websocket = null;
        this.clientId = this.generateClientId();
        this.currentRepoPath = null;
        this.defaultProvider = 'dashscope'; // Fallback default
        
        this.initializeElements();
        this.bindEvents();
        this.loadConfig(); // Load configuration from server
        this.loadInitialData();
        
        // Auto-refresh every 5 seconds
        this.startAutoRefresh();
    }
    
    initializeElements() {
        // Status elements
        this.statusElement = document.getElementById('status');
        this.refreshBtn = document.getElementById('refresh-btn');
        this.queryCountElement = document.getElementById('query-count');
        
        // Sidebar elements
        this.queryListElement = document.getElementById('query-list');
        
        // Content elements
        this.noSelectionElement = document.getElementById('no-selection');
        this.queryDetailsElement = document.getElementById('query-details');
        this.selectedQuestionElement = document.getElementById('selected-question');
        this.selectedRepoElement = document.getElementById('selected-repo');
        this.selectedTimestampElement = document.getElementById('selected-timestamp');
        this.selectedAnswerElement = document.getElementById('selected-answer');
        this.selectedRationaleElement = document.getElementById('selected-rationale');
        this.selectedDocumentsElement = document.getElementById('selected-documents');
        
        // Chat elements
        this.chatInterface = document.getElementById('chat-interface');
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.sendMessageBtn = document.getElementById('send-message-btn');
        this.startChatBtn = document.getElementById('start-chat-btn');
        this.closeChatBtn = document.getElementById('close-chat-btn');
        this.repoPathInput = document.getElementById('repo-path');
        this.chatRepo = document.getElementById('chat-repo');
        this.providerSelect = document.getElementById('provider-select');
        this.modelSelect = document.getElementById('model-select');
    }
    
    bindEvents() {
        this.refreshBtn.addEventListener('click', () => {
            this.loadQueries();
        });
        
        // Chat events
        this.startChatBtn.addEventListener('click', () => {
            this.startChat();
        });
        
        this.closeChatBtn.addEventListener('click', () => {
            this.closeChat();
        });
        
        this.sendMessageBtn.addEventListener('click', () => {
            this.sendMessage();
        });
        
        // Add event handling for Enter key in chat input
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.providerSelect.addEventListener('change', () => this.updateModelSelect());
    }
    
    async loadConfig() {
        this.statusElement.textContent = 'Loading configuration...';
        try {
            const response = await fetch('/api/config');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.config = await response.json();
            this.statusElement.textContent = 'Configuration loaded.';
            this.populateProviderSelect();
        } catch (error) {
            console.error('Error loading config:', error);
            this.statusElement.textContent = 'Error loading configuration.';
            this.showError('Failed to load configuration from the server.');
        }
    }

    populateProviderSelect() {
        if (!this.config || !this.config.providers || this.config.providers.length === 0) {
            this.statusElement.textContent = 'No providers found in configuration.';
            return;
        }
        this.providerSelect.innerHTML = '';
        this.config.providers.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider.name;
            option.textContent = provider.name;
            if (provider.name === this.config.default_provider) {
                option.selected = true;
            }
            this.providerSelect.appendChild(option);
        });
        this.statusElement.textContent = `Populated ${this.config.providers.length} providers.`;
        this.updateModelSelect();
    }

    updateModelSelect() {
        if (!this.config || !this.config.providers) return;
        const selectedProviderName = this.providerSelect.value;
        const selectedProvider = this.config.providers.find(p => p.name === selectedProviderName);

        this.modelSelect.innerHTML = '';
        if (selectedProvider && selectedProvider.models && selectedProvider.models.length > 0) {
            selectedProvider.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = model.name;
                if (model.name === selectedProvider.default_model) {
                    option.selected = true;
                }
                this.modelSelect.appendChild(option);
            });
            this.statusElement.textContent += ` Populated ${selectedProvider.models.length} models for ${selectedProviderName}.`;
        } else {
             this.statusElement.textContent += ` No models found for ${selectedProviderName}.`;
        }
    }

    async loadInitialData() {
        await this.loadServerStatus();
        await this.loadQueries();
    }
    
    async loadServerStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            this.statusElement.textContent = `Server: ${data.status} | Cached repos: ${data.cached_repos.length}`;
            this.statusElement.style.background = 'rgba(34, 197, 94, 0.2)';
        } catch (error) {
            console.error('Failed to load server status:', error);
            this.statusElement.textContent = 'Server: Error';
            this.statusElement.style.background = 'rgba(239, 68, 68, 0.2)';
        }
    }
    
    async loadQueries() {
        try {
            const response = await fetch('/api/queries');
            const data = await response.json();
            
            this.queries = data.queries || [];
            this.updateQueryCount();
            this.renderQueryList();
            
            // If this is the first load and we have queries, select the latest one
            if (this.selectedQueryIndex === null && this.queries.length > 0) {
                this.selectQuery(0);
            }
            
        } catch (error) {
            console.error('Failed to load queries:', error);
            this.showError('Failed to load queries');
        }
    }
    
    updateQueryCount() {
        const count = this.queries.length;
        this.queryCountElement.textContent = count === 0 ? 'No queries' : 
            count === 1 ? '1 query' : `${count} queries`;
    }
    
    renderQueryList() {
        if (this.queries.length === 0) {
            this.queryListElement.innerHTML = `
                <div class="no-queries">
                    <i class="fas fa-comments"></i>
                    <p>No queries yet. Use the client to ask questions!</p>
                    <code>python client.py /path/to/repo "Your question here"</code>
                </div>
            `;
            return;
        }
        
        const queryItems = this.queries.map((query, index) => {
            const timestamp = new Date(query.timestamp).toLocaleString();
            const repoName = this.getRepoName(query.repo_path);
            const question = this.truncateText(query.question, 60);
            
            return `
                <div class="query-item ${index === this.selectedQueryIndex ? 'active' : ''}" 
                     data-index="${index}">
                    <div class="query-item-question">${this.escapeHtml(question)}</div>
                    <div class="query-item-meta">
                        <span class="query-item-repo">${this.escapeHtml(repoName)}</span>
                        <span>${timestamp}</span>
                    </div>
                </div>
            `;
        }).join('');
        
        this.queryListElement.innerHTML = queryItems;
        
        // Bind click events
        this.queryListElement.querySelectorAll('.query-item').forEach(item => {
            item.addEventListener('click', () => {
                const index = parseInt(item.dataset.index);
                this.selectQuery(index);
            });
        });
    }
    
    selectQuery(index) {
        if (index < 0 || index >= this.queries.length) return;
        
        this.selectedQueryIndex = index;
        this.renderQueryList(); // Re-render to update active state
        this.showQueryDetails(this.queries[index]);
    }
    
    showQueryDetails(query) {
        // Hide no-selection, show details
        this.noSelectionElement.style.display = 'none';
        this.queryDetailsElement.classList.remove('hidden');
        
        // Populate query information
        this.selectedQuestionElement.textContent = query.question;
        this.selectedRepoElement.textContent = query.repo_path;
        this.selectedTimestampElement.textContent = new Date(query.timestamp).toLocaleString();
        
        // Populate answer
        this.selectedAnswerElement.textContent = query.answer || 'No answer available';
        
        // Populate rationale
        if (query.rationale && query.rationale.trim()) {
            this.selectedRationaleElement.textContent = query.rationale;
            this.selectedRationaleElement.parentElement.style.display = 'block';
        } else {
            this.selectedRationaleElement.parentElement.style.display = 'none';
        }
        
        // Populate retrieved documents
        this.renderRetrievedDocuments(query.relevant_documents || []);
    }
    
    renderRetrievedDocuments(documents) {
        if (documents.length === 0) {
            this.selectedDocumentsElement.innerHTML = `
                <div class="no-documents">
                    <p>No relevant documents found for this query.</p>
                </div>
            `;
            return;
        }
        
        const documentCards = documents.map((doc, index) => {
            const fileName = this.getFileName(doc.file_path);
            const content = doc.content_preview || 'No content preview available';
            
            return `
                <div class="document-card">
                    <div class="document-header">
                        <i class="fas fa-file-code"></i>
                        <span class="document-path">${this.escapeHtml(doc.file_path)}</span>
                    </div>
                    <div class="document-content">${this.escapeHtml(content)}</div>
                </div>
            `;
        }).join('');
        
        this.selectedDocumentsElement.innerHTML = documentCards;
    }
    
    startAutoRefresh() {
        this.autoRefreshInterval = setInterval(() => {
            this.loadQueries();
        }, 5000); // Refresh every 5 seconds
    }
    
    stopAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
        }
    }
    
    // Chat methods
    startChat() {
        const repoPath = this.repoPathInput.value.trim();
        if (!repoPath) {
            this.showError('Please enter a repository path');
            return;
        }
        
        this.currentRepoPath = repoPath;
        this.chatRepo.textContent = repoPath;
        
        // Hide other views, show chat interface
        this.noSelectionElement.style.display = 'none';
        this.queryDetailsElement.classList.add('hidden');
        this.chatInterface.classList.remove('hidden');
        
        // Clear previous messages
        this.chatMessages.innerHTML = '';
        
        // Connect WebSocket
        this.connectWebSocket();
    }
    
    closeChat() {
        // Close WebSocket connection
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        // Hide chat interface, show no selection
        this.chatInterface.classList.add('hidden');
        this.noSelectionElement.style.display = 'flex';
        this.currentRepoPath = null;
    }
    
    connectWebSocket() {
        // Close existing connection if any
        if (this.websocket) {
            this.websocket.close();
        }
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Add a timestamp to bypass caching
        console.log(`protocol: ${protocol}`)
        const wsUrl = `${protocol}//${window.location.host}/chat`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connection established');
            this.addSystemMessage('Connected to server');
        };
        
        this.websocket.onmessage = (event) => {
            try {
                // Try to parse as JSON first
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (e) {
                // If it's not JSON, treat it as a streaming text response
                console.log('Received streaming text:', event.data);
                this.handleStreamingResponse(event.data);
            }
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket connection closed');
            this.addSystemMessage('Disconnected from server');
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError('WebSocket connection error');
        };
    }
    
    handleWebSocketMessage(data) {
        console.log('Received WebSocket message:', data);
        
        if (data.type === 'history') {
            // Handle chat history
            this.chatMessages.innerHTML = '';
            data.history.forEach(msg => {
                const messageElement = this.createMessageElement(msg.role, msg.content);
                this.chatMessages.appendChild(messageElement);
            });
        } else if (data.type === 'response') {
            // Handle assistant response
            const messageElement = this.createMessageElement('assistant', data.content || data.answer);
            this.chatMessages.appendChild(messageElement);
            
            // Scroll to bottom
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        } else if (data.type === 'streaming') {
            // Handle streaming response
            this.handleStreamingResponse(data.content);
        } else if (data.type === 'error') {
            // Handle error
            this.showError(data.message);
            this.addSystemMessage(`Error: ${data.message}`);
        }
    }
    
    sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || !this.currentRepoPath || !this.websocket) return;

        // Add user message to chat
        const messageElement = this.createMessageElement('user', message);
        this.chatMessages.appendChild(messageElement);

        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;

        const selectedProvider = this.providerSelect.value;
        const selectedModel = this.modelSelect.value;

        // Send message to server
        this.websocket.send(JSON.stringify({
            repo_path: this.currentRepoPath,
            messages: [
                { role: 'user', content: message }
            ],
            provider: selectedProvider,
            model: selectedModel
        }));

        // Clear input
        this.chatInput.value = '';
    }
    
    createMessageElement(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}-message`;
        
        if (role === 'assistant' && content) {
            // For assistant messages, use a pre element to preserve formatting
            const preElement = document.createElement('pre');
            preElement.style.whiteSpace = 'pre-wrap';
            preElement.style.margin = '0';
            preElement.style.fontFamily = 'inherit';
            preElement.textContent = content;
            messageDiv.appendChild(preElement);
        } else {
            // For user and system messages, use regular text
            messageDiv.textContent = content;
        }
        
        return messageDiv;
    }
    
    addSystemMessage(message) {
        const systemDiv = document.createElement('div');
        systemDiv.className = 'chat-message system-message';
        systemDiv.textContent = message;
        this.chatMessages.appendChild(systemDiv);
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substring(2, 15);
    }
    
    handleStreamingResponse(text) {
        // Check if this is a retrieved document section
        if (text.includes('[Retrieved Document]')) {
            // Create or get the documents container
            let docsContainer = this.chatMessages.querySelector('.retrieved-documents-container');
            
            if (!docsContainer) {
                // Create a new container for retrieved documents
                docsContainer = document.createElement('div');
                docsContainer.className = 'retrieved-documents-container';
                docsContainer.innerHTML = '<h4>Retrieved Documents:</h4>';
                this.chatMessages.appendChild(docsContainer);
            }
            
            // Add the document to the container
            const docElement = document.createElement('div');
            docElement.className = 'retrieved-document';
            docElement.innerHTML = `<pre>${this.escapeHtml(text)}</pre>`;
            docsContainer.appendChild(docElement);
        } else {
            // Regular response handling
            // Check if we already have an assistant message element
            let messageElement = this.chatMessages.querySelector('.assistant-message:last-child');
            
            if (!messageElement) {
                // Create a new message element if none exists
                messageElement = this.createMessageElement('assistant', '');
                this.chatMessages.appendChild(messageElement);
            }
            
            // Create a pre element to preserve formatting if it doesn't exist
            let preElement = messageElement.querySelector('pre');
            if (!preElement) {
                preElement = document.createElement('pre');
                preElement.style.whiteSpace = 'pre-wrap';
                preElement.style.margin = '0';
                preElement.style.fontFamily = 'inherit';
                messageElement.textContent = ''; // Clear the text content
                messageElement.appendChild(preElement);
            }
            
            // Append the new text to the pre element
            preElement.textContent += text;
        }
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    // Utility functions
    getRepoName(repoPath) {
        return repoPath.split('/').pop() || repoPath;
    }
    
    getFileName(filePath) {
        return filePath.split('/').pop() || filePath;
    }
    
    truncateText(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showError(message) {
        console.error(message);
        // You could show a toast notification here
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.RAGalyzeApp = new RAGalyzeApp();
});

// Handle page visibility changes to pause/resume auto-refresh
document.addEventListener('visibilitychange', () => {
    if (window.RAGalyzeApp) {
        if (document.hidden) {
            window.RAGalyzeApp.stopAutoRefresh();
        } else {
            window.RAGalyzeApp.startAutoRefresh();
        }
    }
});