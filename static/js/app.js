// RAGalyze Web Interface JavaScript

class RAGalyzeApp {
    constructor() {
        this.queries = [];
        this.selectedQueryIndex = null;
        this.autoRefreshInterval = null;
        
        this.initializeElements();
        this.bindEvents();
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
    }
    
    bindEvents() {
        this.refreshBtn.addEventListener('click', () => {
            this.loadQueries();
        });
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