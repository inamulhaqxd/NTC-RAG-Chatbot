const messagesContainer = document.getElementById('messages');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const welcomeMessage = document.getElementById('welcomeMessage');
const loadingOverlay = document.getElementById('loadingOverlay');

let chatHistory = [];

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

function newChat() {
  chatHistory = [];
  messagesContainer.innerHTML = '';
  welcomeMessage.style.display = 'flex';
}

async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question) return;

  welcomeMessage.style.display = 'none';
  
  addMessage('user', question);
  questionInput.value = '';
  questionInput.style.height = 'auto';
  
  const typingId = addTypingIndicator();
  sendBtn.disabled = true;

  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });

    if (!response.ok) {
      throw new Error('Failed to get answer');
    }

    const data = await response.json();
    removeTypingIndicator(typingId);
    addMessage('assistant', data.answer);
  } catch (error) {
    removeTypingIndicator(typingId);
    addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
    showToast('Error: ' + error.message, 'error');
  } finally {
    sendBtn.disabled = false;
    questionInput.focus();
  }
}

function addMessage(role, content) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;
  
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = role === 'user' ? 'U' : 'AI';
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'message-content';
  contentDiv.innerHTML = formatMessage(content);
  
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(contentDiv);
  messagesContainer.appendChild(messageDiv);
  
  scrollToBottom();
  chatHistory.push({ role, content });
}

function formatMessage(text) {
  return marked.parse(text);
}

function addTypingIndicator() {
  const id = 'typing-' + Date.now();
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message assistant';
  messageDiv.id = id;
  
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = 'AI';
  
  const typingDiv = document.createElement('div');
  typingDiv.className = 'message-content';
  typingDiv.innerHTML = `
    <div class="typing-indicator">
      <span></span><span></span><span></span>
    </div>
  `;
  
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(typingDiv);
  messagesContainer.appendChild(messageDiv);
  
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  const element = document.getElementById(id);
  if (element) element.remove();
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function uploadFile(input) {
  const file = input.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  loadingOverlay.classList.add('active');

  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error('Upload failed');
    }

    const data = await response.json();
    showToast(`Uploaded: ${data.chunks} chunks indexed`, 'success');
  } catch (error) {
    showToast('Upload failed: ' + error.message, 'error');
  } finally {
    loadingOverlay.classList.remove('active');
    input.value = '';
  }
}

function showToast(message, type = 'info') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => toast.remove(), 3000);
}

questionInput.focus();
