const state = {
  apiUrl: 'http://localhost:8000',
  history: [],
  currentSources: []
};

const chatScroll = document.getElementById('chatScroll');
const messagesEl = document.getElementById('messages');
const welcomeBlock = document.getElementById('welcomeBlock');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const statusEl = document.getElementById('status');
const fileListEl = document.getElementById('fileList');
const drawerEmptyEl = document.getElementById('drawerEmpty');
const drawer = document.getElementById('drawer');
const drawerOverlay = document.getElementById('drawerOverlay');
const drawerToggleBtn = document.getElementById('drawerToggleBtn');
const drawerCloseBtn = document.getElementById('drawerCloseBtn');

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatText(text) {
  return escapeHtml(text).replace(/\n/g, '<br>');
}

function autoResizeTextarea() {
  questionInput.style.height = 'auto';
  questionInput.style.height = Math.min(questionInput.scrollHeight, 180) + 'px';
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatScroll.scrollTop = chatScroll.scrollHeight;
  });
}

function hideWelcome() {
  if (!welcomeBlock.classList.contains('hidden')) {
    welcomeBlock.classList.add('hidden');
  }
}

function renderSources(sources) {
  if (!Array.isArray(sources) || !sources.length) return '';

  const links = sources.map(src => {
    const href = src.download_url || '#';
    const label = src.source || src.file_name || 'Tài liệu';
    return `
      <a class="source-link" href="${href}" target="_blank" rel="noopener noreferrer">
        <span>📄 ${escapeHtml(label)}</span>
      </a>
    `;
  }).join('');

  return `
    <div class="sources">
      <div class="sources-title">Tài liệu liên quan</div>
      <div>${links}</div>
    </div>
  `;
}

function addMessage({ role, text, sources = [], typing = false }) {
  hideWelcome();

  const wrapper = document.createElement('div');
  wrapper.className = 'message';
  if (typing) wrapper.id = 'typingIndicator';

  const avatarClass = role === 'user' ? 'user' : 'assistant';
  const avatarText = role === 'user' ? 'U' : 'AI';
  const label = role === 'user' ? 'Bạn' : 'Trợ lý';

  wrapper.innerHTML = `
    <div class="avatar ${avatarClass}">${avatarText}</div>
    <div class="message-body">
      <div class="message-label">${label}</div>
      <div class="message-text">
        ${typing ? `
          <span class="typing-row">
            <span>Đang phân tích và trả lời</span>
            <span class="typing-dots"><span></span><span></span><span></span></span>
          </span>
        ` : formatText(text)}
      </div>
      ${typing ? '' : renderSources(sources)}
    </div>
  `;

  messagesEl.appendChild(wrapper);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

function setLoadingState(isLoading) {
  sendBtn.disabled = isLoading;
  questionInput.disabled = isLoading;
  sendBtn.textContent = isLoading ? 'Đợi...' : 'Gửi';
}

function normalizeSources(sources) {
  if (!Array.isArray(sources)) return [];
  return sources.filter(Boolean).map(item => ({
    source: item.source_file || item.file_name || 'Tài liệu',
    download_url: item.download_url || '#'
  }));
}

async function sendMessage(prefilledQuestion) {
  const rawQuestion = typeof prefilledQuestion === 'string' ? prefilledQuestion : questionInput.value;
  const question = String(rawQuestion || '').trim();
  if (!question) return;

  addMessage({ role: 'user', text: question });
  questionInput.value = '';
  autoResizeTextarea();
  addMessage({ role: 'assistant', typing: true });
  setLoadingState(true);

  try {
    const base = state.apiUrl.replace(/\/$/, '');
    const response = await fetch(base + '/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });

    const data = await response.json();
    removeTypingIndicator();

    if (!response.ok) {
      addMessage({ role: 'assistant', text: data.detail || 'Có lỗi xảy ra khi xử lý yêu cầu.' });
    } else {
      addMessage({
        role: 'assistant',
        text: data.answer || 'Không có nội dung trả lời.',
        sources: normalizeSources(data.sources)
      });
    }
  } catch (error) {
    removeTypingIndicator();
    addMessage({ role: 'assistant', text: 'Không thể kết nối tới server. Hãy kiểm tra backend và thử lại.' });
  } finally {
    setLoadingState(false);
    questionInput.focus();
  }
}

async function loadFiles() {
  fileListEl.innerHTML = '';
  drawerEmptyEl.classList.add('hidden');

  try {
    const base = state.apiUrl.replace(/\/$/, '');
    const response = await fetch(base + '/api/documents');
    const data = await response.json();
    const items = Array.isArray(data.items) ? data.items : [];

    if (!items.length) {
      drawerEmptyEl.classList.remove('hidden');
      return;
    }

    items.forEach(file => {
      const name = file.file_name || 'Tài liệu';
      const openUrl = file.download_url;
      const downloadUrl = 'http://localhost:8000' + file.download_url;
      const card = document.createElement('div');
      card.className = 'file-card';
      card.innerHTML = `
        <div class="file-name">${escapeHtml(name)}</div>
        <div class="file-actions">
          <a href="${downloadUrl}" target="_blank" rel="noopener noreferrer">Tải xuống</a>
        </div>
      `;
      fileListEl.appendChild(card);
    });
  } catch (error) {
    drawerEmptyEl.textContent = 'Không tải được danh sách tài liệu.';
    drawerEmptyEl.classList.remove('hidden');
  }
}

async function checkHealth() {
  statusEl.classList.remove('online', 'offline', 'loading');
  statusEl.classList.add('loading');
  statusEl.textContent = 'Đang kiểm tra kết nối';

  try {
    const base = state.apiUrl.replace(/\/$/, '');
    const response = await fetch(base + '/api/health', { method: 'GET' });
    const data = await response.json();

    if (data.status === 'ok') {
      statusEl.classList.remove('loading', 'offline');
      statusEl.classList.add('online');
      statusEl.textContent = 'Backend đang hoạt động';
    } else {
      throw new Error('Health check failed');
    }
  } catch (error) {
    statusEl.classList.remove('loading', 'online');
    statusEl.classList.add('offline');
    statusEl.textContent = 'Không kết nối được backend';
  }
}

function openDrawer() {
  drawer.classList.add('open');
  drawerOverlay.classList.add('active');
  drawer.setAttribute('aria-hidden', 'false');
}

function closeDrawer() {
  drawer.classList.remove('open');
  drawerOverlay.classList.remove('active');
  drawer.setAttribute('aria-hidden', 'true');
}

drawerToggleBtn.addEventListener('click', openDrawer);
drawerCloseBtn.addEventListener('click', closeDrawer);
drawerOverlay.addEventListener('click', closeDrawer);

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeDrawer();
});

questionInput.addEventListener('input', autoResizeTextarea);
questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

document.querySelectorAll('.suggestion').forEach(button => {
  button.addEventListener('click', () => {
    const question = button.getAttribute('data-question') || '';
    sendMessage(question);
  });
});

sendBtn.addEventListener('click', () => sendMessage());

autoResizeTextarea();
questionInput.focus();
loadFiles();
checkHealth();