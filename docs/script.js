// Set year in footer
document.getElementById('year').textContent = new Date().getFullYear();

// Copy buttons
document.querySelectorAll('.copy').forEach(btn => {
  btn.addEventListener('click', () => {
    const id = btn.getAttribute('data-target');
    const el = document.getElementById(id);
    const text = el ? el.textContent : '';
    navigator.clipboard.writeText(text).then(() => {
      btn.textContent = 'Copied!';
      setTimeout(() => (btn.textContent = 'Copy'), 1200);
    });
  });
});
