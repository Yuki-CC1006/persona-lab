(function () {
  // Injected by base.html so the product name lives in exactly one place.
  var BRAND = (document.querySelector('meta[name="app-brand"]') || {}).content || "Persona Lab";

  function fillPrompt(text, mode) {
    var input = document.getElementById("prompt-input");
    var modeSelect = document.getElementById("mode-select");
    if (input) {
      input.value = text;
      input.focus();
    }
    if (modeSelect && mode) modeSelect.value = mode;
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function formatAnswer(text) {
    var escaped = escapeHtml(text || "").trim();
    if (!escaped) return "";
    escaped = escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    escaped = escaped.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    return escaped
      .split(/\n\s*\n/)
      .filter(Boolean)
      .map(function (block) {
        var lines = block.split("\n").map(function (ln) { return ln.trim(); }).filter(Boolean);
        if (!lines.length) return "";
        if (lines.every(function (ln) { return /^[-•]\s+/.test(ln); })) {
          return "<ul>" + lines.map(function (ln) {
            return "<li>" + ln.replace(/^[-•]\s+/, "") + "</li>";
          }).join("") + "</ul>";
        }
        return "<p>" + lines.join("<br>") + "</p>";
      })
      .join("");
  }

  function ensureThread() {
    var thread = document.querySelector(".chat-thread");
    if (thread) return thread;
    var empty = document.querySelector(".chat-empty");
    var stage = document.querySelector(".chat-stage");
    thread = el("div", "chat-thread");
    var dock = el("div", "composer-dock");
    var pill = document.querySelector(".prompt-pill");
    var forms = [];
    if (pill && pill.parentNode) {
      Array.prototype.slice.call(pill.parentNode.children).forEach(function (child) {
        forms.push(child);
      });
    }
    if (empty) empty.replaceWith(thread);
    else if (stage) stage.appendChild(thread);
    if (stage) stage.appendChild(dock);
    forms.forEach(function (node) { dock.appendChild(node); });
    var note = el("div", "fineprint", "Say \"remember\" or \"correct\" explicitly to update the profile · expand an answer to see its evidence");
    dock.appendChild(note);
    return thread;
  }

  function appendMsg(role, content) {
    var thread = ensureThread();
    var msg = el("div", "msg " + role);
    var avatar = el("div", "msg-avatar", role === "user" ? "You" : "");
    var body = el("div", "msg-body");
    body.appendChild(el("div", "msg-role", role === "user" ? "You" : BRAND));
    var text = el("div", "msg-text" + (role === "user" ? "" : " prose"));
    if (role === "user") text.textContent = content || "";
    body.appendChild(text);
    msg.appendChild(avatar);
    msg.appendChild(body);
    thread.appendChild(msg);
    thread.scrollTop = thread.scrollHeight;
    return { msg: msg, body: body, text: text, thread: thread };
  }

  function setThreadId(id, title) {
    var form = document.getElementById("chat-form");
    if (!form) return;
    var input = form.querySelector("[name=thread_id]");
    if (!input) {
      input = document.createElement("input");
      input.type = "hidden";
      input.name = "thread_id";
      form.appendChild(input);
    }
    input.value = String(id);
    var next = document.querySelector("#model-form [name=next]");
    if (next) next.value = "/chat/" + id;
    if (window.history && history.replaceState) {
      history.replaceState(null, "", "/chat/" + id);
    }
    if (title) {
      document.title = title + " · " + BRAND;
      upsertHistory(id, title);
    }
  }

  function upsertHistory(id, title) {
    var mid = document.querySelector(".sidebar-mid");
    if (!mid) return;
    var existing = mid.querySelector('.history-row[data-thread-id="' + id + '"]');
    if (existing) {
      var a = existing.querySelector(".history-item");
      if (a) {
        a.textContent = title;
        a.title = title;
      }
      existing.classList.add("active");
      return;
    }
    var label = mid.querySelector(".section-label");
    if (!label) {
      label = el("div", "section-label", "Recent");
      mid.appendChild(label);
    }
    var row = document.createElement("div");
    row.className = "history-row active";
    row.setAttribute("data-thread-id", id);
    var link = el("a", "history-item active", title);
    link.href = "/chat/" + id;
    link.title = title;
    var more = document.createElement("details");
    more.className = "history-more";
    more.innerHTML =
      '<summary title="More" aria-label="More">⋮</summary>' +
      '<div class="history-menu" role="menu">' +
      '<button type="button" class="menu-item" data-rename>Rename</button>' +
      '<form action="/chat/' + id + '/delete" method="post" onsubmit="return confirm(\'Delete this conversation?\')">' +
      '<button type="submit" class="menu-item danger">Delete</button></form></div>';
    row.appendChild(link);
    row.appendChild(more);
    if (label.nextSibling) mid.insertBefore(row, label.nextSibling);
    else mid.appendChild(row);
  }

  function startRename(row) {
    var id = row.getAttribute("data-thread-id");
    var link = row.querySelector(".history-item");
    if (!id || !link || row.querySelector(".history-rename-input")) return;
    var input = el("input", "history-rename-input");
    input.type = "text";
    input.value = (link.textContent || "").trim();
    input.maxLength = 40;
    link.hidden = true;
    row.insertBefore(input, link);
    input.focus();
    input.select();
    var done = false;
    function finish(save) {
      if (done) return;
      done = true;
      var next = (input.value || "").trim() || (link.textContent || "").trim();
      input.remove();
      link.hidden = false;
      if (!save || next === (link.textContent || "").trim()) return;
      link.textContent = next;
      link.title = next;
      fetch("/chat/" + id + "/rename", {
        method: "POST",
        headers: { Accept: "application/json" },
        body: new URLSearchParams({ title: next }),
      })
        .then(function (res) { return res.json(); })
        .then(function (data) {
          if (!data || !data.title) return;
          link.textContent = data.title;
          link.title = data.title;
          var current = document.querySelector("#chat-form [name=thread_id]");
          if (current && current.value === String(id)) {
            document.title = data.title + " · " + BRAND;
          }
        });
    }
    input.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") {
        ev.preventDefault();
        finish(true);
      }
      if (ev.key === "Escape") finish(false);
    });
    input.addEventListener("blur", function () { finish(true); });
  }

  function renderChips(body, chips) {
    if (!chips || !chips.length) return;
    var row = el("div", "meta-chips");
    chips.forEach(function (c) {
      row.appendChild(el("span", "chip " + (c.class || ""), c.text || ""));
    });
    body.appendChild(row);
  }

  function renderEvidence(body, items, label) {
    if (!items || !items.length) return;
    var box = el("details", "evidence");
    var sum = document.createElement("summary");
    sum.textContent = label || ("Referenced " + items.length + " memories");
    box.appendChild(sum);
    var ul = document.createElement("ul");
    items.forEach(function (e) {
      var li = document.createElement("li");
      var id = e.memory_id || e.id;
      li.textContent = (e.content || "").slice(0, 160);
      if (id && (e.kind === "memory" || e.memory_id || !e.kind)) {
        var a = document.createElement("a");
        a.href = "/memories/" + id;
        a.textContent = "Memory #" + id;
        li.prepend(a, document.createTextNode(" · "));
      }
      ul.appendChild(li);
    });
    box.appendChild(ul);
    body.appendChild(box);
  }

  function renderDistill(body, items) {
    if (!items || !items.length) return;
    var box = el("div", "distill-note muted tiny");
    items.forEach(function (d) {
      var p = document.createElement("p");
      var a = document.createElement("a");
      a.href = "/memories/" + d.id;
      a.textContent = "View";
      if (d.action === "replace") {
        p.appendChild(document.createTextNode("Corrected: "));
        if (d.replaced) {
          var s = document.createElement("s");
          s.textContent = d.replaced;
          p.appendChild(s);
          p.appendChild(document.createTextNode(" → "));
        }
        p.appendChild(document.createTextNode(d.content || ""));
      } else if (d.action === "retract") {
        p.appendChild(document.createTextNode("Voided: "));
        var s2 = document.createElement("s");
        s2.textContent = d.replaced || d.content || "";
        p.appendChild(s2);
      } else {
        p.appendChild(document.createTextNode("Written: " + (d.content || "")));
      }
      p.appendChild(document.createTextNode(" "));
      p.appendChild(a);
      box.appendChild(p);
    });
    body.appendChild(box);
  }

  function startChatStream(form) {
    var input = document.getElementById("prompt-input");
    var question = input ? input.value.trim() : "";
    if (!question) return;
    var btn = document.querySelector(".send-btn");
    if (btn) btn.disabled = true;
    if (input) input.value = "";

    var user = appendMsg("user", question);
    var asst = appendMsg("assistant", "");
    asst.msg.classList.add("streaming");
    var status = el("p", "stream-status muted tiny", "Searching memories…");
    var thinking = document.createElement("details");
    thinking.className = "thinking";
    thinking.hidden = true;
    thinking.open = true;
    var sum = document.createElement("summary");
    sum.textContent = "Thinking";
    var thinkBody = el("div", "thinking-body");
    thinking.appendChild(sum);
    thinking.appendChild(thinkBody);
    asst.body.insertBefore(thinking, asst.text);
    asst.body.insertBefore(status, thinking);

    var fd = new FormData(form);
    fd.set("question", question);
    var thinkText = "";
    var answerText = "";

    fetch("/chat/stream", { method: "POST", body: fd, headers: { Accept: "text/event-stream" } })
      .then(function (res) {
        if (!res.ok || !res.body) throw new Error("stream failed");
        var reader = res.body.getReader();
        var decoder = new TextDecoder();
        var buf = "";
        function pump() {
          return reader.read().then(function (result) {
            buf += decoder.decode(result.value || new Uint8Array(), { stream: !result.done });
            var parts = buf.split("\n\n");
            buf = parts.pop();
            parts.forEach(handleSsePart);
            if (result.done) {
              finish();
              return;
            }
            return pump();
          });
        }
        function handleSsePart(part) {
          if (!part.trim()) return;
          var event = "message";
          var dataLines = [];
          part.split("\n").forEach(function (line) {
            if (line.indexOf("event:") === 0) event = line.slice(6).trim();
            else if (line.indexOf("data:") === 0) dataLines.push(line.slice(5).trim());
          });
          if (!dataLines.length) return;
          var data;
          try { data = JSON.parse(dataLines.join("\n")); } catch (err) { return; }
          if (event === "meta") {
            setThreadId(data.thread_id, data.title);
          } else if (event === "status") {
            status.hidden = false;
            status.textContent = data.text || "Thinking…";
          } else if (event === "thinking") {
            status.hidden = true;
            thinking.hidden = false;
            thinkText += data.text || "";
            thinkBody.textContent = thinkText;
            thinkBody.scrollTop = thinkBody.scrollHeight;
          } else if (event === "token") {
            status.hidden = true;
            answerText += data.text || "";
            asst.text.textContent = answerText;
          } else if (event === "done") {
            if (data.thread_id) setThreadId(data.thread_id, data.title);
            thinkText = data.thinking || thinkText;
            answerText = data.answer || answerText;
            if (thinkText) {
              thinking.hidden = false;
              thinking.open = false;
              sum.textContent = "Thinking process";
              thinkBody.textContent = thinkText;
            } else {
              thinking.remove();
            }
            asst.text.innerHTML = formatAnswer(answerText);
            renderChips(asst.body, data.meta_chips);
            renderDistill(asst.body, data.distilled);
            renderEvidence(asst.body, data.evidence);
            renderEvidence(asst.body, data.support, "Referenced memories");
            renderEvidence(asst.body, data.counter, "Counter-evidence");
          } else if (event === "error") {
            asst.text.textContent = data.text || "Something went wrong.";
          }
          user.thread.scrollTop = user.thread.scrollHeight;
        }
        function finish() {
          asst.msg.classList.remove("streaming");
          status.remove();
          if (btn) btn.disabled = false;
          if (input) input.focus();
        }
        return pump().catch(function (err) {
          asst.text.textContent = asst.text.textContent || "Something went wrong while generating the answer.";
          finish();
        });
      })
      .catch(function () {
        asst.msg.classList.remove("streaming");
        status.remove();
        asst.text.textContent = asst.text.textContent || "Could not reach the model; please try again.";
        if (btn) btn.disabled = false;
      });
  }

  document.querySelectorAll("[data-suggest]").forEach(function (el) {
    el.addEventListener("click", function () {
      fillPrompt(el.getAttribute("data-suggest"), el.getAttribute("data-mode") || "ask");
    });
  });

  var form = document.getElementById("chat-form");
  if (form) {
    form.addEventListener("submit", function (e) {
      if (!window.fetch || !window.ReadableStream) return;
      e.preventDefault();
      startChatStream(form);
    });
  }

  // Prefill from query string (?q=&mode=)
  var params = new URLSearchParams(window.location.search);
  var q = params.get("q");
  var mode = params.get("mode");
  if (q) fillPrompt(q, mode || "ask");

  // Import dropzone
  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("file-input");
  var fileName = document.getElementById("file-name");
  var previewBtn = document.getElementById("preview-btn");
  if (dropzone && fileInput) {
    function setFile(file) {
      if (!file) return;
      try {
        var dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
      } catch (e) {
        // Safari fallback: user must use click picker
      }
      if (fileName) {
        fileName.textContent = file.name;
        fileName.classList.add("has-file");
      }
      if (previewBtn) previewBtn.disabled = false;
    }

    fileInput.addEventListener("change", function () {
      if (fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
    });

    ["dragenter", "dragover"].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (evt) {
      dropzone.addEventListener(evt, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", function (e) {
      var files = e.dataTransfer && e.dataTransfer.files;
      if (files && files[0]) setFile(files[0]);
    });
  }

  // Import source switch: upload a file or paste text
  var sourceTabs = document.querySelectorAll("[data-source-tab]");
  var sourcePanes = document.querySelectorAll("[data-source-pane]");
  var sourceInput = document.getElementById("source-input");
  if (sourceTabs.length) {
    sourceTabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        var target = tab.getAttribute("data-source-tab");
        sourceTabs.forEach(function (t) {
          var active = t === tab;
          t.classList.toggle("active", active);
          t.setAttribute("aria-selected", active ? "true" : "false");
        });
        sourcePanes.forEach(function (pane) {
          pane.hidden = pane.getAttribute("data-source-pane") !== target;
        });
        if (sourceInput) sourceInput.value = target === "text" ? "text_chat" : "";
        if (previewBtn) previewBtn.disabled = false;
      });
    });
  }

  var memoryForm = document.getElementById("memory-create-form");
  if (memoryForm) {
    memoryForm.addEventListener("submit", function () {
      var btn = document.getElementById("memory-create-btn");
      if (btn) {
        btn.disabled = true;
        btn.textContent = "Classifying…";
      }
    });
  }

  var selectAll = document.getElementById("select-all");
  var selectedCount = document.getElementById("selected-count");
  function updateSelectedCount() {
    var n = document.querySelectorAll(".cand-check:checked").length;
    if (!selectedCount) return;
    if (n > 0) {
      selectedCount.hidden = false;
      selectedCount.textContent = "Selected " + n;
    } else {
      selectedCount.hidden = true;
    }
  }
  if (selectAll) {
    selectAll.addEventListener("change", function () {
      document.querySelectorAll(".cand-check").forEach(function (el) {
        el.checked = selectAll.checked;
      });
      updateSelectedCount();
    });
  }
  document.querySelectorAll(".cand-check").forEach(function (el) {
    el.addEventListener("change", updateSelectedCount);
  });

  function openTypeFold() {
    if (!location.hash || location.hash.indexOf("#type-") !== 0) return;
    var fold = document.querySelector(location.hash);
    if (fold && fold.tagName === "DETAILS") {
      fold.open = true;
      fold.scrollIntoView({ block: "nearest" });
    }
  }
  openTypeFold();
  window.addEventListener("hashchange", openTypeFold);

  document.addEventListener("toggle", function (e) {
    if (!e.target.classList || !e.target.classList.contains("history-more") || !e.target.open) return;
    document.querySelectorAll(".history-more[open]").forEach(function (other) {
      if (other !== e.target) other.removeAttribute("open");
    });
  }, true);

  document.addEventListener("click", function (e) {
    var renameBtn = e.target.closest && e.target.closest("[data-rename]");
    if (renameBtn) {
      e.preventDefault();
      var row = renameBtn.closest(".history-row");
      var more = renameBtn.closest(".history-more");
      if (more) more.removeAttribute("open");
      if (row) startRename(row);
      return;
    }
    if (!e.target.closest || !e.target.closest(".history-more")) {
      document.querySelectorAll(".history-more[open]").forEach(function (d) {
        d.removeAttribute("open");
      });
    }
  });
})();
