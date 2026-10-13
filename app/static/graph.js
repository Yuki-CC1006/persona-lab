/* Canvas force-directed graph for portrait facets. */
(function () {
  var canvas = document.getElementById("memory-graph");
  var dataEl = document.getElementById("graph-data");
  if (!canvas || !dataEl) return;

  var data;
  try {
    data = JSON.parse(dataEl.textContent || "{}");
  } catch (e) {
    return;
  }

  var rawNodes = data.nodes || [];
  var widthHint = 960;
  var heightHint = 640;
  var cx = widthHint / 2;
  var cy = heightHint / 2;

  var typeNodes = rawNodes.filter(function (n) { return n.kind === "type"; });
  var typeAngle = {};
  typeNodes.forEach(function (t, i) {
    typeAngle[t.group] = (i / Math.max(typeNodes.length, 1)) * Math.PI * 2 - Math.PI / 2;
  });

  var facetByType = {};
  rawNodes.forEach(function (n) {
    if (n.kind === "facet" || n.kind === "overflow") {
      if (!facetByType[n.group]) facetByType[n.group] = [];
      facetByType[n.group].push(n);
    }
  });

  var nodes = rawNodes.map(function (n) {
    var x = cx;
    var y = cy;
    if (n.kind === "subject") {
      x = cx;
      y = cy;
    } else if (n.kind === "type") {
      var a = typeAngle[n.group] || 0;
      x = cx + Math.cos(a) * 160;
      y = cy + Math.sin(a) * 120;
    } else if (n.kind === "facet" || n.kind === "overflow") {
      var ta = typeAngle[n.group] || 0;
      var siblings = facetByType[n.group] || [n];
      var si = siblings.indexOf(n);
      var spread = (si - (siblings.length - 1) / 2) * 0.35;
      var rad = 250;
      x = cx + Math.cos(ta + spread) * rad;
      y = cy + Math.sin(ta + spread) * rad * 0.78;
    }
    return Object.assign({}, n, { x: x, y: y, vx: 0, vy: 0 });
  });

  var nodeMap = {};
  nodes.forEach(function (n) { nodeMap[n.id] = n; });
  var links = (data.links || [])
    .map(function (l) {
      return {
        source: nodeMap[l.source],
        target: nodeMap[l.target],
        kind: l.kind,
        strength: l.strength || 0.3,
      };
    })
    .filter(function (l) { return l.source && l.target; });

  var ctx = canvas.getContext("2d");
  var dpr = window.devicePixelRatio || 1;
  var cssW = canvas.clientWidth || 960;
  var cssH = 640;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  canvas.style.height = cssH + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  var width = cssW;
  var height = cssH;
  var selected = null;
  var dragging = null;
  var offset = { x: 0, y: 0 };
  var scale = 1;
  var pan = { x: 0, y: 0 };
  var detail = document.getElementById("graph-detail");

  function resize() {
    cssW = canvas.parentElement ? canvas.parentElement.clientWidth - 24 : 960;
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    width = cssW;
  }
  resize();
  window.addEventListener("resize", resize);

  function tick() {
    for (var i = 0; i < nodes.length; i++) {
      for (var j = i + 1; j < nodes.length; j++) {
        var a = nodes[i];
        var b = nodes[j];
        var dx = a.x - b.x;
        var dy = a.y - b.y;
        var dist = Math.sqrt(dx * dx + dy * dy) || 1;
        var force = 900 / (dist * dist);
        var fx = (dx / dist) * force;
        var fy = (dy / dist) * force;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }
    }

    links.forEach(function (l) {
      var dx = l.target.x - l.source.x;
      var dy = l.target.y - l.source.y;
      var dist = Math.sqrt(dx * dx + dy * dy) || 1;
      var ideal =
        l.kind === "contains" ? 85 :
        l.kind === "owns" ? 145 :
        110;
      var k = l.strength * 0.08;
      var f = (dist - ideal) * k;
      var fx = (dx / dist) * f;
      var fy = (dy / dist) * f;
      l.source.vx += fx;
      l.source.vy += fy;
      l.target.vx -= fx;
      l.target.vy -= fy;
    });

    nodes.forEach(function (n) {
      n.vx += (width / 2 - n.x) * 0.004;
      n.vy += (height / 2 - n.y) * 0.004;
      if (n.kind === "subject") {
        n.vx += (width / 2 - n.x) * 0.02;
        n.vy += (height / 2 - n.y) * 0.02;
      }
      if (dragging !== n) {
        n.vx *= 0.85;
        n.vy *= 0.85;
        n.x += n.vx;
        n.y += n.vy;
      }
      n.x = Math.max(24, Math.min(width - 24, n.x));
      n.y = Math.max(24, Math.min(height - 24, n.y));
    });
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(scale, scale);

    links.forEach(function (l) {
      ctx.beginPath();
      ctx.moveTo(l.source.x, l.source.y);
      ctx.lineTo(l.target.x, l.target.y);
      if (l.kind === "conflict") {
        ctx.strokeStyle = "rgba(242,139,130,0.7)";
        ctx.setLineDash([5, 4]);
        ctx.lineWidth = 1.4;
      } else if (l.kind === "similar") {
        ctx.strokeStyle = "rgba(253,214,99,0.45)";
        ctx.setLineDash([2, 4]);
        ctx.lineWidth = 1;
      } else if (l.kind === "owns") {
        ctx.strokeStyle = "rgba(232,234,237,0.2)";
        ctx.setLineDash([]);
        ctx.lineWidth = 1.2;
      } else {
        ctx.strokeStyle = "rgba(138,180,248,0.28)";
        ctx.setLineDash([]);
        ctx.lineWidth = 1.1;
      }
      ctx.stroke();
      ctx.setLineDash([]);
    });

    nodes.forEach(function (n) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.size || 8, 0, Math.PI * 2);
      ctx.fillStyle = n.color || "#8ab4f8";
      ctx.globalAlpha = selected && selected !== n ? 0.35 : 1;
      ctx.fill();
      if (selected === n) {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      if (n.has_conflict) {
        ctx.strokeStyle = "rgba(242,139,130,0.95)";
        ctx.lineWidth = selected === n ? 1.5 : 2;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      var label = n.label || "";
      if ((n.kind === "facet" || n.kind === "overflow") && label.length > 12) {
        label = label.slice(0, 11) + "…";
      }
      ctx.fillStyle = "rgba(232,234,237,0.9)";
      ctx.font = n.kind === "subject" ? "600 13px sans-serif" : "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, n.x, n.y + (n.size || 8) + 14);
    });

    ctx.restore();
  }

  function loop() {
    for (var i = 0; i < 2; i++) tick();
    draw();
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  function toWorld(clientX, clientY) {
    var rect = canvas.getBoundingClientRect();
    return {
      x: (clientX - rect.left - pan.x) / scale,
      y: (clientY - rect.top - pan.y) / scale,
    };
  }

  function hit(x, y) {
    for (var i = nodes.length - 1; i >= 0; i--) {
      var n = nodes[i];
      var dx = n.x - x;
      var dy = n.y - y;
      if (dx * dx + dy * dy <= (n.size + 4) * (n.size + 4)) return n;
    }
    return null;
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function showDetail(n) {
    if (!detail || !n) return;
    if (n.kind === "facet") {
      var html =
        "<strong>" + escapeHtml(n.type_label || "") + "</strong>" +
        (n.has_conflict ? " <span class='chip' style='color:#f28b82'>Conflict</span>" : "") +
        "<br/>" + escapeHtml(n.full || n.label) +
        "<br/><span class='tiny'>Confidence " +
        (n.confidence != null ? Number(n.confidence).toFixed(2) : "-") +
        " · " + (n.member_count || (n.memories || []).length || 1) + " memories" +
        " · evidence " + (n.evidence_count || 0) +
        "</span>";
      var mems = n.memories || [];
      if (mems.length) {
        html += "<ul class='graph-mem-list'>";
        mems.forEach(function (m) {
          html +=
            "<li><a href='/memories/" + m.id + "'>" +
            escapeHtml(m.content) +
            "</a>" +
            (m.conflict_group ? " <span class='tiny' style='color:#f28b82'>Conflict</span>" : "") +
            "</li>";
        });
        html += "</ul>";
      }
      detail.innerHTML = html;
      detail.classList.remove("muted");
    } else if (n.kind === "overflow") {
      detail.innerHTML =
        "<strong>" + escapeHtml(n.type_label || "") + "</strong><br/>" +
        escapeHtml(n.full || n.label) +
        (n.href ? "<br/><a href='" + escapeHtml(n.href) + "'>View all →</a>" : "");
      detail.classList.remove("muted");
    } else if (n.kind === "type") {
      detail.textContent =
        "Type: " + (n.full || n.label) +
        " · " + (n.facet_count || 0) + " facets";
      detail.classList.add("muted");
    } else {
      detail.textContent = "Subject: " + (n.full || n.label);
      detail.classList.add("muted");
    }
  }

  canvas.addEventListener("mousedown", function (e) {
    var p = toWorld(e.clientX, e.clientY);
    var n = hit(p.x, p.y);
    if (n) {
      dragging = n;
      selected = n;
      showDetail(n);
      offset.x = p.x - n.x;
      offset.y = p.y - n.y;
    } else {
      selected = null;
      if (detail) {
        detail.textContent = "Click a node to see details";
        detail.classList.add("muted");
      }
      dragging = { pan: true, sx: e.clientX - pan.x, sy: e.clientY - pan.y };
    }
  });
  window.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    if (dragging.pan) {
      pan.x = e.clientX - dragging.sx;
      pan.y = e.clientY - dragging.sy;
      return;
    }
    var p = toWorld(e.clientX, e.clientY);
    dragging.x = p.x - offset.x;
    dragging.y = p.y - offset.y;
    dragging.vx = 0;
    dragging.vy = 0;
  });
  window.addEventListener("mouseup", function () { dragging = null; });

  canvas.addEventListener("wheel", function (e) {
    e.preventDefault();
    var delta = e.deltaY > 0 ? 0.92 : 1.08;
    scale = Math.max(0.5, Math.min(2.2, scale * delta));
  }, { passive: false });

  var resetBtn = document.getElementById("graph-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", function () {
      scale = 1;
      pan.x = 0;
      pan.y = 0;
      // Re-apply layered layout from current width/height
      var tNodes = nodes.filter(function (n) { return n.kind === "type"; });
      var tAngle = {};
      tNodes.forEach(function (t, i) {
        tAngle[t.group] = (i / Math.max(tNodes.length, 1)) * Math.PI * 2 - Math.PI / 2;
      });
      var byT = {};
      nodes.forEach(function (n) {
        if (n.kind === "facet" || n.kind === "overflow") {
          if (!byT[n.group]) byT[n.group] = [];
          byT[n.group].push(n);
        }
      });
      nodes.forEach(function (n) {
        n.vx = 0;
        n.vy = 0;
        if (n.kind === "subject") {
          n.x = width / 2;
          n.y = height / 2;
        } else if (n.kind === "type") {
          var a = tAngle[n.group] || 0;
          n.x = width / 2 + Math.cos(a) * 160;
          n.y = height / 2 + Math.sin(a) * 120;
        } else if (n.kind === "facet" || n.kind === "overflow") {
          var ta = tAngle[n.group] || 0;
          var siblings = byT[n.group] || [n];
          var si = siblings.indexOf(n);
          var spread = (si - (siblings.length - 1) / 2) * 0.35;
          n.x = width / 2 + Math.cos(ta + spread) * 250;
          n.y = height / 2 + Math.sin(ta + spread) * 250 * 0.78;
        }
      });
    });
  }
})();
