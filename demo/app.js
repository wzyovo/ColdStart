const stageData = {
  zero_shot: {
    recommendations: [
      { name: "川味麻辣香锅", score: 0.82, tags: ["川菜", "夜宵", "高匹配"] },
      { name: "牛肉拌面", score: 0.79, tags: ["面食", "工作餐"] },
      { name: "椒麻鸡饭", score: 0.76, tags: ["川味", "高热度"] }
    ],
    contributions: [
      { label: "川菜标签", value: 0.60 },
      { label: "夜宵场景", value: 0.20 },
      { label: "预算匹配", value: 0.20 }
    ],
    evolution: [
      ["阶段 0", "仅依赖用户圈子、预算、查询词和多标签画像，推荐偏向规则匹配和热门召回。"],
      ["策略", "主要依据口味、场景、价格等静态标签权重进行排序。"]
    ]
  },
  one_shot: {
    recommendations: [
      { name: "川味麻辣香锅", score: 0.87, tags: ["川菜", "点击相似"] },
      { name: "藤椒鸡腿饭", score: 0.84, tags: ["川味", "新偏好"] },
      { name: "椒香牛肉面", score: 0.81, tags: ["点击增强", "工作餐"] }
    ],
    contributions: [
      { label: "首次点击相似度", value: 0.38 },
      { label: "川菜标签", value: 0.34 },
      { label: "预算匹配", value: 0.16 },
      { label: "夜宵场景", value: 0.12 }
    ],
    evolution: [
      ["阶段 1", "系统引入首个点击行为，用户短序列开始影响召回与重排。"],
      ["策略", "静态标签仍然重要，但行为相似性已经开始重写权重。"]
    ]
  },
  three_shot: {
    recommendations: [
      { name: "藤椒鸡腿饭", score: 0.92, tags: ["行为主导", "高转化"] },
      { name: "椒香牛肉面", score: 0.89, tags: ["序列偏好", "复购倾向"] },
      { name: "麻辣烫套餐", score: 0.85, tags: ["协同过滤", "口味稳定"] }
    ],
    contributions: [
      { label: "最近 3 次交互序列", value: 0.44 },
      { label: "协同过滤相似用户", value: 0.24 },
      { label: "川菜标签", value: 0.20 },
      { label: "预算匹配", value: 0.12 }
    ],
    evolution: [
      ["阶段 3", "三次交互后，推荐主导因素从静态画像转向动态行为和相似用户偏好。"],
      ["策略", "LSTM 序列表示与协同过滤信号叠加，规则权重退居辅助角色。"]
    ]
  }
};

const recommendationRoot = document.getElementById("recommendations");
const contributionRoot = document.getElementById("contributions");
const evolutionRoot = document.getElementById("evolution");
const stageButtons = document.querySelectorAll(".stage");

function renderStage(stage) {
  const data = stageData[stage];

  recommendationRoot.innerHTML = data.recommendations.map((item) => `
    <div class="recommendation-card">
      <div class="meta">
        <strong>${item.name}</strong>
        <span>score ${item.score.toFixed(2)}</span>
      </div>
      <div class="tags">
        ${item.tags.map((tag) => `<span class="tag">${tag}</span>`).join("")}
      </div>
    </div>
  `).join("");

  contributionRoot.innerHTML = data.contributions.map((row) => `
    <div class="bar-row">
      <div class="bar-label">
        <span>${row.label}</span>
        <span>${Math.round(row.value * 100)}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${row.value * 100}%"></div>
      </div>
    </div>
  `).join("");

  evolutionRoot.innerHTML = data.evolution.map(([badge, copy]) => `
    <div class="evolution-step">
      <div><span class="step-badge">${badge}</span></div>
      <div>${copy}</div>
    </div>
  `).join("");
}

stageButtons.forEach((button) => {
  button.addEventListener("click", () => {
    stageButtons.forEach((node) => node.classList.remove("active"));
    button.classList.add("active");
    renderStage(button.dataset.stage);
  });
});

renderStage("zero_shot");
