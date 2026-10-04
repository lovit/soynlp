"use strict";

const article = document.querySelector(".document");
const toc = document.querySelector(".page-toc");
const tocList = document.querySelector("#toc-list");

if (article && toc && tocList) {
  const headings = article.querySelectorAll("h2[id], h3[id]");
  for (const heading of headings) {
    const item = document.createElement("li");
    if (heading.tagName === "H3") item.className = "toc-subheading";
    const link = document.createElement("a");
    link.href = `#${heading.id}`;
    link.textContent = heading.textContent;
    item.append(link);
    tocList.append(item);
  }
  toc.hidden = headings.length === 0;

  const tables = article.querySelectorAll("table");
  for (const table of tables) {
    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", "표, 필요한 경우 가로로 스크롤");
    table.before(wrapper);
    wrapper.append(table);
  }

  if (navigator.clipboard) {
    for (const block of article.querySelectorAll("div.highlighter-rouge")) {
      const code = block.querySelector("pre code");
      if (!code) continue;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "copy-button";
      button.textContent = "복사";
      button.setAttribute("aria-label", "코드 복사");
      button.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(code.textContent);
          button.textContent = "복사됨";
        } catch {
          button.textContent = "직접 선택해 주세요";
        }
        window.setTimeout(() => {
          button.textContent = "복사";
        }, 2000);
      });
      block.append(button);
    }
  }
}
