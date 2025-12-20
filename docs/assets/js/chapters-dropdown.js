document.addEventListener("click", (e) => {
  document.querySelectorAll(".chapters-dd[open]").forEach((dd) => {
    if (!dd.contains(e.target)) dd.removeAttribute("open");
  });

  const link = e.target.closest(".chapters-menu a");
  if (link) {
    const dd = link.closest(".chapters-dd");
    if (dd) dd.removeAttribute("open");
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    document.querySelectorAll(".chapters-dd[open]").forEach((dd) => {
      dd.removeAttribute("open");
    });
  }
});
