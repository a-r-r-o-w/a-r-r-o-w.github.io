let is_dark_mode = localStorage.getItem("is_dark_mode") === null ? true : localStorage.getItem("is_dark_mode") === "true";
const toggleButton = document.getElementById("toggle-button");
const body = document.body;

if (!is_dark_mode) {
  body.classList.toggle("dark-mode");
  body.classList.toggle("light-mode");
}

toggleButton.addEventListener("click", () => {
  is_dark_mode = !is_dark_mode;
  localStorage.setItem("is_dark_mode", is_dark_mode);
  body.classList.toggle("dark-mode");
  body.classList.toggle("light-mode");
});

// create a function load_projects that sends a request to /projects/index.json and parses the response as JSON
const load_projects = () => {
  console.log("load_projects");
  const projectsDiv = document.getElementById("projects");

  fetch("/projects/index.json")
    .then(response => response.json())
    .then(json => json["projects"])
    .then(projects => {
      projects.forEach((project, index) => {
        const projectDiv = document.createElement("div");
        projectDiv.classList.add("cards__project");

        const canPlayVideo = document.createElement("video").canPlayType("video/mp4") !== "";
        let displayContent = `<img class="missing-image-gradient" style="--gradient-angle: ${index % 2 === 0 ? '45deg' : '135deg'};" />`;

        if (project["image"])
          displayContent = `<img src="${project.image}" alt="${project.name}" />`;
        else if (project["video"] && canPlayVideo)
          displayContent = `<video loop muted playsinline src="${project.video}" alt="${project.name}" onmouseover="this.play()" onmouseout="this.pause(); this.currentTime = 0;"> Your browser does not support the video tag. </video>`;

        projectDiv.innerHTML = `
        <a href='${project["url"]}'>
          ${displayContent}
          <div class="cards__text">
            <span id="name">${project["name"]}</span>
            <span id="description">${project["description"]}</span>
            <div class="cards__languages">
              ${project["languages"].map(language => `<span class="cards__language-container">${language}</span>`).join("")}
            </div>
          </div>
        </a>
        `;
        projectsDiv.appendChild(projectDiv);
      });
    });
};
