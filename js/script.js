let is_dark_mode = true;
const toggleButton = document.getElementById("toggle-button");
const body = document.body;

toggleButton.addEventListener("click", () => {
  is_dark_mode = !is_dark_mode;
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

        const displayImg = project.image
        ? `<img src="${project.image}" alt="${project.name}" />`
        : `<img class="missing-image-gradient" style="--gradient-angle: ${index % 2 === 0 ? '45deg' : '135deg'};" />`;

        projectDiv.innerHTML = `
        <a href='${project["url"]}'>
          ${displayImg}
          <div class="cards__text">
            <h2>${project["name"]}</h2>
            <p>${project["short_description"]}</p>
          </div>
        </a>
        `;
        projectsDiv.appendChild(projectDiv);
      });
    });
};
