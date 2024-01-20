import { Octokit } from "octokit";

import { ProjectEntity, PaperImplementationEntity } from "../types";
import ProjectCard from "../../components/project_card";

import data from "./data.json";

const getProjects = (data) => {
  return data.map(project => new ProjectEntity(project));
};

const getOpenSource = (username, token) => {
  // const octokit = Octokit({
  //   auth: token,
  // });
};

const getPaperImplementations = (data) => {
  // return data.map(paper => new PaperImplementationEntity(paper));
  return data.map(paper => new ProjectEntity(paper));
}

export default function ProjectsRoute() {
  const project_list = getProjects(data["projects"]);
  const open_source = getOpenSource(data["github_username"], process.env.GITHUB_TOKEN);

  // todo
  const paper_implementations_list = getPaperImplementations(data["paper_implementations"]);

  return (
    // <!-- Main Content -->
    <main>
      <span className="heading">personal projects</span>
      
      <div className="cards">
        {
          project_list.map((project, index) => {
            return (
              <ProjectCard
                key={index}
                url={project.url}
                name={project.name}
                description={project.description}
                languages={project.languages}
                image={project.image}
                video={project.video}
                index={index}
              />
            )
          })
        }

        {/* <div className="cards__project">
          <a href="https://developer.mozilla.org/en-US/docs/Web/CSS/aspect-ratio">
            <img src='https://images.unsplash.com/photo-1600078686889-8c42747c25fe?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxNDU4OXwwfDF8cmFuZG9tfHx8fHx8fHx8MTY0NDMzMjg5Nw&ixlib=rb-1.2.1&q=80&w=400' alt='Bluetit'>
            <div className="cards__text">
              <h2>test</h2>
              <p>test</p>
            </div>
          </a>
        </div> */}
      </div>

      {/* <span className="heading">open source</span>

      <div className="cards">
        {
          project_list.map((project, index) => {
            return (
              <ProjectCard
                key={index}
                url={project.url}
                name={project.name}
                description={project.description}
                languages={project.languages}
                image={project.image}
                video={project.video}
                index={index}
              />
            )
          })
        }
      </div> */}

      <span className="heading">paper implementations</span>

      <div className="cards">
        {
          paper_implementations_list.map((project, index) => {
            return (
              <ProjectCard
                key={index}
                url={project.url}
                name={project.name}
                description={project.description}
                languages={project.languages}
                image={project.image}
                video={project.video}
                index={index}
              />
            )
          })
        }
      </div>
    </main>
  );
}
