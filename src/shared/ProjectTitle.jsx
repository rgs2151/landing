export default function ProjectTitle({ project, title = project.title }) {
  return <>{project.logo && <img className="pb-project-logo" src={project.logo} alt="" />}{title}</>;
}
