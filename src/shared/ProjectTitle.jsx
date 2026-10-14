export default function ProjectTitle({ project }) {
  return <>{project.logo && <img className="pb-project-logo" src={project.logo} alt="" />}{project.title}</>;
}
